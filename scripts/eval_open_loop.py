#!/usr/bin/env python3
"""Open-loop eval: LeRobot policy on RB-Y1 with a choice of **vision** source and **live** proprio.

``--proprio-source episode``: RGB from paths in ``--episode-dir`` JSON; ``observation.state`` is
**always** current robot joints + gripper. For **PI0.5**, one observation per ``predict_action_chunk``
matches training: we use the frame at the current episode index for inference, then advance that index
by the number of enqueued actions (same as ``--pi05-actions-per-chunk``, default 50), **not** one JSON
frame per control tick.

``--proprio-source live``: live RealSense cameras (same as ``eval.py``) + current joints.

**PI0.5** embeds discretized state in the language prompt; recorded images + live joints can still
mismatch visually and cause jitter.

Commands use the same SDK path as ``eval.py``: ``data_utils.data_eval.send_predicted_action16``."""

from __future__ import annotations

import atexit
import importlib.util
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

import numpy as np
import torch
import tyro
from omegaconf import OmegaConf
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_LEROBOT_SRC = _REPO_ROOT / "lerobot" / "src"
for _p in (_REPO_ROOT, _SCRIPTS_DIR, _LEROBOT_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from camera.realsense_camera import RealSenseCamera, get_device_ids  # noqa: E402
from data_utils.data_eval import send_predicted_action16  # noqa: E402
from data_utils.logging_utils import log_collect_demos  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.processor import PolicyAction, PolicyProcessorPipeline  # noqa: E402

from gripper import Gripper  # noqa: E402
from replay import (  # noqa: E402
    SystemContext,
    connect_gripper_for_replay,
    connect_rby1,
    move_robot_to_ready_pose,
    wait_for_joint_state,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s")

DEVICE = os.environ.get("LEROBOT_TEST_DEVICE", "cuda") if torch.cuda.is_available() else "cpu"

cleanup_in_progress = False
_cleanup_robot = None
_cleanup_no_head = False
_cleanup_gripper: Optional[Gripper] = None


def _cleanup() -> None:
    global cleanup_in_progress
    if cleanup_in_progress:
        return
    cleanup_in_progress = True
    if _cleanup_robot is not None:
        logging.info("Eval open-loop cleanup: moving to ready pose...")
        try:
            move_robot_to_ready_pose(_cleanup_robot, _cleanup_no_head, _cleanup_gripper)
        except Exception as exc:
            logging.warning("Cleanup ready-pose failed: %s", exc)
    logging.info("Cleanup completed.")


def _signal_handler(signum, frame) -> None:  # noqa: ARG001
    _cleanup()
    os._exit(0)


@dataclass
class Args:
    config_path: str = str(_REPO_ROOT / "config.yaml")
    """Path to project config YAML (sensors, storage, policy)."""

    episode_dir: str = str(_REPO_ROOT / "data" / "close_the_oven" / "000001")
    """Directory containing ``<id>/<id>.json`` and ``left_rgb``, ``front_rgb``, ``right_rgb`` images."""

    task: Optional[str] = None
    """Language instruction (default: ``storage.language_instruction`` in YAML)."""

    control_hz: Optional[float] = None
    """Control loop rate in Hz (default: ``lerobot.fps`` or 30)."""

    loop_episode: bool = False
    """If True, wrap image index after the last frame; if False, stop when the episode ends."""

    proprio_source: str = "episode"
    """``episode``: RGB from paths in ``--episode-dir`` JSON + **current** robot joints (default
    open-loop). ``live``: live RealSense cameras + current joints (same vision path as ``eval.py``)."""

    rby1: str = "192.168.30.1:50051"
    rby1_model: str = "m"
    no_head: bool = False
    no_gripper: bool = False

    pi05_actions_per_chunk: int = 50
    """Clip for each ``predict_action_chunk``; in ``--proprio-source episode`` mode the saved episode
    index advances by the number of actions actually enqueued (≤ this, and ≤ model horizon)."""

    pi05_queue_threshold_frac: float = 0.0


def _suffix_from_image_feature_key(key: str) -> str:
    if "camera_" not in key:
        raise ValueError(f"Unexpected image feature key: {key}")
    return key.split("camera_", 1)[1]


def _list_dataset_image_keys(ds_meta: LeRobotDatasetMetadata) -> list[str]:
    keys = [k for k in ds_meta.features if k.startswith("observation.images.")]
    return sorted(keys)


def _numpy_camera_by_suffix(
    left_rgb: np.ndarray,
    front_rgb: np.ndarray,
    right_rgb: np.ndarray,
    suffix: str,
) -> np.ndarray:
    if suffix == "left":
        return left_rgb
    if suffix == "front":
        return front_rgb
    if suffix == "right":
        return right_rgb
    raise KeyError(f"Unknown camera suffix {suffix!r} (expected left, front, right)")


def _resize_hw_from_feature(ds_meta: LeRobotDatasetMetadata, key: str) -> tuple[int, int]:
    shape = tuple(ds_meta.features[key]["shape"])
    if len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    return 360, 640


def build_raw_observation(
    ds_meta: LeRobotDatasetMetadata,
    state_16: np.ndarray,
    left_rgb: np.ndarray,
    front_rgb: np.ndarray,
    right_rgb: np.ndarray,
    device: str,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    st = torch.from_numpy(np.asarray(state_16, dtype=np.float32).reshape(1, -1)).to(device)
    out["observation.state"] = st

    for key in _list_dataset_image_keys(ds_meta):
        suf = _suffix_from_image_feature_key(key)
        img_np = _numpy_camera_by_suffix(left_rgb, front_rgb, right_rgb, suf)
        h, w = _resize_hw_from_feature(ds_meta, key)
        img_pil = Image.fromarray(img_np)
        img_pil = img_pil.resize((w, h))
        arr = np.array(img_pil)
        t = torch.from_numpy(arr).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0).to(device)
        out[key] = t
    return out


def _resolve_joint_indices(robot) -> tuple[np.ndarray, np.ndarray]:
    joint_names = list(robot.model().robot_joint_names)
    left_list = [i for i, n in enumerate(joint_names) if str(n).startswith("left_arm")]
    right_list = [i for i, n in enumerate(joint_names) if str(n).startswith("right_arm")]
    left_list.sort(key=lambda i: joint_names[i])
    right_list.sort(key=lambda i: joint_names[i])
    return np.asarray(left_list, dtype=np.int64), np.asarray(right_list, dtype=np.int64)


def extract_left_right_16(
    q_full: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    gripper: Optional[Gripper],
) -> np.ndarray:
    left_q = q_full[left_idx][:7]
    right_q = q_full[right_idx][:7]
    if gripper is not None:
        g = gripper.get_normalized_target().copy()
        raw_l = float(np.clip(g[1], 0.0, 1.0))
        raw_r = float(np.clip(g[0], 0.0, 1.0))
        left_g = float(np.clip(1.0 - raw_l, 0.0, 1.0))
        right_g = float(np.clip(1.0 - raw_r, 0.0, 1.0))
    else:
        left_g, right_g = 0.0, 0.0
    left8 = np.concatenate([left_q, np.array([left_g], dtype=np.float32)])
    right8 = np.concatenate([right_q, np.array([right_g], dtype=np.float32)])
    return np.concatenate([left8, right8]).astype(np.float32)


def _resolve_image_path(path_str: str, episode_dir: Path) -> Path:
    p = Path(path_str).expanduser()
    if p.is_file():
        return p
    alt = episode_dir / p.name
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"Image not found: {path_str} (tried {p} and {alt})")


def load_episode_rgb_frames(episode_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Load left/front/right RGB arrays from episode JSON image path fields."""
    episode_dir = episode_dir.resolve()
    json_files = sorted(episode_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No episode JSON (*.json) in {episode_dir}")

    with open(json_files[0], encoding="utf-8") as f:
        steps = json.load(f)
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"Expected non-empty list in {json_files[0]}")

    left_list: list[np.ndarray] = []
    front_list: list[np.ndarray] = []
    right_list: list[np.ndarray] = []

    for i, step in enumerate(steps):
        for key in ("image_left_rgb", "image_front_rgb", "image_right_rgb"):
            if key not in step:
                raise KeyError(f"Step {i} missing {key} in {json_files[0]}")
        lp = _resolve_image_path(str(step["image_left_rgb"]), episode_dir)
        fp = _resolve_image_path(str(step["image_front_rgb"]), episode_dir)
        rp = _resolve_image_path(str(step["image_right_rgb"]), episode_dir)
        left_list.append(np.array(Image.open(lp).convert("RGB")))
        front_list.append(np.array(Image.open(fp).convert("RGB")))
        right_list.append(np.array(Image.open(rp).convert("RGB")))

    logging.info(
        "Loaded episode %s: %d frames from %s",
        episode_dir.name,
        len(left_list),
        json_files[0].name,
    )
    return left_list, front_list, right_list


@dataclass
class EpisodeFrameSource:
    """Sequential (optionally looping) RGB triplets from a saved episode."""

    left: list[np.ndarray]
    front: list[np.ndarray]
    right: list[np.ndarray]
    loop: bool = True
    idx: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        n = len(self.left)
        if n == 0 or len(self.front) != n or len(self.right) != n:
            raise ValueError("Episode frame lists must be non-empty and equal length.")

    def next_triplet(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance by one frame (used by diffusion open-loop: one image per control step)."""
        n = len(self.left)
        if not self.loop and self.idx >= n:
            raise StopIteration
        i = self.idx % n
        self.idx += 1
        return self.left[i], self.front[i], self.right[i]

    def peek_triplet(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Current frame at ``idx`` without advancing (PI0.5: one obs per chunk)."""
        n = len(self.left)
        if not self.loop and self.idx >= n:
            raise StopIteration
        i = self.idx % n if self.loop else self.idx
        return self.left[i], self.front[i], self.right[i]

    def advance_by(self, k: int) -> None:
        """Move episode index forward by ``k`` (typically ``n`` actions enqueued from one chunk)."""
        if k < 0:
            raise ValueError("advance_by expects k >= 0")
        self.idx += k

    @property
    def num_frames(self) -> int:
        return len(self.left)


def _cancel_command_stream(stream: Any) -> None:
    try:
        stream.cancel()
    except Exception:
        pass


def _read_rgb_triplet(
    proprio_source: str,
    frames: Optional[EpisodeFrameSource],
    cameras: Optional[dict[str, RealSenseCamera]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vision: ``episode`` = next frame from saved episode; ``live`` = RealSense (same as ``eval.py``)."""
    if proprio_source == "episode":
        if frames is None:
            raise RuntimeError("episode mode requires loaded episode frames")
        return frames.next_triplet()
    if proprio_source == "live":
        if cameras is None:
            raise RuntimeError("live mode requires RealSense cameras")
        left_rgb, _ = cameras["left_camera"].read()
        front_rgb, _ = cameras["front_camera"].read()
        right_rgb, _ = cameras["right_camera"].read()
        return left_rgb, front_rgb, right_rgb
    raise ValueError(f"proprio_source must be 'live' or 'episode', got {proprio_source!r}")


def _send_action_or_refresh_stream(
    robot: Any,
    stream: Any,
    act_np: np.ndarray,
    dt: float,
    gripper: Optional[Gripper],
) -> Any:
    try:
        send_predicted_action16(stream, act_np, dt, gripper)
        return stream
    except RuntimeError as e:
        msg = str(e).lower()
        if "expired" not in msg:
            raise
        logging.warning("Command stream expired; recreating and retrying send.")
        _cancel_command_stream(stream)
        stream = robot.create_command_stream()
        send_predicted_action16(stream, act_np, dt, gripper)
        return stream


def run_diffusion_eval(
    robot,
    policy: DiffusionPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    ds_meta: LeRobotDatasetMetadata,
    frames: Optional[EpisodeFrameSource],
    cameras: Optional[dict[str, RealSenseCamera]],
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    gripper: Optional[Gripper],
    dt: float,
    device: str,
    proprio_source: str,
) -> None:
    policy.reset()
    stream = robot.create_command_stream()
    try:
        while True:
            t0 = time.time()
            if SystemContext.latest_joint_positions.size == 0:
                time.sleep(0.01)
                continue

            try:
                left_rgb, front_rgb, right_rgb = _read_rgb_triplet(proprio_source, frames, cameras)
            except StopIteration:
                logging.info("Episode finished (loop_episode=False). Stopping.")
                break

            state_16 = extract_left_right_16(SystemContext.latest_joint_positions, left_idx, right_idx, gripper)
            raw = build_raw_observation(ds_meta, state_16, left_rgb, front_rgb, right_rgb, device)
            raw = {k: raw[k].to(device, non_blocking=True) for k in raw}
            _cancel_command_stream(stream)
            batch = preprocessor(raw)
            t_inf0 = time.time()
            actions = policy.select_action(batch)
            actions = postprocessor(actions)
            act_np = actions.squeeze(0).detach().cpu().numpy()
            log_collect_demos(f"diffusion inference {(time.time() - t_inf0) * 1000:.1f} ms", "info")
            stream = robot.create_command_stream()

            stream = _send_action_or_refresh_stream(robot, stream, act_np, dt, gripper)

            elapsed = time.time() - t0
            time.sleep(max(0.0, dt - elapsed))
    finally:
        try:
            stream.cancel()
        except Exception:
            pass


def run_pi05_eval(
    robot: Any,
    policy: PI05Policy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    ds_meta: LeRobotDatasetMetadata,
    frames: Optional[EpisodeFrameSource],
    cameras: Optional[dict[str, RealSenseCamera]],
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    gripper: Optional[Gripper],
    task_str: str,
    dt: float,
    device: str,
    actions_per_chunk: int,
    queue_threshold_frac: float,
    proprio_source: str,
) -> None:
    policy.reset()
    action_queue: deque[np.ndarray] = deque()
    stream = robot.create_command_stream()
    episode_done = False
    try:
        while True:
            cycle_start = time.time()

            if len(action_queue) > 0:
                act = action_queue.popleft()
                stream = _send_action_or_refresh_stream(robot, stream, act, dt, gripper)

            if SystemContext.latest_joint_positions.size == 0:
                time.sleep(0.01)
                continue

            if episode_done:
                if len(action_queue) == 0:
                    break
                elapsed = time.time() - cycle_start
                time.sleep(max(0.0, dt - elapsed))
                continue

            need = len(action_queue) <= int(queue_threshold_frac * actions_per_chunk)

            if proprio_source == "live":
                if cameras is None:
                    raise RuntimeError("live mode requires RealSense cameras")
                left_rgb, _ = cameras["left_camera"].read()
                front_rgb, _ = cameras["front_camera"].read()
                right_rgb, _ = cameras["right_camera"].read()
                state_16 = extract_left_right_16(
                    SystemContext.latest_joint_positions, left_idx, right_idx, gripper
                )
                if need:
                    _cancel_command_stream(stream)
                    raw: dict[str, Any] = build_raw_observation(
                        ds_meta, state_16, left_rgb, front_rgb, right_rgb, device
                    )
                    raw["task"] = [task_str]
                    raw_tensors = {k: raw[k].to(device, non_blocking=True) for k in raw if k != "task"}
                    raw_tensors["task"] = [task_str]
                    batch = preprocessor(raw_tensors)
                    t0 = time.time()
                    chunk = policy.predict_action_chunk(batch)
                    log_collect_demos(f"pi05 chunk inference {time.time() - t0:.3f}s", "success")
                    max_chunk = min(actions_per_chunk, chunk.shape[1])
                    _, chunk_size, _ = chunk.shape
                    n_enq = min(max_chunk, chunk_size)
                    for i in range(n_enq):
                        single = chunk[:, i, :]
                        denorm = postprocessor(single)
                        vec = denorm.squeeze(0).detach().cpu().numpy()
                        action_queue.append(vec)
                    log_collect_demos(f"enqueued {n_enq} actions", "info")
                    stream = robot.create_command_stream()
            elif need:
                assert frames is not None
                try:
                    left_rgb, front_rgb, right_rgb = frames.peek_triplet()
                except StopIteration:
                    logging.info("Episode finished (loop_episode=False). Draining action queue then stop.")
                    episode_done = True
                    elapsed = time.time() - cycle_start
                    time.sleep(max(0.0, dt - elapsed))
                    continue

                state_16 = extract_left_right_16(
                    SystemContext.latest_joint_positions, left_idx, right_idx, gripper
                )
                _cancel_command_stream(stream)
                raw = build_raw_observation(ds_meta, state_16, left_rgb, front_rgb, right_rgb, device)
                raw["task"] = [task_str]
                raw_tensors = {k: raw[k].to(device, non_blocking=True) for k in raw if k != "task"}
                raw_tensors["task"] = [task_str]
                batch = preprocessor(raw_tensors)
                t0 = time.time()
                chunk = policy.predict_action_chunk(batch)
                log_collect_demos(f"pi05 chunk inference {time.time() - t0:.3f}s", "success")
                max_chunk = min(actions_per_chunk, chunk.shape[1])
                _, chunk_size, _ = chunk.shape
                n_enq = min(max_chunk, chunk_size)
                for i in range(n_enq):
                    single = chunk[:, i, :]
                    denorm = postprocessor(single)
                    vec = denorm.squeeze(0).detach().cpu().numpy()
                    action_queue.append(vec)
                log_collect_demos(f"enqueued {n_enq} actions", "info")
                frames.advance_by(n_enq)
                stream = robot.create_command_stream()

            elapsed = time.time() - cycle_start
            time.sleep(max(0.0, dt - elapsed))
    finally:
        try:
            stream.cancel()
        except Exception:
            pass


def main() -> None:
    global _cleanup_robot, _cleanup_no_head, _cleanup_gripper

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    args = tyro.cli(Args)
    ps = str(args.proprio_source).lower()
    if ps not in ("live", "episode"):
        raise ValueError(f"proprio_source must be 'live' or 'episode', got {args.proprio_source!r}")
    proprio_source = ps

    cfg_path = str(Path(args.config_path).expanduser().resolve())
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)

    episode_dir = Path(args.episode_dir).expanduser().resolve()
    frames: Optional[EpisodeFrameSource] = None
    cameras: Optional[dict[str, RealSenseCamera]] = None
    if proprio_source == "episode":
        left_rgb, front_rgb, right_rgb = load_episode_rgb_frames(episode_dir)
        frames = EpisodeFrameSource(left_rgb, front_rgb, right_rgb, loop=args.loop_episode)
    else:
        ids = get_device_ids()
        logging.info("RealSense devices: %s", ids)
        camera_cfg = cfg["sensors"]["cameras"]
        cameras = {
            "left_camera": RealSenseCamera(camera_cfg["left_camera"]["device_id"]),
            "front_camera": RealSenseCamera(camera_cfg["front_camera"]["device_id"]),
            "right_camera": RealSenseCamera(camera_cfg["right_camera"]["device_id"]),
        }

    policy_cfg = cfg["policy"]
    repo_id = policy_cfg["repo_id"]
    ckpt = policy_cfg["checkpoint_path"]
    ptype = str(policy_cfg["type"]).lower()

    ds_meta = LeRobotDatasetMetadata(repo_id=repo_id)

    hz = args.control_hz
    if hz is None:
        hz = float(cfg.get("lerobot", {}).get("fps", 30.0))
    dt = 1.0 / hz

    task_str = args.task or cfg.get("storage", {}).get("language_instruction") or ""

    robot = connect_rby1(args.rby1, args.rby1_model, args.no_head, state_update_hz=1.0 / dt)
    if not robot.wait_for_control_ready(2000):
        raise RuntimeError("Control manager not ready after bring-up.")
    wait_for_joint_state(timeout_sec=5.0)

    gripper = None
    if not args.no_gripper:
        gripper = connect_gripper_for_replay(robot)
        if gripper is None:
            logging.warning("Gripper not available; continuing without gripper commands.")

    left_idx, right_idx = _resolve_joint_indices(robot)

    _cleanup_robot = robot
    _cleanup_no_head = args.no_head
    _cleanup_gripper = gripper

    logging.info("Moving to ready pose before policy...")
    move_robot_to_ready_pose(robot, args.no_head, gripper)
    wait_for_joint_state(timeout_sec=5.0)

    if frames is not None:
        logging.info(
            "Open-loop eval: %d recorded frames | loop=%s | vision=episode | proprio=live (joints)",
            frames.num_frames,
            args.loop_episode,
        )
    else:
        logging.info("Open-loop eval: vision=live cameras | proprio=live (joints) — same as `eval.py`")
    if proprio_source == "episode" and ptype == "pi05":
        logging.info(
            "PI0.5 embeds proprio in the text prompt; JSON images + live joints can still mismatch "
            "visually and cause jitter."
        )

    if ptype in ("dp", "diffusion"):
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        policy = DiffusionPolicy.from_pretrained(ckpt, dataset_stats=ds_meta.stats)
        policy.to(DEVICE)
        pre, post = make_pre_post_processors(
            policy.config,
            ckpt,
            dataset_stats=ds_meta.stats,
        )
        logging.info("Starting diffusion open-loop eval at %.2f Hz | dataset=%s | ckpt=%s", hz, repo_id, ckpt)
        run_diffusion_eval(
            robot,
            policy,
            pre,
            post,
            ds_meta,
            frames,
            cameras,
            left_idx,
            right_idx,
            gripper,
            dt,
            DEVICE,
            proprio_source,
        )
    elif ptype == "pi05":
        if importlib.util.find_spec("transformers") is None:
            raise RuntimeError(
                "PI0.5 requires the Hugging Face `transformers` package. Without it, "
                "lerobot sets CONFIG_MAPPING to None and model construction fails. "
                "Install with: pip install transformers"
            )
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        policy = PI05Policy.from_pretrained(ckpt)
        policy.dataset_stats = ds_meta.stats
        policy.to(DEVICE)
        pre, post = make_pre_post_processors(policy.config, ckpt, dataset_stats=ds_meta.stats)
        if not task_str:
            raise ValueError("PI0.5 requires a language task; set storage.language_instruction or --task.")
        logging.info("Starting PI0.5 open-loop eval at %.2f Hz | dataset=%s | ckpt=%s", hz, repo_id, ckpt)
        run_pi05_eval(
            robot,
            policy,
            pre,
            post,
            ds_meta,
            frames,
            cameras,
            left_idx,
            right_idx,
            gripper,
            task_str,
            dt,
            DEVICE,
            args.pi05_actions_per_chunk,
            args.pi05_queue_threshold_frac,
            proprio_source,
        )
    else:
        raise ValueError(f"Unsupported policy.type in config: {policy_cfg['type']!r} (use dp/diffusion or pi05)")


if __name__ == "__main__":
    main()
