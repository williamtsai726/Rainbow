#!/usr/bin/env python3
"""Run MolmoAct-style remote policy inference on the real RB-Y1.

Mirrors ``eval.py`` for robot setup, cameras, and ``send_predicted_action16``; replaces local
LeRobot inference with HTTP POST (``json_numpy``) to a server that returns an ``actions`` list
(16-D targets per step: left 8 + right 8), same layout as ``eval.py``."""

from __future__ import annotations

import atexit
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import tyro
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_LEROBOT_SRC = _REPO_ROOT / "lerobot" / "src"
for _p in (_REPO_ROOT, _SCRIPTS_DIR, _LEROBOT_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from camera.realsense_camera import RealSenseCamera, get_device_ids  # noqa: E402
from data_utils.data_eval import send_predicted_action16  # noqa: E402
from data_utils.logging_utils import log_collect_demos  # noqa: E402
from gripper import Gripper  # noqa: E402
from replay import (  # noqa: E402
    SystemContext,
    connect_gripper_for_replay,
    connect_rby1,
    move_robot_to_ready_pose,
    wait_for_joint_state,
)

from molmoact import MolmoAct  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s")


# Same joint / command helpers as ``eval.py`` (kept local so this script does not import LeRobot).
def _resolve_joint_indices(robot: Any) -> tuple[np.ndarray, np.ndarray]:
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


def _cancel_command_stream(stream: Any) -> None:
    try:
        stream.cancel()
    except Exception:
        pass


def _invert_gripper_dims_for_rby_inplace(act_16: np.ndarray) -> None:
    """In-place: dataset gripper convention for ``send_predicted_action16`` (dims 7 and 15)."""
    act_16[7] = float(np.clip(1.0 - act_16[7], 0.0, 1.0))
    act_16[15] = float(np.clip(1.0 - act_16[15], 0.0, 1.0))


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
        logging.info("Eval cleanup: moving to ready pose...")
        try:
            move_robot_to_ready_pose(_cleanup_robot, _cleanup_no_head, _cleanup_gripper)
        except Exception as exc:
            logging.warning("Cleanup ready-pose failed: %s", exc)
    logging.info("Cleanup completed.")


def _signal_handler(signum, frame) -> None:  # noqa: ARG001
    _cleanup()
    os._exit(0)


def _policy_read_size_from_max_width(max_width: Optional[int]) -> Optional[Tuple[int, int]]:
    """Match RealSense 640×360 aspect when downsampling (faster RGB + smaller HTTP body)."""
    if max_width is None or max_width <= 0:
        return None
    if max_width >= 640:
        return None
    h = max(1, int(round(360 * (max_width / 640.0))))
    return (max_width, h)


def run_molmoact_eval(
    robot: Any,
    cameras: dict[str, RealSenseCamera],
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    gripper: Optional[Gripper],
    instruction: str,
    dt: float,
    policy: MolmoAct,
    invert_gripper_actions: bool,
    policy_read_size: Optional[Tuple[int, int]],
) -> None:
    stream = robot.create_command_stream()
    try:
        while True:
            if SystemContext.latest_joint_positions.size == 0:
                time.sleep(0.01)
                continue

            cycle_start = time.time()

            left_rgb, _ = cameras["left_camera"].read(img_size=policy_read_size)
            front_rgb, _ = cameras["front_camera"].read(img_size=policy_read_size)
            right_rgb, _ = cameras["right_camera"].read(img_size=policy_read_size)

            state_16 = extract_left_right_16(
                SystemContext.latest_joint_positions, left_idx, right_idx, gripper
            )

            _cancel_command_stream(stream)

            t_inf0 = time.time()
            response = policy.infer_from_observation(
                left_rgb, front_rgb, right_rgb, state_16, instruction
            )
            log_collect_demos(
                f"molmoact inference {(time.time() - t_inf0):.2f} s", "info"
            )

            if "actions" not in response:
                raise KeyError(f"Server response missing 'actions' key: {list(response.keys())}")
            actions = response["actions"]
            logging.info("Received %s action(s) from server", len(actions))

            stream = robot.create_command_stream()

            for act in actions:
                act_np = np.asarray(act, dtype=np.float32).reshape(-1)
                if act_np.size != 16:
                    raise ValueError(f"Each action must be 16-D; got shape {act_np.shape}")
                if invert_gripper_actions and gripper is not None:
                    _invert_gripper_dims_for_rby_inplace(act_np)
                t0 = time.time()
                stream = _send_action_or_refresh_stream(robot, stream, act_np, dt, gripper)
                time.sleep(max(0.0, dt - (time.time() - t0)))

            logging.debug(
                "full cycle %.3fs (server returned %s actions)",
                time.time() - cycle_start,
                len(actions),
            )
    finally:
        try:
            stream.cancel()
        except Exception:
            pass


@dataclass
class Args:
    config_path: str = str(_REPO_ROOT / "config.yaml")
    """Path to project config YAML (sensors, storage)."""

    molmoact_url: str = os.environ.get("MOLMOACT_URL", "https://3495-71-41-244-70.ngrok-free.app/act")
    """Remote inference HTTP endpoint (POST). Override with env MOLMOACT_URL."""

    task: Optional[str] = None
    """Language instruction (default: ``storage.language_instruction`` in YAML)."""

    control_hz: Optional[float] = None
    """Control loop / per-action dt from ``lerobot.fps`` or 30."""

    rby1: str = "192.168.30.1:50051"
    rby1_model: str = "m"
    no_head: bool = True
    no_gripper: bool = False

    multi_views: bool = True
    """Send left/top/right cameras (``False`` = single ``front`` as ``image``)."""

    request_timeout_sec: float = 120.0
    """HTTP timeout for one inference call."""

    ngrok_skip_browser_warning: bool = True
    """If True, add header to skip ngrok interstitial on free tier."""

    invert_gripper_actions: bool = True
    """If True, flip gripper dims (7, 15) on each action so remote outputs match RBY
    ``set_normalized_target`` / dataset convention. Use ``--no-invert-gripper-actions`` if
    your server already returns the same 16-D gripper convention as local LeRobot eval."""

    policy_image_max_width: Optional[int] = None
    """If set (e.g. 320 or 480), RealSense RGB is resized before upload (smaller JSON +
    faster network). Native is 640×360; aspect is preserved. Omit for full resolution."""


def main() -> None:
    global _cleanup_robot, _cleanup_no_head, _cleanup_gripper

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    args = tyro.cli(Args)
    cfg_path = str(Path(args.config_path).expanduser().resolve())
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)

    ids = get_device_ids()
    logging.info("RealSense devices: %s", ids)

    camera_cfg = cfg["sensors"]["cameras"]
    cameras = {
        "left_camera": RealSenseCamera(camera_cfg["left_camera"]["device_id"]),
        "front_camera": RealSenseCamera(camera_cfg["front_camera"]["device_id"]),
        "right_camera": RealSenseCamera(camera_cfg["right_camera"]["device_id"]),
    }

    hz = args.control_hz
    if hz is None:
        hz = float(cfg.get("lerobot", {}).get("fps", 30.0))
    dt = 1.0 / hz

    task_str = args.task or cfg.get("storage", {}).get("language_instruction") or ""
    if not task_str:
        logging.warning("No language instruction: set storage.language_instruction or --task")

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

    logging.info("Moving to ready pose before remote policy...")
    move_robot_to_ready_pose(robot, args.no_head, gripper)
    wait_for_joint_state(timeout_sec=5.0)

    extra: Optional[dict[str, str]] = None
    if args.ngrok_skip_browser_warning:
        extra = {"ngrok-skip-browser-warning": "69420"}

    policy = MolmoAct(
        args.molmoact_url,
        multi_views=args.multi_views,
        request_timeout_sec=args.request_timeout_sec,
        extra_headers=extra,
    )

    policy_read_size = _policy_read_size_from_max_width(args.policy_image_max_width)
    if policy_read_size is not None:
        logging.info(
            "Policy camera read size %s×%s (set --policy-image-max-width to change)",
            policy_read_size[0],
            policy_read_size[1],
        )

    logging.info(
        "Starting MolmoAct remote eval at %.2f Hz | url=%s | multi_views=%s",
        hz,
        args.molmoact_url,
        args.multi_views,
    )
    run_molmoact_eval(
        robot,
        cameras,
        left_idx,
        right_idx,
        gripper,
        task_str,
        dt,
        policy,
        args.invert_gripper_actions,
        policy_read_size,
    )


if __name__ == "__main__":
    main()
