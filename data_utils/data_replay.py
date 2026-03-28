import ast
import concurrent.futures
import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from data_utils.logging_utils import log_data_utils, log_demo_data_info, log_replay

try:
    import rby1_sdk as rby
except Exception:  # pragma: no cover
    rby = None


class DataReplayer:
    def __init__(self, save_format: str = "json", old_format: bool = False):
        self.save_format = save_format
        self.old_format = old_format
        self.demo = None

        self.left_camera_key = "left_rgb"
        self.right_camera_key = "right_rgb"
        self.front_camera_key = "front_rgb"

        self.left_rgb_paths = None
        self.right_rgb_paths = None
        self.front_rgb_paths = None

    @staticmethod
    def _as_float_array(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value.astype(np.float64)
        if isinstance(value, (list, tuple)):
            return np.asarray(value, dtype=np.float64)
        if isinstance(value, str):
            return np.asarray(ast.literal_eval(value), dtype=np.float64)
        raise TypeError(f"Unsupported joint value type: {type(value)}")

    @staticmethod
    def _resolve_value(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
            try:
                return [float(x) for x in ast.literal_eval(value)]
            except (SyntaxError, ValueError):
                return value
        return value

    def load_episode(self, root_dir: str, episode_number: int) -> bool:
        episode_number = f"{int(episode_number):06d}"
        try:
            if self.save_format == "json":
                demo_path = (
                    f"{root_dir}/{episode_number}/{episode_number}.json"
                    if not self.old_format
                    else f"{root_dir}_pickle/{episode_number}.pkl"
                )

                if not os.path.exists(demo_path):
                    log_data_utils(f"Episode file not found: {demo_path}", "error")
                    return False

                log_replay(f"Found episode file: {demo_path}", "info")
                with open(demo_path, "rb") as f:
                    demo = json.load(f)

                if not isinstance(demo, list):
                    log_data_utils("Expected list-based episode for json format.", "error")
                    return False

                self.demo = [{k: self._resolve_value(v) for k, v in step.items()} for step in demo]

                left_dir = f"{root_dir}/{episode_number}/{self.left_camera_key}"
                right_dir = f"{root_dir}/{episode_number}/{self.right_camera_key}"
                front_dir = f"{root_dir}/{episode_number}/{self.front_camera_key}"

                self.left_rgb_paths = (
                    sorted([os.path.join(left_dir, x) for x in os.listdir(left_dir)])
                    if os.path.isdir(left_dir)
                    else []
                )
                self.right_rgb_paths = (
                    sorted([os.path.join(right_dir, x) for x in os.listdir(right_dir)])
                    if os.path.isdir(right_dir)
                    else []
                )
                self.front_rgb_paths = (
                    sorted([os.path.join(front_dir, x) for x in os.listdir(front_dir)])
                    if os.path.isdir(front_dir)
                    else []
                )

                def load_image(path: str):
                    try:
                        return np.array(Image.open(path))
                    except Exception:
                        return None

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    left_images = list(executor.map(load_image, self.left_rgb_paths))
                    right_images = list(executor.map(load_image, self.right_rgb_paths))
                    front_images = list(executor.map(load_image, self.front_rgb_paths))

                for i, step in enumerate(self.demo):
                    if i < len(left_images) and left_images[i] is not None:
                        step["image_left_rgb"] = left_images[i]
                    if i < len(right_images) and right_images[i] is not None:
                        step["image_right_rgb"] = right_images[i]
                    if i < len(front_images) and front_images[i] is not None:
                        step["image_front_rgb"] = front_images[i]

                if self.demo:
                    log_demo_data_info(self.demo[0], demo_path)
                return True

            if self.save_format == "npy":
                demo_path = os.path.join(root_dir, self.save_format, f"{episode_number}.npy")
                with open(demo_path, "rb") as f:
                    self.demo = pickle.load(f)
                if self.demo:
                    log_demo_data_info(self.demo, demo_path)
                return True

            log_data_utils(f"Unsupported save_format: {self.save_format}", "error")
            return False
        except Exception as e:
            log_data_utils(f"Error loading demo data: {str(e)}", "error")
            return False

    def get_demo_length(self) -> int:
        return 0 if self.demo is None else len(self.demo)

    def get_step(self, step_idx: int) -> Dict[str, Any]:
        if self.demo is None:
            raise RuntimeError("No demo data loaded. Call load_episode first.")
        if step_idx < 0 or step_idx >= len(self.demo):
            raise IndexError(f"step_idx={step_idx} out of range.")
        return self.demo[step_idx]

    def get_observation(self, step_idx: int) -> Dict[str, Any]:
        return dict(self.get_step(step_idx))

    def get_instruction(self) -> Optional[str]:
        if not self.demo:
            return None
        return self.demo[0].get("language_instruction")

    def get_step_arm_joints(
        self,
        step_idx: int,
        left_key: str = "left_joint",
        right_key: str = "right_joint",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-side vectors from JSON: 8 floats (7 arm + 1 gripper) or legacy 7 arm-only."""
        step = self.get_step(step_idx)
        if left_key not in step or right_key not in step:
            raise KeyError(f"Missing keys: {left_key}, {right_key}")
        return self._as_float_array(step[left_key]), self._as_float_array(step[right_key])

    def resolve_arm_joint_indices(
        self,
        joint_names: Sequence[str],
        left_prefix: str = "left_arm",
        right_prefix: str = "right_arm",
        expected_per_arm: int = 7,
    ) -> Tuple[np.ndarray, np.ndarray]:
        left_idx = [i for i, name in enumerate(joint_names) if str(name).startswith(left_prefix)]
        right_idx = [i for i, name in enumerate(joint_names) if str(name).startswith(right_prefix)]
        left_idx.sort(key=lambda i: joint_names[i])
        right_idx.sort(key=lambda i: joint_names[i])
        if len(left_idx) < expected_per_arm or len(right_idx) < expected_per_arm:
            raise RuntimeError(
                f"Invalid joint map for {left_prefix}/{right_prefix}: "
                f"left={len(left_idx)} right={len(right_idx)}"
            )
        return (
            np.asarray(left_idx[:expected_per_arm], dtype=np.int64),
            np.asarray(right_idx[:expected_per_arm], dtype=np.int64),
        )

    def build_target_joint_position(
        self,
        step_idx: int,
        current_joint_positions: np.ndarray,
        joint_names: Sequence[str],
        left_prefix: str = "left_arm",
        right_prefix: str = "right_arm",
        expected_per_arm: int = 7,
        left_key: str = "left_joint",
        right_key: str = "right_joint",
    ) -> np.ndarray:
        left_idx, right_idx = self.resolve_arm_joint_indices(
            joint_names=joint_names,
            left_prefix=left_prefix,
            right_prefix=right_prefix,
            expected_per_arm=expected_per_arm,
        )
        left_joint, right_joint = self.get_step_arm_joints(
            step_idx, left_key=left_key, right_key=right_key
        )
        lu = np.asarray(self._as_float_array(left_joint), dtype=np.float64).reshape(-1)[:7]
        ru = np.asarray(self._as_float_array(right_joint), dtype=np.float64).reshape(-1)[:7]
        target = np.asarray(current_joint_positions, dtype=np.float64).copy()
        target[left_idx] = lu[: len(left_idx)]
        target[right_idx] = ru[: len(right_idx)]
        return target

    def get_step_camera_images(
        self,
        step_idx: int,
        left_key: str = "image_left_rgb",
        front_key: str = "image_front_rgb",
        right_key: str = "image_right_rgb",
    ) -> Dict[str, Any]:
        step = self.get_step(step_idx)
        return {
            left_key: step.get(left_key),
            front_key: step.get(front_key),
            right_key: step.get(right_key),
        }


def pick_replay_modes(task_mode: str, explicit_camera: bool, explicit_robot: bool) -> Tuple[bool, bool]:
    if explicit_camera or explicit_robot:
        return explicit_camera, explicit_robot
    if task_mode == "camera":
        return True, False
    if task_mode == "robot":
        return False, True
    return True, True


def create_replayer_from_storage(storage_cfg: Dict[str, Any], task: str, episode: int) -> DataReplayer:
    root_dir = f"{storage_cfg['base_dir']}/{task}"
    replayer = DataReplayer(
        save_format=storage_cfg.get("save_format", "json"),
        old_format=storage_cfg.get("old_format", False),
    )
    if not replayer.load_episode(root_dir, episode):
        raise RuntimeError(f"Failed to load episode {episode} from {root_dir}")
    return replayer


def replay_camera_episode(
    replayer: DataReplayer,
    dt: float,
    left_key: str = "image_left_rgb",
    front_key: str = "image_front_rgb",
    right_key: str = "image_right_rgb",
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Camera replay requires matplotlib. Install with: pip install matplotlib"
        ) from exc

    if not replayer.demo:
        log_data_utils("No demo steps found for camera replay.", "warning")
        return

    plt.ion()
    _, axs = plt.subplots(1, 3, figsize=(16, 5))
    labels = [("left", left_key), ("front", front_key), ("right", right_key)]
    total = replayer.get_demo_length()

    for idx in range(total):
        images = replayer.get_step_camera_images(
            idx,
            left_key=left_key,
            front_key=front_key,
            right_key=right_key,
        )
        for ax, (title, key) in zip(axs, labels):
            ax.clear()
            img = images.get(key)
            if img is None:
                ax.text(0.5, 0.5, f"missing: {key}", ha="center", va="center")
                ax.set_axis_off()
            else:
                ax.imshow(img)
                ax.set_axis_off()
            ax.set_title(f"{title} | step {idx + 1}/{total}")
        plt.tight_layout()
        plt.pause(max(dt, 0.001))

    plt.ioff()
    plt.show(block=False)


def _resolve_replay_joint_keys(replayer: DataReplayer, step_idx: int, joint_source: str) -> Tuple[str, str]:
    """Return (left_key, right_key) for per-side 8-float rows (7 arm + gripper)."""
    if joint_source == "next":
        lk, rk = "next_left_joint", "next_right_joint"
        step = replayer.get_step(step_idx)
        if lk not in step or rk not in step:
            return "left_joint", "right_joint"
        return lk, rk
    return "left_joint", "right_joint"


def _validate_joint_episode(replayer: DataReplayer, joint_source: str) -> None:
    n = replayer.get_demo_length()
    for idx in range(n):
        lk, rk = _resolve_replay_joint_keys(replayer, idx, joint_source)
        step = replayer.get_step(idx)
        for key in (lk, rk):
            if key not in step:
                raise KeyError(
                    f"Step {idx} missing '{key}' (expected at least 7 arm floats + optional gripper). "
                    "Episodes must use left_joint/right_joint (+ next_* when using --joint-source next)."
                )
            v = np.asarray(DataReplayer._as_float_array(step[key]), dtype=np.float64).reshape(-1)
            if v.size < 7:
                raise ValueError(f"Step {idx} '{key}' must have at least 7 arm floats, got {v.size}")


def replay_robot_episode(
    robot: Any,
    replayer: DataReplayer,
    dt: float,
    joint_source: str = "current",
    log_every: int = 50,
    gripper: Any = None,
):
    """Replay arms from JSON using 8 floats per side: 7 arm joints + gripper (1=open / 0=close).

    Uses ``left_joint`` / ``right_joint`` (or ``next_left_joint`` / ``next_right_joint`` when
    ``joint_source=='next'``). Sends the same command shape as ``scripts/collect.py`` teleop:
    ``BodyComponentBasedCommandBuilder`` with per-arm ``JointPositionCommandBuilder`` (7 DOF each),
    not a single full-body joint vector (which the stack may not apply the same way).
    """
    if rby is None:
        raise RuntimeError("rby1_sdk is required for robot replay.")
    if joint_source not in ("current", "next"):
        raise ValueError("joint_source must be 'current' or 'next'")
    if not replayer.demo:
        log_data_utils("No demo steps found for robot replay.", "warning")
        return
    if not robot.wait_for_control_ready(1000):
        raise RuntimeError("Robot control is not ready.")

    _validate_joint_episode(replayer, joint_source)
    log_data_utils(
        "Replaying JointPosition from 8-float arm+gripper rows per side "
        f"(joint_source={joint_source!r}).",
        "info",
    )

    stream = robot.create_command_stream()
    demo_length = replayer.get_demo_length()
    hold = dt * 10.0
    min_t = dt * 1.01

    for idx in range(demo_length):
        lk, rk = _resolve_replay_joint_keys(replayer, idx, joint_source)
        lj, rj = replayer.get_step_arm_joints(idx, left_key=lk, right_key=rk)
        lj = np.asarray(lj, dtype=np.float64).reshape(-1)
        rj = np.asarray(rj, dtype=np.float64).reshape(-1)
        if lj.size < 7 or rj.size < 7:
            raise ValueError(
                f"Step {idx}: need at least 7 arm values per side; got left={lj.size} right={rj.size}"
            )
        lu, ru = lj[:7], rj[:7]

        right_builder = (
            rby.JointPositionCommandBuilder()
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(hold))
            .set_minimum_time(min_t)
            .set_position(ru.tolist())
        )
        left_builder = (
            rby.JointPositionCommandBuilder()
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(hold))
            .set_minimum_time(min_t)
            .set_position(lu.tolist())
        )
        ctrl_builder = (
            rby.BodyComponentBasedCommandBuilder()
            .set_right_arm_command(right_builder)
            .set_left_arm_command(left_builder)
        )
        stream.send_command(
            rby.RobotCommandBuilder().set_command(
                rby.ComponentBasedCommandBuilder().set_body_command(ctrl_builder)
            )
        )
        if gripper is not None:
            lg = float(np.clip(lj[7], 0.0, 1.0)) if lj.size > 7 else 0.0
            rg = float(np.clip(rj[7], 0.0, 1.0)) if rj.size > 7 else 0.0
            gripper.set_normalized_target(np.array([rg, lg]))
        if idx % max(1, log_every) == 0:
            log_data_utils(f"Replay step {idx + 1}/{demo_length} (joint_keys={lk},{rk})", "info")
        time.sleep(dt)

    stream.cancel()


def replay_episode_pickle(demo_dir: str, env: Any):
    with open(demo_dir, "rb") as f:
        demo = pickle.load(f)
    for act in demo["action"]:
        env.step(act)
