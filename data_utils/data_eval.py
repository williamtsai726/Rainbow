"""Robot command helpers for policy evaluation.

Copied from the SDK path used in ``data_replay.replay_robot_episode`` so eval does not
depend on edits to ``data_replay.py``."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import rby1_sdk as rby
except Exception:  # pragma: no cover
    rby = None


def send_bimanual_joint_position_command(
    stream: Any,
    left_joint_row: np.ndarray,
    right_joint_row: np.ndarray,
    dt: float,
    gripper: Any = None,
) -> None:
    """Send one dual-arm joint target using the same SDK path as ``replay_robot_episode``.

    Each row is at least 7 arm joints; index 7 is optional gripper (normalized 0–1), matching
    saved episodes and ``scripts/collect.py``.

    Uses ``BodyComponentBasedCommandBuilder`` with per-arm ``JointPositionCommandBuilder`` (7 DOF),
    not a single full-body vector.
    """
    if rby is None:
        raise RuntimeError("rby1_sdk is required for joint commands.")
    lj = np.asarray(left_joint_row, dtype=np.float64).reshape(-1)
    rj = np.asarray(right_joint_row, dtype=np.float64).reshape(-1)
    if lj.size < 7 or rj.size < 7:
        raise ValueError(
            f"Need at least 7 arm values per side; got left={lj.size} right={rj.size}"
        )
    lu, ru = lj[:7], rj[:7]
    hold = dt * 10.0
    min_t = dt * 1.01
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


def send_predicted_action16(
    stream: Any,
    action_16: np.ndarray,
    dt: float,
    gripper: Any = None,
) -> None:
    """Apply one policy target (16-D = left8 + right8) via ``send_bimanual_joint_position_command``."""
    a = np.asarray(action_16, dtype=np.float64).reshape(-1)
    if a.size != 16:
        raise ValueError(f"Expected action dim 16, got {a.size}")
    send_bimanual_joint_position_command(stream, a[:8], a[8:], dt, gripper)
