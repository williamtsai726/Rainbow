import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np
import rby1_sdk as rby
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from gripper import Gripper  # noqa: E402

from data_utils.data_replay import (
    create_replayer_from_storage,
    pick_replay_modes,
    replay_camera_episode,
    replay_robot_episode,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s")


class SystemContext:
    robot_model = None
    latest_joint_positions = np.array([])


def robot_state_callback(robot_state: rby.RobotState_A):
    SystemContext.latest_joint_positions = robot_state.position


def connect_rby1(
    address: str,
    model: str = "m",
    no_head: bool = False,
    state_update_hz: float = 1.0 / 0.033,
):
    """Same bring-up sequence as scripts/collect.py and scripts/teleop.py: connect, power, servo, fault reset, CM."""
    logging.info(f"Attempting to connect to RB-Y1... (Address: {address}, Model: {model})")
    robot = rby.create_robot(address, model)

    if not robot.connect():
        raise RuntimeError("Failed to connect to RB-Y1.")
    logging.info("Successfully connected to RB-Y1.")

    servo_pattern = "^(?!head_).*" if no_head else ".*"

    if not robot.is_power_on(".*"):
        logging.warning("Robot power is off. Turning it on...")
        if not robot.power_on(".*"):
            raise RuntimeError("Failed to power on.")
        logging.info("Power turned on successfully.")
    else:
        logging.info("Power is already on.")

    if not robot.is_servo_on(servo_pattern):
        logging.warning("Servo is off. Turning it on...")
        ok = robot.servo_on(servo_pattern)
        if not ok and no_head:
            logging.warning("Retrying servo_on with pattern '.*' (all joints including head)...")
            ok = robot.servo_on(".*")
        if not ok:
            raise RuntimeError("Failed to turn on the servo.")
        logging.info("Servo turned on successfully.")
    else:
        logging.info("Servo is already on.")

    cm_state = robot.get_control_manager_state().state
    if cm_state in (
        rby.ControlManagerState.State.MajorFault,
        rby.ControlManagerState.State.MinorFault,
    ):
        logging.warning("Control Manager is in Fault state: %s. Attempting reset...", cm_state.name)
        if not robot.reset_fault_control_manager():
            raise RuntimeError("Failed to reset Control Manager.")
        logging.info("Control Manager reset successfully.")

    if not robot.enable_control_manager(unlimited_mode_enabled=True):
        raise RuntimeError("Failed to enable Control Manager.")
    logging.info("Control Manager successfully enabled. (Unlimited Mode: enabled)")

    SystemContext.robot_model = robot.model()
    robot.start_state_update(robot_state_callback, state_update_hz)
    return robot


def wait_for_joint_state(timeout_sec: float = 5.0):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if SystemContext.latest_joint_positions.size > 0:
            return
        time.sleep(0.02)
    raise TimeoutError("Timed out waiting for robot state update.")


def move_robot_to_ready_pose(
    robot, no_head: bool, gripper: Optional[Gripper] = None
) -> None:
    """Same ready pose as scripts/collect.py `move_robot_to_ready_pose` / VR handle_vr_button_event."""
    if robot.get_control_manager_state().control_state != rby.ControlManagerState.ControlState.Idle:
        robot.cancel_control()
    if robot.wait_for_control_ready(1000):
        ready_pose = np.deg2rad(
            [0.0, 45.0, -90.0, 45.0, 0.0, 0.0]
            + [0.0, -15.0, 0.0, -120.0, 0.0, 70.0, 0.0]
            + [0.0, 15.0, 0.0, -120.0, 0.0, 70.0, 0.0]
        )
        cbc = (
            rby.ComponentBasedCommandBuilder()
            .set_body_command(
                rby.JointImpedanceControlCommandBuilder()
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1))
                .set_position(ready_pose)
                .set_stiffness([400.0] * 6 + [60.0] * 7 + [60.0] * 7)
                .set_torque_limit([500.0] * 6 + [30.0] * 7 + [30.0] * 7)
                .set_minimum_time(2)
            )
        )
        if not no_head:
            cbc.set_head_command(
                rby.JointPositionCommandBuilder()
                .set_position([0.0] * len(SystemContext.robot_model.head_idx))
                .set_minimum_time(2)
            )
        robot.send_command(rby.RobotCommandBuilder().set_command(cbc)).get()
        if gripper is not None:
            gripper.set_normalized_target(np.array([1.0, 1.0]))


def connect_gripper_for_replay(robot) -> Optional[Gripper]:
    """Same bring-up as collect.py: tool flange 12 V, Dynamixel homing, control thread."""
    for arm in ["left", "right"]:
        if not robot.set_tool_flange_output_voltage(arm, 12):
            logging.error("Failed to supply 12V to tool flange. (%s)", arm)
            return None
    time.sleep(0.5)
    gripper = Gripper()
    if not gripper.initialize(verbose=True):
        logging.error("Gripper initialize() failed.")
        return None
    time.sleep(0.3)
    gripper.homing()
    gripper.start()
    return gripper


def main(args: argparse.Namespace):
    cfg = OmegaConf.to_container(OmegaConf.load(str(Path(args.config_path).expanduser().resolve())), resolve=True)

    task = args.task or cfg["storage"]["task_directory"]
    episode = args.episode
    if episode is None:
        episode = int(input("Enter episode number: "))

    if not args.camera_replay and not args.robot_replay and args.mode is None:
        replay_robot = input("Replay robot trajectory? (y/n): ").strip().lower() in {"y", "yes"}
        replay_camera = input("Replay camera images? (y/n): ").strip().lower() in {"y", "yes"}
    else:
        replay_camera, replay_robot = pick_replay_modes(
            args.mode or "both", args.camera_replay, args.robot_replay
        )
    if not replay_camera and not replay_robot:
        raise ValueError("No replay mode selected.")

    replayer = create_replayer_from_storage(cfg["storage"], task, episode)
    demo = replayer.demo
    episode_has_gripper = False
    if demo and isinstance(demo[0], dict):
        s0 = demo[0]
        for _k in ("left_ee", "right_ee", "next_left_ee", "next_right_ee"):
            _v = s0.get(_k)
            if _v is not None and np.asarray(_v, dtype=np.float64).reshape(-1).size >= 7:
                episode_has_gripper = True
                break
    logging.info(
        f"Loaded task={task}, episode={episode}, steps={len(demo)} | "
        f"camera={replay_camera}, robot={replay_robot} | gripper_in_episode={episode_has_gripper}"
    )

    if replay_camera:
        replay_camera_episode(replayer, args.dt)
    if replay_robot:
        robot = connect_rby1(
            args.rby1,
            args.rby1_model,
            args.no_head,
            state_update_hz=1.0 / args.dt,
        )
        if not robot.wait_for_control_ready(2000):
            raise RuntimeError("Control manager not ready after bring-up (wait_for_control_ready timed out).")
        wait_for_joint_state(timeout_sec=5.0)

        gripper = None
        if episode_has_gripper and not args.no_gripper:
            logging.info("Initializing external grippers for replay (episode contains gripper channels)...")
            gripper = connect_gripper_for_replay(robot)
            if gripper is None:
                logging.warning("Gripper init failed; replaying arms only.")

        logging.info("Moving robot to ready pose before replay...")
        move_robot_to_ready_pose(robot, args.no_head, gripper)
        wait_for_joint_state(timeout_sec=5.0)

        replay_robot_episode(
            robot=robot,
            replayer=replayer,
            dt=args.dt,
            joint_source=args.joint_source,
            gripper=gripper,
        )
        logging.info("Moving robot to ready pose after replay...")
        move_robot_to_ready_pose(robot, args.no_head, gripper)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Episode replay for Rainbow dataset")
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task directory name under storage.base_dir (default: config value)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode number to replay (e.g. 1 for 000001)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["both", "camera", "robot"],
        default=None,
        help="Replay mode when not using --camera_replay/--robot_replay.",
    )
    parser.add_argument("--camera_replay", action="store_true", help="Replay camera images.")
    parser.add_argument("--robot_replay", action="store_true", help="Replay robot joint trajectory.")
    parser.add_argument("--dt", type=float, default=0.033, help="Replay timestep in seconds.")
    parser.add_argument("--rby1", default="192.168.30.1:50051", type=str, help="Robot gRPC address.")
    parser.add_argument("--rby1_model", default="m", type=str, help="Robot model type.")
    parser.add_argument("--no_head", action="store_true", help="Exclude head servo from bring-up.")
    parser.add_argument(
        "--no_gripper",
        action="store_true",
        help="Do not initialize or replay external Dynamixel grippers (arms only).",
    )
    parser.add_argument(
        "--joint-source",
        type=str,
        choices=["current", "next"],
        default="current",
        help="Which pose6 keys drive replay: right_ee/left_ee (current) or next_right_ee/next_left_ee (next).",
    )

    main(parser.parse_args())