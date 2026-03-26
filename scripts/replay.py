import argparse
import logging
import sys
import time
from pathlib import Path
import numpy as np
import rby1_sdk as rby
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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


def connect_rby1(address: str, model: str = "m", no_head: bool = False):
    logging.info(f"Connecting to RB-Y1 ({address}, model={model})")
    robot = rby.create_robot(address, model)
    if not robot.connect():
        raise RuntimeError("Failed to connect to robot.")

    servo_pattern = "^(?!head_).*" if no_head else ".*"
    if not robot.is_power_on(".*") and not robot.power_on(".*"):
        raise RuntimeError("Failed to power on robot.")
    if not robot.is_servo_on(servo_pattern) and not robot.servo_on(servo_pattern):
        raise RuntimeError("Failed to turn on servo.")

    cm_state = robot.get_control_manager_state().state
    if cm_state in [rby.ControlManagerState.State.MajorFault, rby.ControlManagerState.State.MinorFault]:
        if not robot.reset_fault_control_manager():
            raise RuntimeError("Failed to reset control manager fault.")
    if not robot.enable_control_manager(unlimited_mode_enabled=True):
        raise RuntimeError("Failed to enable control manager.")

    SystemContext.robot_model = robot.model()
    robot.start_state_update(robot_state_callback, 30.0)
    return robot


def wait_for_joint_state(timeout_sec: float = 5.0):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if SystemContext.latest_joint_positions.size > 0:
            return
        time.sleep(0.02)
    raise TimeoutError("Timed out waiting for robot state update.")


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
    logging.info(
        f"Loaded task={task}, episode={episode}, steps={len(demo)} | "
        f"camera={replay_camera}, robot={replay_robot}"
    )

    if replay_camera:
        replay_camera_episode(replayer, args.dt)
    if replay_robot:
        robot = connect_rby1(args.rby1, args.rby1_model, args.no_head)
        wait_for_joint_state(timeout_sec=5.0)
        replay_robot_episode(
            robot=robot,
            replayer=replayer,
            current_joint_positions=SystemContext.latest_joint_positions,
            joint_names=list(SystemContext.robot_model.robot_joint_names),
            dt=args.dt,
            left_prefix="left_arm",
            right_prefix="right_arm",
            expected_per_arm=7,
        )


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

    main(parser.parse_args())