import argparse
import logging
import shutil
import subprocess
import zmq
import time
import threading
from dataclasses import dataclass
from pathlib import Path
import sys
import rby1_sdk as rby
import socket
from typing import Union
import json
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from gripper import Gripper
from vr_control_state import VRControlState
import pickle

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s"
)

# Ensure repo root is on PYTHONPATH so imports like `camera.realsense_camera` work
# even when you run `python scripts/data_collect.py` from within `scripts/`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

T_conv = np.array([
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
])


@dataclass(frozen=True)
class Settings:
    dt: float = 0.033
    hand_offset: float = np.array([0.0, 0.0, 0.0])

    T_hand_offset = np.identity(4)
    T_hand_offset[0:3, 3] = hand_offset

    vr_control_local_port: int = 5005
    vr_control_meta_quest_port: int = 6000

    mobile_linear_acceleration_gain: float = 0.15
    mobile_angular_acceleration_gain: float = 0.15
    mobile_linear_damping_gain: float = 0.3
    mobile_angular_damping_gain: float = 0.3


class SystemContext:
    robot_model: Union[rby.Model_A, rby.Model_M] = None
    vr_state: VRControlState = VRControlState()


def open_zmq_pub_socket(bind_address: str) -> zmq.Socket:
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(bind_address)
    logging.info(f"ZMQ PUB server opened at {bind_address}")
    return socket


def robot_state_callback(robot_state: rby.RobotState_A):
    SystemContext.vr_state.joint_positions = robot_state.position
    SystemContext.vr_state.center_of_mass = robot_state.center_of_mass


def connect_rby1(address: str, model: str = "a", no_head: bool = False):
    logging.info(f"Attempting to connect to RB-Y1... (Address: {address}, Model: {model})")
    robot = rby.create_robot(address, model)

    connected = robot.connect()
    if not connected:
        logging.critical("Failed to connect to RB-Y1. Exiting program.")
        exit(1)
    logging.info("Successfully connected to RB-Y1.")

    servo_pattern = "^(?!head_).*" if no_head else ".*"
    if not robot.is_power_on(".*"):
        logging.warning("Robot power is off. Turning it on...")
        if not robot.power_on(".*"):
            logging.critical("Failed to power on. Exiting program.")
            exit(1)
        logging.info("Power turned on successfully.")
    else:
        logging.info("Power is already on.")

    if not robot.is_servo_on(servo_pattern):
        logging.warning("Servo is off. Turning it on...")
        if not robot.servo_on(servo_pattern):
            logging.critical("Failed to turn on the servo. Exiting program.")
            exit(1)
        logging.info("Servo turned on successfully.")
    else:
        logging.info("Servo is already on.")

    cm_state = robot.get_control_manager_state().state
    if cm_state in [
        rby.ControlManagerState.State.MajorFault,
        rby.ControlManagerState.State.MinorFault,
    ]:
        logging.warning(f"Control Manager is in Fault state: {cm_state.name}. Attempting reset...")
        if not robot.reset_fault_control_manager():
            logging.critical("Failed to reset Control Manager. Exiting program.")
            exit(1)
        logging.info("Control Manager reset successfully.")
    if not robot.enable_control_manager(unlimited_mode_enabled=True):
        logging.critical("Failed to enable Control Manager. Exiting program.")
        exit(1)
    logging.info("Control Manager successfully enabled. (Unlimited Mode: enabled)")

    SystemContext.robot_model = robot.model()
    robot.start_state_update(robot_state_callback, 1 / Settings.dt)

    return robot


def _cancel_command_stream(stream) -> None:
    if stream is None:
        return
    try:
        stream.cancel()
    except Exception:
        pass


def _send_robot_command_with_stream_retry(robot, stream, robot_cmd):
    """Send on the command stream; if the SDK reports an expired stream, recreate and retry once."""
    try:
        stream.send_command(robot_cmd)
        return stream
    except Exception as e:
        if "expired" not in str(e).lower():
            raise
        logging.warning("Command stream expired; recreating and retrying send.")
        _cancel_command_stream(stream)
        stream = robot.create_command_stream()
        stream.send_command(robot_cmd)
        return stream


def setup_meta_quest_udp_communication(local_ip: str, local_port: int, meta_quest_ip: str, meta_quest_port: int,
                                       power_off=None):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        target_info = {
            "ip": local_ip,
            "port": local_port
        }
        message = json.dumps(target_info).encode('utf-8')
        sock.sendto(message, (meta_quest_ip, meta_quest_port))
        logging.info(f"Sent local PC info to Meta Quest: {target_info}")

    def udp_server():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server_sock:
            server_sock.bind((local_ip, local_port))
            logging.info(f"UDP server running to receive Meta Quest Controller data... {local_ip}:{local_port}")
            while True:
                data, addr = server_sock.recvfrom(4096)
                udp_msg = data.decode('utf-8')
                try:
                    SystemContext.vr_state.controller_state = json.loads(udp_msg)
                    if "left" in SystemContext.vr_state.controller_state["hands"]:
                        buttons = SystemContext.vr_state.controller_state["hands"]["left"]["buttons"]
                        primary_button = buttons["primaryButton"]
                        secondary_button = buttons["secondaryButton"]

                        SystemContext.vr_state.event_left_a_pressed |= primary_button
                        SystemContext.vr_state.event_left_b_pressed |= secondary_button

                        if primary_button:
                            if power_off is not None:
                                logging.warning("Left X button pressed. Shutting down power.")
                                power_off()

                    if "right" in SystemContext.vr_state.controller_state["hands"]:
                        buttons = SystemContext.vr_state.controller_state["hands"]["right"]["buttons"]
                        primary_button = buttons["primaryButton"]
                        secondary_button = buttons["secondaryButton"]

                        SystemContext.vr_state.event_right_a_pressed |= primary_button
                        SystemContext.vr_state.event_right_b_pressed |= secondary_button

                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to decode JSON: {e} (from {addr}) - received data: {message[:100]}")

    thread = threading.Thread(target=udp_server, daemon=True)
    thread.start()


def handle_vr_button_event(robot: Union[rby.Robot_A, rby.Robot_M], no_head: bool):
    if SystemContext.vr_state.event_right_a_pressed:
        logging.info("Right A button pressed. Moving robot to ready pose.")
        if robot.get_control_manager_state().control_state != rby.ControlManagerState.ControlState.Idle:
            robot.cancel_control()
        if robot.wait_for_control_ready(1000):
            ready_pose = np.deg2rad(
                [0.0, 45.0, -90.0, 45.0, 0.0, 0.0] +
                [0.0, -15.0, 0.0, -120.0, 0.0, 70.0, 0.0] +
                [0.0, 15.0, 0.0, -120.0, 0.0, 70.0, 0.0])
            cbc = (
                rby.ComponentBasedCommandBuilder()
                .set_body_command(
                    rby.JointPositionCommandBuilder()
                    .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1))
                    .set_position(ready_pose)
                    .set_minimum_time(2)
                )
            )
            if not no_head:
                cbc.set_head_command(
                    rby.JointPositionCommandBuilder()
                    .set_position([0.] * len(SystemContext.robot_model.head_idx))
                    .set_minimum_time(2)
                )
            robot.send_command(
                rby.RobotCommandBuilder().set_command(
                    cbc
                )
            ).get()
        SystemContext.vr_state.is_initialized = True
        SystemContext.vr_state.is_stopped = False

    elif SystemContext.vr_state.event_right_b_pressed:
        logging.info("Right B button pressed. Stopping.")
        SystemContext.vr_state.is_stopped = True

    else:
        return False

    SystemContext.vr_state.event_right_a_pressed = False
    SystemContext.vr_state.event_right_b_pressed = False
    SystemContext.vr_state.event_left_a_pressed = False
    SystemContext.vr_state.event_left_b_pressed = False

    return True


def pose_to_se3(position, rotation_quat):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(rotation_quat).as_matrix()
    T[:3, 3] = position
    return T


def average_so3_slerp(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    # 두 회전을 Rotation 객체로 변환
    rot1 = R.from_matrix(R1)
    rot2 = R.from_matrix(R2)

    # 보간 설정: t=0 => rot1, t=1 => rot2
    slerp = Slerp([0, 1], R.concatenate([rot1, rot2]))

    # 평균값은 중간지점 t=0.5
    rot_avg = slerp(0.5)
    return rot_avg.as_matrix()


def publish_gv(sock: zmq.Socket):
    while True:
        sock.send(pickle.dumps(SystemContext.vr_state))
        time.sleep(0.1)

from omegaconf import OmegaConf
from camera.realsense_camera import RealSenseCamera, get_device_ids
from data_utils.data_saver import DataSaver
from data_utils.data_saver_thread import EpisodeSaverThread
from data_utils.arm_ik import solve_ik_arm_7dof


def run_post_collection_pipeline(cfg: dict) -> None:
    """Run optional MolmoAct → LeRobot conversion (and optional HF upload) after collection."""
    storage_cfg = cfg.get("storage", {}) or {}
    lerobot_cfg = cfg.get("lerobot", {}) or {}
    auto_convert = bool(lerobot_cfg.get("auto_convert", False))
    auto_upload = bool(lerobot_cfg.get("auto_upload", False))
    if not auto_convert and not auto_upload:
        return
    if auto_upload and not auto_convert:
        logging.info(
            "Skipping post-collection upload because lerobot.auto_convert is false. "
            "Enable lerobot.auto_convert to run conversion+upload pipeline."
        )
        return

    base_dir = Path(storage_cfg["base_dir"]).expanduser()
    task_directory = storage_cfg["task_directory"]
    json_data_dir = base_dir / task_directory
    lerobot_dir = base_dir / f"{task_directory}_lerobot_v30"
    repo_id = lerobot_cfg.get("hf_repo_id", storage_cfg.get("hf_repo_id"))
    if auto_upload and not repo_id:
        raise ValueError(
            "lerobot.hf_repo_id is required when lerobot.auto_upload is true."
        )

    converter_script = _REPO_ROOT / "molmoact_to_lerobot.py"
    if not converter_script.exists():
        raise FileNotFoundError(f"Converter script not found: {converter_script}")
    if not json_data_dir.exists():
        raise FileNotFoundError(f"Collected json directory not found: {json_data_dir}")
    if lerobot_dir.exists():
        remove_dir = input(
            f"The LeRobot output directory {lerobot_dir} already exists. "
            "Do you want to remove it and continue? (y/n): "
        ).strip().lower()
        if remove_dir == "y":
            shutil.rmtree(lerobot_dir)
            lerobot_dir.mkdir(parents=True, exist_ok=True)
            logging.info("Removed and recreated output directory: %s", lerobot_dir)
        elif remove_dir == "n":
            logging.info("Conversion canceled by user because output directory already exists.")
            return
        else:
            logging.info("Invalid input. Conversion canceled.")
            return

    convert_cmd = [
        sys.executable,
        str(converter_script),
        "--data_dir",
        str(json_data_dir),
        "--output_dir",
        str(lerobot_dir),
        "--repo_id",
        str(repo_id or "molmoact_v30"),
        "--fps",
        str(lerobot_cfg.get("fps", storage_cfg.get("lerobot_fps", cfg.get("hz", 30)))),
        "--robot_type",
        str(
            lerobot_cfg.get(
                "robot_type", storage_cfg.get("lerobot_robot_type", "molmoact_dual_arm")
            )
        ),
        "--skip_initial_frames",
        str(lerobot_cfg.get("skip_initial_frames", storage_cfg.get("lerobot_skip_initial_frames", 0))),
        "--action_mode",
        str(
            lerobot_cfg.get(
                "action_mode", storage_cfg.get("lerobot_action_mode", "next_joint_fields")
            )
        ),
        "--task_instruction",
        str(storage_cfg.get("language_instruction", "perform the task")),
        "--sanitize_online_viz_meta",
        str(
            int(
                bool(
                    lerobot_cfg.get(
                        "sanitize_online_viz_meta",
                        storage_cfg.get("sanitize_online_viz_meta", True),
                    )
                )
            )
        ),
        "--vcodec",
        str(lerobot_cfg.get("vcodec", "h264")),
        "--image_writer_processes",
        str(int(lerobot_cfg.get("image_writer_processes", 8))),
        "--image_writer_threads",
        str(int(lerobot_cfg.get("image_writer_threads", 8))),
        "--parallel_encoding",
        str(int(bool(lerobot_cfg.get("parallel_encoding", True)))),
        "--upload_to_hf",
        str(int(auto_upload)),
        "--delete_local_after_upload",
        str(
            int(
                bool(
                    lerobot_cfg.get(
                        "delete_local_after_upload",
                        storage_cfg.get("delete_local_after_upload", True),
                    )
                )
            )
        ),
    ]
    logging.info("Running post-collection pipeline: %s", " ".join(convert_cmd))
    subprocess.run(convert_cmd, check=True, cwd=str(_REPO_ROOT))
    logging.info("Post-collection pipeline completed successfully.")


def main(args: argparse.Namespace):
    ids = get_device_ids()
    print(f"Found {len(ids)} camera devices")
    print(ids)

    # Load configs
    config_path = args.config_path
    config_path = str(Path(config_path).expanduser().resolve())
    cfg = OmegaConf.to_container(
        OmegaConf.load(config_path), resolve=True
    )

    # Initialize data saver and keyboard interface
    data_saver = DataSaver(
        save_dir=cfg["storage"]["base_dir"],
        task_directory=cfg["storage"]["task_directory"],
        language_instruction=cfg["storage"]["language_instruction"],
        saver_max_workers=cfg["storage"].get("saver_max_workers"),
        png_compress_level=cfg["storage"].get("png_compress_level", 1),
    )

    camera_cfg = cfg["sensors"]["cameras"]
    cameras = {
        "left_camera": RealSenseCamera(camera_cfg["left_camera"]["device_id"]),
        "front_camera": RealSenseCamera(camera_cfg["front_camera"]["device_id"]),
        "right_camera": RealSenseCamera(camera_cfg["right_camera"]["device_id"]),
    }


    logging.info("=== VR Control System Starting ===")
    logging.info(f"Server Address       : {args.server}")
    logging.info(f"Local (UPC) IP       : {args.local_ip}:{Settings.vr_control_local_port}")
    logging.info(f"Meta Quest IP        : {args.meta_quest_ip}:{Settings.vr_control_meta_quest_port}")
    logging.info(f"Use Gripper          : {'No' if args.no_gripper else 'Yes'}")
    logging.info(f"RB-Y1 gRPC Address   : {args.rby1}")
    logging.info(f"RB-Y1 Model          : {args.rby1_model}")
    logging.info(f"Use Head             : {'No' if args.no_head else 'Yes'}")

    socket = open_zmq_pub_socket(args.server)
    robot = connect_rby1(args.rby1, args.rby1_model, args.no_head)
    model = robot.model()
    setup_meta_quest_udp_communication(args.local_ip, Settings.vr_control_local_port, args.meta_quest_ip,
                                       Settings.vr_control_meta_quest_port, lambda: robot.power_off(".*"))

    gripper = None
    gripper_cmd_target = None
    if not args.no_gripper:
        for arm in ["left", "right"]:
            if not robot.set_tool_flange_output_voltage(arm, 12):
                logging.error(f"Failed to supply 12V to tool flange. ({arm})")
        time.sleep(0.5)
        gripper = Gripper()
        if not gripper.initialize(verbose=True):
            exit(1)
        time.sleep(0.3)
        gripper.homing()
        gripper.start()
        gripper_cmd_target = np.array([1.0, 1.0], dtype=np.float64)
        gripper.set_normalized_target(gripper_cmd_target)

    pub_thread = threading.Thread(target=publish_gv, args=(socket,), daemon=True)
    pub_thread.start()

    dyn_robot = robot.get_dynamics()
    dyn_state = dyn_robot.make_state(["base", "link_torso_5", "link_right_arm_6", "link_left_arm_6"],
                                     SystemContext.robot_model.robot_joint_names)
    base_link_idx, link_torso_5_idx, link_right_arm_6_idx, link_left_arm_6_idx = 0, 1, 2, 3

    joint_names = list(SystemContext.robot_model.robot_joint_names)
    left_arm_joint_indices = [i for i, name in enumerate(joint_names) if name.startswith("left_arm")]
    right_arm_joint_indices = [i for i, name in enumerate(joint_names) if name.startswith("right_arm")]
    # Stable chain order; 16-float buffer: left (7 arm + 1 gripper) + right (7 arm + 1 gripper).
    left_arm_joint_indices.sort(key=lambda i: joint_names[i])
    right_arm_joint_indices.sort(key=lambda i: joint_names[i])

    if len(left_arm_joint_indices) != 7 or len(right_arm_joint_indices) != 7:
        logging.warning(
            "Expected 7 left/right arm joints, got "
            f"left={len(left_arm_joint_indices)} right={len(right_arm_joint_indices)}. "
            "Joint extraction may be incorrect."
        )

    def extract_left_right_joint_payload(q_full: np.ndarray) -> np.ndarray:
        """16 floats: left_joint[8] + right_joint[8]. Gripper scalars: 1=open, 0=close (dataset convention)."""
        left_q = q_full[left_arm_joint_indices]
        right_q = q_full[right_arm_joint_indices]
        if left_q.shape[0] > 7:
            left_q = left_q[:7]
        if right_q.shape[0] > 7:
            right_q = right_q[:7]
        if gripper is not None:
            g = gripper.get_normalized_target().copy()
            # Same indexing as teleop: g[0]←right trigger, g[1]←1−left trigger (asymmetric VR).
            raw_l = float(np.clip(g[1], 0.0, 1.0))
            raw_r = float(np.clip(g[0], 0.0, 1.0))
            left_g = float(np.clip(1.0 - raw_l, 0.0, 1.0))
            right_g = float(np.clip(1.0 - raw_r, 0.0, 1.0))
        else:
            right_g, left_g = 0.0, 0.0
        left8 = np.concatenate([left_q, np.array([left_g], dtype=np.float32)])
        right8 = np.concatenate([right_q, np.array([right_g], dtype=np.float32)])
        return np.concatenate([left8, right8]).astype(np.float32)

    def move_robot_to_ready_pose():
        # Matches the existing ready pose used in the prior VR-button-based flow.
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
                    rby.JointPositionCommandBuilder()
                    .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1))
                    .set_position(ready_pose)
                    .set_minimum_time(2)
                )
            )
            if not args.no_head:
                cbc.set_head_command(
                    rby.JointPositionCommandBuilder()
                    .set_position([0.] * len(SystemContext.robot_model.head_idx))
                    .set_minimum_time(2)
                )
            robot.send_command(rby.RobotCommandBuilder().set_command(cbc)).get()
            if gripper is not None:
                # Same convention as replay / JSON: 1 = open (Gripper inverts internally).
                gripper_cmd_target = np.array([1.0, 1.0], dtype=np.float64)
                gripper.set_normalized_target(gripper_cmd_target)

    max_episode_length = int(cfg["collection"]["max_episode_length"])
    episode_frame_count = 0  # raw frames, not observations (observations start on frame 2)
    prev_capture = None  # last frame: cameras + joint_positions (commanded 16 floats after IK)
    last_arm_joints = None
    latest_dashboard_cameras = {"left_camera": None, "front_camera": None, "right_camera": None}

    next_time = time.monotonic()
    stream = None
    torso_reset = False
    right_reset = False
    left_reset = False
    now = 0
    right_primary_prev = False
    right_secondary_prev = False
    num_traj = data_saver.traj_count if data_saver.traj_count > 0 else 1
    saver_thread = EpisodeSaverThread(data_saver)
    saver_thread.start()
    move_robot_to_ready_pose()
    while True:
        # logging.info(f"Loop execution time: {time.monotonic() - now:.4f}s")
        now = time.monotonic()
        if now < next_time:
            time.sleep(next_time - now)
        next_time += Settings.dt

        # --- Episode control via VR controller A/B ---
        if num_traj > cfg["storage"]["episodes"] and not SystemContext.vr_state.is_initialized:
            logging.info("Reached max episode count; exiting.")
            break

        right_hand = SystemContext.vr_state.controller_state.get("hands", {}).get("right", {})
        right_buttons = right_hand.get("buttons", {})
        right_primary = bool(right_buttons.get("primaryButton", False))
        right_secondary = bool(right_buttons.get("secondaryButton", False))
        right_side_active = bool(right_buttons.get("grip", 0.0) > 0.8)

        left_hand = SystemContext.vr_state.controller_state.get("hands", {}).get("left", {})
        left_buttons = left_hand.get("buttons", {})
        left_side_active = bool(left_buttons.get("grip", 0.0) > 0.8)
        collect_active = right_side_active or left_side_active

        # Rising-edge detection to avoid repeated triggers while holding the button.
        a_rise = right_primary and not right_primary_prev
        b_rise = right_secondary and not right_secondary_prev
        right_primary_prev = right_primary
        right_secondary_prev = right_secondary

        if b_rise and SystemContext.vr_state.is_initialized:
            # Stop teleop and discard previously collected episode.
            logging.info("Right B pressed. Discarding current episode.")
            if stream is not None:
                stream.cancel()
                stream = None
            SystemContext.vr_state.is_initialized = False
            SystemContext.vr_state.is_stopped = False
            move_robot_to_ready_pose()
            data_saver.reset_buffer()
            prev_capture = None
            episode_frame_count = 0
            last_arm_joints = None
            latest_dashboard_cameras = {"left_camera": None, "front_camera": None, "right_camera": None}
            continue

        if a_rise:
            if not SystemContext.vr_state.is_initialized:
                # Start teleop + data collection.
                logging.info(f"Right A pressed. Starting teleop + data collection for episode {num_traj}.")
                if stream is not None:
                    stream.cancel()
                    stream = None
                prev_capture = None
                episode_frame_count = 0
                last_arm_joints = None
                latest_dashboard_cameras = {"left_camera": None, "front_camera": None, "right_camera": None}
                SystemContext.vr_state.is_initialized = True
                SystemContext.vr_state.is_stopped = False
                continue
            else:
                # Save episode and reset robot to ready pose.
                logging.info(f"Right A pressed. Saving episode {num_traj} + resetting robot.")
                if stream is not None:
                    stream.cancel()
                    stream = None
                SystemContext.vr_state.is_initialized = False
                SystemContext.vr_state.is_stopped = False

                if data_saver.buffer:
                    saver_thread.save_episode(data_saver.buffer.copy())
                    num_traj += 1
                else:
                    logging.info("No frames collected; skipping save.")

                data_saver.reset_buffer()
                prev_capture = None
                episode_frame_count = 0
                last_arm_joints = None
                latest_dashboard_cameras = {"left_camera": None, "front_camera": None, "right_camera": None}
                move_robot_to_ready_pose()
                continue

        if "hands" in SystemContext.vr_state.controller_state and gripper is not None:
            if gripper_cmd_target is None:
                gripper_cmd_target = np.array(gripper.get_normalized_target(), dtype=np.float64)
            updated_gripper = False
            if "right" in SystemContext.vr_state.controller_state["hands"] and right_side_active:
                right_controller = SystemContext.vr_state.controller_state["hands"]["right"]
                # Keep teleop/dataset convention aligned: trigger 1.0 means open in saved data.
                gripper_cmd_target[0] = 1.0 - float(right_controller["buttons"]["trigger"])
                updated_gripper = True
            if "left" in SystemContext.vr_state.controller_state["hands"] and left_side_active:
                left_controller = SystemContext.vr_state.controller_state["hands"]["left"]
                gripper_cmd_target[1] = 1.0 - float(left_controller["buttons"]["trigger"])
                updated_gripper = True
            if updated_gripper:
                gripper.set_normalized_target(gripper_cmd_target)

        if SystemContext.vr_state.joint_positions.size == 0:
            # No state updates: if we keep a live stream without sending, the SDK expires it and the next send aborts.
            if stream is not None:
                try:
                    stream.cancel()
                except Exception:
                    pass
                stream = None
            continue

        if not SystemContext.vr_state.is_initialized:
            if stream is not None:
                try:
                    stream.cancel()
                except Exception:
                    pass
                stream = None
            continue

        # logging.info(f"{SystemContext.vr_state.center_of_mass = }")

        dyn_state.set_q(SystemContext.vr_state.joint_positions.copy())
        dyn_robot.compute_forward_kinematics(dyn_state)

        SystemContext.vr_state.base_pose = dyn_robot.compute_transformation(dyn_state, base_link_idx, link_torso_5_idx)
        SystemContext.vr_state.torso_current_pose = dyn_robot.compute_transformation(dyn_state, base_link_idx,
                                                                                     link_torso_5_idx)
        SystemContext.vr_state.right_ee_current_pose = dyn_robot.compute_transformation(dyn_state, base_link_idx,
                                                                                        link_right_arm_6_idx) @ Settings.T_hand_offset
        SystemContext.vr_state.left_ee_current_pose = dyn_robot.compute_transformation(dyn_state, base_link_idx,
                                                                                       link_left_arm_6_idx) @ Settings.T_hand_offset

        trans_12 = dyn_robot.compute_transformation(dyn_state, 1, 2)
        trans_13 = dyn_robot.compute_transformation(dyn_state, 1, 3)
        center = (trans_12[:3, 3] + trans_13[:3, 3]) / 2
        yaw = np.atan2(center[1], center[0])
        pitch = np.atan2(-center[2], center[0]) - np.deg2rad(10)
        yaw = np.clip(yaw, -np.deg2rad(29), np.deg2rad(29))
        pitch = np.clip(pitch, -np.deg2rad(19), np.deg2rad(89))

        q_arms = extract_left_right_joint_payload(SystemContext.vr_state.joint_positions)
        last_arm_joints = q_arms

        # --- Capture camera + joints for the current frame ---
        capture_ok = False
        if not collect_active:
            prev_capture = None
        else:
            try:
                left_rgb, _ = cameras["left_camera"].read()
                front_rgb, _ = cameras["front_camera"].read()
                right_rgb, _ = cameras["right_camera"].read()

                latest_dashboard_cameras = {
                    "left_camera": left_rgb,
                    "front_camera": front_rgb,
                    "right_camera": right_rgb,
                }

                current_capture = {
                    "left_camera_rgb": left_rgb,
                    "right_camera_rgb": right_rgb,
                    "front_camera_rgb": front_rgb,
                }
                capture_ok = True
            except Exception as exc:
                logging.warning(f"Frame capture failed: {exc}")

        # Tracking
        if stream is None:
            if robot.wait_for_control_ready(0):
                stream = robot.create_command_stream()
                SystemContext.vr_state.mobile_linear_velocity = np.array([0.0, 0.0])
                SystemContext.vr_state.mobile_angular_velocity = 0.
                SystemContext.vr_state.is_right_following = False
                SystemContext.vr_state.is_left_following = False
                SystemContext.vr_state.base_start_pose = SystemContext.vr_state.base_pose
                SystemContext.vr_state.torso_locked_pose = SystemContext.vr_state.torso_current_pose
                SystemContext.vr_state.right_hand_locked_pose = SystemContext.vr_state.right_ee_current_pose
                SystemContext.vr_state.left_hand_locked_pose = SystemContext.vr_state.left_ee_current_pose

        if "hands" in SystemContext.vr_state.controller_state:
            if "right" in SystemContext.vr_state.controller_state["hands"]:
                right_controller = SystemContext.vr_state.controller_state["hands"]["right"]
                thumbstick_axis = right_controller["buttons"]["thumbstickAxis"]
                acc = np.array([thumbstick_axis[1], thumbstick_axis[0]])
                SystemContext.vr_state.mobile_linear_velocity += Settings.mobile_linear_acceleration_gain * acc
                # SystemContext.vr_state.mobile_angular_velocity += Settings.mobile_angular_acceleration_gain * \
                #                                                   thumbstick_axis[0]
                SystemContext.vr_state.right_controller_current_pose = T_conv.T @ pose_to_se3(
                    right_controller["position"],
                    right_controller["rotation"]) @ T_conv

                trigger_pressed = right_controller["buttons"]["grip"] > 0.8
                if SystemContext.vr_state.is_right_following and not trigger_pressed:
                    SystemContext.vr_state.is_right_following = False
                if not SystemContext.vr_state.is_right_following and trigger_pressed:
                    SystemContext.vr_state.right_controller_start_pose = SystemContext.vr_state.right_controller_current_pose
                    SystemContext.vr_state.right_ee_start_pose = SystemContext.vr_state.right_ee_current_pose
                    SystemContext.vr_state.is_right_following = True
                    right_reset = True
            else:
                SystemContext.vr_state.is_right_following = False

            if "left" in SystemContext.vr_state.controller_state["hands"]:
                left_controller = SystemContext.vr_state.controller_state["hands"]["left"]
                thumbstick_axis = left_controller["buttons"]["thumbstickAxis"]
                # SystemContext.vr_state.mobile_linear_velocity += Settings.mobile_linear_acceleration_gain * \
                #                                                  thumbstick_axis[1]
                SystemContext.vr_state.mobile_angular_velocity += Settings.mobile_angular_acceleration_gain * \
                                                                  thumbstick_axis[0]
                SystemContext.vr_state.left_controller_current_pose = T_conv.T @ pose_to_se3(
                    left_controller["position"],
                    left_controller["rotation"]) @ T_conv

                trigger_pressed = left_controller["buttons"]["grip"] > 0.8
                if SystemContext.vr_state.is_left_following and not trigger_pressed:
                    SystemContext.vr_state.is_left_following = False
                if not SystemContext.vr_state.is_left_following and trigger_pressed:
                    SystemContext.vr_state.left_controller_start_pose = SystemContext.vr_state.left_controller_current_pose
                    SystemContext.vr_state.left_ee_start_pose = SystemContext.vr_state.left_ee_current_pose
                    SystemContext.vr_state.is_left_following = True
                    left_reset = True
            else:
                SystemContext.vr_state.is_left_following = False

            if "head" in SystemContext.vr_state.controller_state:
                head_controller = SystemContext.vr_state.controller_state["head"]
                SystemContext.vr_state.head_controller_current_pose = T_conv.T @ pose_to_se3(
                    head_controller["position"],
                    head_controller["rotation"]) @ T_conv

                following = SystemContext.vr_state.is_right_following and SystemContext.vr_state.is_left_following
                if SystemContext.vr_state.is_torso_following and not following:
                    SystemContext.vr_state.is_torso_following = False
                if not SystemContext.vr_state.is_torso_following and following:
                    SystemContext.vr_state.head_controller_start_pose = SystemContext.vr_state.head_controller_current_pose
                    SystemContext.vr_state.torso_start_pose = SystemContext.vr_state.torso_current_pose
                    SystemContext.vr_state.is_torso_following = True
                    torso_reset = True
            else:
                SystemContext.vr_state.is_torso_following = False

        SystemContext.vr_state.mobile_linear_velocity -= Settings.mobile_linear_damping_gain * SystemContext.vr_state.mobile_linear_velocity
        SystemContext.vr_state.mobile_angular_velocity -= Settings.mobile_angular_damping_gain * SystemContext.vr_state.mobile_angular_velocity

        if stream:
            try:
                if SystemContext.vr_state.is_right_following:
                    diff = np.linalg.inv(
                        SystemContext.vr_state.right_controller_start_pose) @ SystemContext.vr_state.right_controller_current_pose

                    T_global2start = np.identity(4)
                    T_global2start[:3, :3] = R.from_euler('y', 90, degrees=True).as_matrix()
                    diff_global = T_global2start @ diff @ T_global2start.T

                    T = np.identity(4)
                    T[:3, :3] = SystemContext.vr_state.right_ee_start_pose[:3, :3]
                    right_T = SystemContext.vr_state.right_ee_start_pose @ diff_global
                    SystemContext.vr_state.right_hand_locked_pose = right_T
                else:
                    right_T = SystemContext.vr_state.right_hand_locked_pose

                if SystemContext.vr_state.is_left_following:
                    diff = np.linalg.inv(
                        SystemContext.vr_state.left_controller_start_pose) @ SystemContext.vr_state.left_controller_current_pose

                    T_global2start = np.identity(4)
                    T_global2start[:3, :3] = R.from_euler('y', 90, degrees=True).as_matrix()
                    diff_global = T_global2start @ diff @ T_global2start.T

                    T = np.identity(4)
                    T[:3, :3] = SystemContext.vr_state.left_ee_start_pose[:3, :3]
                    left_T = SystemContext.vr_state.left_ee_start_pose @ diff_global
                    SystemContext.vr_state.left_hand_locked_pose = left_T
                else:
                    left_T = SystemContext.vr_state.left_hand_locked_pose

                if SystemContext.vr_state.is_torso_following:
                    diff = np.linalg.inv(
                        SystemContext.vr_state.head_controller_start_pose) @ SystemContext.vr_state.head_controller_current_pose

                    T = np.identity(4)
                    T[:3, :3] = SystemContext.vr_state.torso_start_pose[:3, :3]
                    torso_T = SystemContext.vr_state.torso_start_pose @ diff
                    SystemContext.vr_state.torso_locked_pose = torso_T
                else:
                    torso_T = SystemContext.vr_state.torso_locked_pose

                T_hi = np.linalg.inv(Settings.T_hand_offset)

                if args.whole_body:
                    ctrl_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_minimum_time(Settings.dt * 1.01)
                        .set_joint_stiffness([400.] * 6 + [60] * 7 + [60] * 7)
                        .set_joint_torque_limit([500] * 6 + [30] * 7 + [30] * 7)
                        .add_joint_limit("right_arm_3", -2.6, -0.5)
                        .add_joint_limit("right_arm_5", 0.2, 1.9)
                        .add_joint_limit("left_arm_3", -2.6, -0.5)
                        .add_joint_limit("left_arm_5", 0.2, 1.9)
                        .add_joint_limit("torso_1", -0.523598776, 1.3)
                        .add_joint_limit("torso_2", -2.617993878, -0.2)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_stop_joint_position_tracking_error(0)
                        .set_reset_reference(right_reset | left_reset | torso_reset)
                    )
                    ctrl_builder.add_target("base", "link_torso_5", torso_T, 1, np.pi * 0.5, 10, np.pi * 20)
                    ctrl_builder.add_target("base", "link_right_arm_6", right_T @ T_hi,
                                            2, np.pi * 2, 20, np.pi * 80)
                    ctrl_builder.add_target("base", "link_left_arm_6", left_T @ T_hi,
                                            2, np.pi * 2, 20, np.pi * 80)
                    q_cmd_16 = extract_left_right_joint_payload(SystemContext.vr_state.joint_positions.copy())

                else:
                    q_full = SystemContext.vr_state.joint_positions.copy().astype(np.float64)
                    T_des_r = (right_T @ T_hi).astype(np.float64)
                    T_des_l = (left_T @ T_hi).astype(np.float64)
                    if SystemContext.vr_state.is_right_following:
                        q_full = solve_ik_arm_7dof(
                            dyn_robot,
                            dyn_state,
                            q_full,
                            base_link_idx,
                            link_right_arm_6_idx,
                            np.asarray(right_arm_joint_indices, dtype=np.int64),
                            T_des_r,
                        )
                    if SystemContext.vr_state.is_left_following:
                        q_full = solve_ik_arm_7dof(
                            dyn_robot,
                            dyn_state,
                            q_full,
                            base_link_idx,
                            link_left_arm_6_idx,
                            np.asarray(left_arm_joint_indices, dtype=np.int64),
                            T_des_l,
                        )
                    q_cmd_16 = extract_left_right_joint_payload(q_full)

                    right_q = q_full[np.array(right_arm_joint_indices)][:7]
                    left_q = q_full[np.array(left_arm_joint_indices)][:7]
                    right_builder = (
                        rby.JointPositionCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_minimum_time(Settings.dt * 1.01)
                        .set_position(right_q.tolist())
                    )
                    left_builder = (
                        rby.JointPositionCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_minimum_time(Settings.dt * 1.01)
                        .set_position(left_q.tolist())
                    )
                    ctrl_builder = (
                        rby.BodyComponentBasedCommandBuilder()
                        .set_right_arm_command(right_builder)
                        .set_left_arm_command(left_builder)
                    )

                torso_reset = False
                right_reset = False
                left_reset = False

                if capture_ok:
                    current_capture["joint_positions"] = q_cmd_16

                if capture_ok and prev_capture is not None:
                    obs_next = dict(prev_capture)
                    obs_next["next_joint"] = q_cmd_16
                    data_saver.add_observation(obs_next)

                robot_cmd = rby.RobotCommandBuilder().set_command(
                    rby.ComponentBasedCommandBuilder()
                    # .set_head_command(
                    #     rby.JointPositionCommandBuilder()
                    #     .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                    #     .set_position([float(yaw), float(pitch)])
                    #     .set_minimum_time(Settings.dt * 1.01)
                    # )
                    # .set_mobility_command(
                    #     rby.SE2VelocityCommandBuilder()
                    #     .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                    #     .set_velocity(-SystemContext.vr_state.mobile_linear_velocity,
                    #                   -SystemContext.vr_state.mobile_angular_velocity)
                    #     .set_minimum_time(Settings.dt * 1.01)
                    # )
                    .set_body_command(
                        ctrl_builder
                    )
                )
                stream = _send_robot_command_with_stream_retry(robot, stream, robot_cmd)
            except Exception as e:
                logging.error("Command stream error: %s", e)
                if stream is not None:
                    try:
                        stream.cancel()
                    except Exception:
                        pass
                stream = None
                continue

        if capture_ok:
            prev_capture = current_capture
            episode_frame_count += 1

    saver_thread.stop()
    saver_thread.join()
    logging.info("Data collection complete.")
    run_post_collection_pipeline(cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RB-Y1 VR Control Launcher")

    parser.add_argument(
        "-s", "--server", type=str, default="tcp://*:5555",
        help="ZMQ server address for the UPC (default: tcp://*:5555)"
    )
    parser.add_argument(
        "--local_ip", required=True, type=str,
        help="Local Wi-Fi (or LAN) IP address of the UPC"
    )
    parser.add_argument(
        "--meta_quest_ip", required=True, type=str,
        help="Wi-Fi (or LAN) IP address of the Meta Quest"
    )
    parser.add_argument(
        "--no_gripper", action="store_true",
        help="Run without gripper support"
    )
    parser.add_argument(
        "--rby1", default="192.168.30.1:50051", type=str,
        help="gRPC address of the RB-Y1 robot (default: 192.168.30.1:50051)"
    )
    parser.add_argument(
        "--rby1_model", default="m", type=str,
        help="Model type of the RB-Y1 robot (default: m)"
    )
    parser.add_argument(
        "--no_head", action="store_true", 
        help="Run without controlling the head"
    )
    parser.add_argument(
        "--whole_body", action="store_true",
        help="Use a whole-body optimization formulation (single control for all joints)"
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
        help="Path to config.yaml (default: repo config.yaml)",
    )

    args = parser.parse_args()
    main(args)