import argparse
import logging
import zmq
import time
import threading
from dataclasses import dataclass
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

T_conv = np.array([
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
])


@dataclass(frozen=True)
class Settings:
    dt: float = 0.1
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
                    rby.JointImpedanceControlCommandBuilder()
                    .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1))
                    .set_position(ready_pose)
                    .set_stiffness([400.] * 6 + [60] * 7 + [60] * 7)
                    .set_torque_limit([500] * 6 + [30] * 7 + [30] * 7)
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


def main(args: argparse.Namespace):
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
        gripper.set_normalized_target(np.array([0.0, 0.0]))

    pub_thread = threading.Thread(target=publish_gv, args=(socket,), daemon=True)
    pub_thread.start()

    dyn_robot = robot.get_dynamics()
    dyn_state = dyn_robot.make_state(["base", "link_torso_5", "link_right_arm_6", "link_left_arm_6"],
                                     SystemContext.robot_model.robot_joint_names)
    base_link_idx, link_torso_5_idx, link_right_arm_6_idx, link_left_arm_6_idx = 0, 1, 2, 3

    next_time = time.monotonic()
    stream = None
    torso_reset = False
    right_reset = False
    left_reset = False
    while True:
        now = time.monotonic()
        if now < next_time:
            time.sleep(next_time - now)
        next_time += Settings.dt

        if "hands" in SystemContext.vr_state.controller_state:
            if "right" in SystemContext.vr_state.controller_state["hands"]:
                right_controller = SystemContext.vr_state.controller_state["hands"]["right"]
                if gripper is not None:
                    gripper_target = gripper.get_normalized_target()
                    gripper_target[0] = right_controller["buttons"]["trigger"]
                    gripper.set_normalized_target(gripper_target)
            if "left" in SystemContext.vr_state.controller_state["hands"]:
                left_controller = SystemContext.vr_state.controller_state["hands"]["left"]
                if gripper is not None:
                    gripper_target = gripper.get_normalized_target()
                    gripper_target[1] = 1. - left_controller["buttons"]["trigger"]
                    gripper.set_normalized_target(gripper_target)

        if SystemContext.vr_state.joint_positions.size == 0:
            continue

        if handle_vr_button_event(robot, args.no_head):
            if stream is not None:
                stream.cancel()
                stream = None

        if not SystemContext.vr_state.is_initialized:
            continue

        if SystemContext.vr_state.is_stopped:
            if stream is not None:
                stream.cancel()
                stream = None
            SystemContext.vr_state.is_initialized = False
            continue

        logging.info(f"{SystemContext.vr_state.center_of_mass = }")

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
                    print('a')
                    diff = np.linalg.inv(
                        SystemContext.vr_state.head_controller_start_pose) @ SystemContext.vr_state.head_controller_current_pose
                    print(SystemContext.vr_state.head_controller_start_pose)

                    T = np.identity(4)
                    T[:3, :3] = SystemContext.vr_state.torso_start_pose[:3, :3]
                    torso_T = SystemContext.vr_state.torso_start_pose @ diff
                    SystemContext.vr_state.torso_locked_pose = torso_T
                else:
                    torso_T = SystemContext.vr_state.torso_locked_pose

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
                    ctrl_builder.add_target("base", "link_right_arm_6", right_T @ np.linalg.inv(Settings.T_hand_offset),
                                            2, np.pi * 2, 20, np.pi * 80)
                    ctrl_builder.add_target("base", "link_left_arm_6", left_T @ np.linalg.inv(Settings.T_hand_offset),
                                            2, np.pi * 2, 20, np.pi * 80)

                else:
                    torso_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_minimum_time(Settings.dt * 1.01)
                        .set_joint_stiffness([400.] * 6)
                        .set_joint_torque_limit([500] * 6)
                        .add_joint_limit("torso_1", -0.523598776, 1.3)
                        .add_joint_limit("torso_2", -2.617993878, -0.2)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_stop_joint_position_tracking_error(0)
                        .set_reset_reference(torso_reset)
                    )
                    right_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_minimum_time(Settings.dt * 1.01)
                        .set_joint_stiffness([80, 80, 80, 80, 80, 80, 40])
                        .set_joint_torque_limit([30] * 7)
                        .add_joint_limit("right_arm_3", -2.6, -0.5)
                        .add_joint_limit("right_arm_5", 0.2, 1.9)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_stop_joint_position_tracking_error(0)
                        .set_reset_reference(right_reset)
                    )
                    left_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_minimum_time(Settings.dt * 1.01)
                        .set_joint_stiffness([80, 80, 80, 80, 80, 80, 40])
                        .set_joint_torque_limit([30] * 7)
                        .add_joint_limit("left_arm_3", -2.6, -0.5)
                        .add_joint_limit("left_arm_5", 0.2, 1.9)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_stop_joint_position_tracking_error(0)
                        .set_reset_reference(left_reset)
                    )
                    torso_builder.add_target("base", "link_torso_5", torso_T, 1, np.pi * 0.5, 10, np.pi * 20)
                    right_builder.add_target("base", "link_right_arm_6",
                                             right_T @ np.linalg.inv(Settings.T_hand_offset),
                                             2, np.pi * 2, 20, np.pi * 80)
                    left_builder.add_target("base", "link_left_arm_6", left_T @ np.linalg.inv(Settings.T_hand_offset),
                                            2, np.pi * 2, 20, np.pi * 80)

                    ctrl_builder = (
                        rby.BodyComponentBasedCommandBuilder()
                        .set_torso_command(torso_builder)
                        .set_right_arm_command(right_builder)
                        .set_left_arm_command(left_builder)
                    )

                torso_reset = False
                right_reset = False
                left_reset = False

                stream.send_command(
                    rby.RobotCommandBuilder().set_command(
                        rby.ComponentBasedCommandBuilder()
                        .set_head_command(
                            rby.JointPositionCommandBuilder()
                            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                            .set_position([float(yaw), float(pitch)])
                            .set_minimum_time(Settings.dt * 1.01)
                        )
                        .set_mobility_command(
                            rby.SE2VelocityCommandBuilder()
                            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                            .set_velocity(-SystemContext.vr_state.mobile_linear_velocity,
                                          -SystemContext.vr_state.mobile_angular_velocity)
                            .set_minimum_time(Settings.dt * 1.01)
                        )
                        .set_body_command(
                            ctrl_builder
                        )
                    )
                )
            except Exception as e:
                logging.error(e)
                stream = None
                exit(1)

        # ...


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
        "--rby1_model", default="a", type=str,
        help="Model type of the RB-Y1 robot (default: a)"
    )
    parser.add_argument(
        "--no_head", action="store_true", 
        help="Run without controlling the head"
    )
    parser.add_argument(
        "--whole_body", action="store_true",
        help="Use a whole-body optimization formulation (single control for all joints)"
    )

    args = parser.parse_args()
    main(args)
