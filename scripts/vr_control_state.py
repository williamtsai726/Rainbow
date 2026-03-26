import pickle
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(slots=True)
class VRControlState:
    joint_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    center_of_mass: np.ndarray = field(default_factory=lambda: np.array([]))
    controller_state: dict = field(default_factory=dict)

    # Flags
    is_initialized: bool = False
    is_stopped: bool = False

    # Mobile base velocities
    mobile_linear_velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    mobile_angular_velocity: float = 0.0

    # Following state
    is_torso_following: bool = False
    is_right_following: bool = False
    is_left_following: bool = False

    # Base pose
    base_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    base_start_pose: np.ndarray = field(default_factory=lambda: np.identity(4))

    # Head controller
    head_controller_start_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    head_controller_current_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    torso_start_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    torso_current_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    torso_locked_pose: np.ndarray = field(default_factory=lambda: np.identity(4))

    # Right hand controller & EE
    right_controller_start_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    right_controller_current_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    right_ee_start_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    right_ee_current_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    right_hand_locked_pose: np.ndarray = field(default_factory=lambda: np.identity(4))

    # Left hand controller & EE
    left_controller_start_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    left_controller_current_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    left_ee_start_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    left_ee_current_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    left_hand_locked_pose: np.ndarray = field(default_factory=lambda: np.identity(4))

    # Button event states
    event_right_a_pressed: bool = False
    event_right_b_pressed: bool = False
    event_left_a_pressed: bool = False
    event_left_b_pressed: bool = False
