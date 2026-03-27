"""End-effector pose as 6 floats: translation (3) + rotation vector (3), radians.

In JSON, each arm is often stored as 7 floats: these 6 + gripper (1=open / 0=close)."""

import numpy as np
from scipy.spatial.transform import Rotation as SciR


def pose6_from_matrix44(T: np.ndarray) -> np.ndarray:
    """4×4 homogeneous base→link transform → [tx, ty, tz, rx, ry, rz] (rotvec, rad)."""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    out = np.empty(6, dtype=np.float64)
    out[:3] = T[:3, 3]
    out[3:6] = SciR.from_matrix(T[:3, :3]).as_rotvec()
    return out


def matrix44_from_pose6(p: np.ndarray) -> np.ndarray:
    """Inverse of pose6_from_matrix44."""
    p = np.asarray(p, dtype=np.float64).reshape(6)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = p[:3]
    T[:3, :3] = SciR.from_rotvec(p[3:6]).as_matrix()
    return T
