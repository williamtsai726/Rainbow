"""Numerical IK for 7-DOF arms from desired link SE(3) in base frame (warm-started)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as SciR


def se3_error(T: np.ndarray, T_des: np.ndarray) -> np.ndarray:
    """6-vector: translation error + rotation error (axis-angle from R_rel)."""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    T_des = np.asarray(T_des, dtype=np.float64).reshape(4, 4)
    R_rel = T_des[:3, :3] @ T[:3, :3].T
    rvec = SciR.from_matrix(R_rel).as_rotvec()
    t_err = T_des[:3, 3] - T[:3, 3]
    return np.concatenate([t_err, rvec])


def solve_ik_arm_7dof(
    dyn_robot,
    dyn_state,
    q_full: np.ndarray,
    base_link_idx: int,
    ee_link_idx: int,
    arm_joint_indices: np.ndarray,
    T_des: np.ndarray,
    *,
    max_iters: int = 25,
    eps: float = 1e-5,
    step: float = 0.45,
    pos_tol: float = 2e-3,
    rot_tol: float = 3e-3,
) -> np.ndarray:
    """
    Damped least-squares IK on full joint vector; only arm_joint_indices are updated.

    Returns updated full ``q_full`` (copy).
    """
    q = np.asarray(q_full, dtype=np.float64).copy()
    T_des = np.asarray(T_des, dtype=np.float64).reshape(4, 4)
    idx = np.asarray(arm_joint_indices, dtype=np.int64).reshape(-1)
    if idx.size != 7:
        raise ValueError(f"Expected 7 arm joint indices, got {idx.size}")

    for _ in range(max_iters):
        dyn_state.set_q(q)
        dyn_robot.compute_forward_kinematics(dyn_state)
        T = dyn_robot.compute_transformation(dyn_state, base_link_idx, ee_link_idx)
        e = se3_error(T, T_des)
        if np.linalg.norm(e[:3]) < pos_tol and np.linalg.norm(e[3:]) < rot_tol:
            break

        J = np.zeros((6, 7), dtype=np.float64)
        for j in range(7):
            qi = int(idx[j])
            dq = np.zeros_like(q)
            dq[qi] = eps
            dyn_state.set_q(q + dq)
            dyn_robot.compute_forward_kinematics(dyn_state)
            Tp = dyn_robot.compute_transformation(dyn_state, base_link_idx, ee_link_idx)
            ep = se3_error(Tp, T_des)
            J[:, j] = (ep - e) / eps

        # Damped least squares: dq = (J^T J + λ I)^-1 J^T e
        lam = 1e-4
        dq7 = np.linalg.solve(J.T @ J + lam * np.eye(7), J.T @ e)
        q[idx] -= step * dq7

    dyn_state.set_q(q)
    dyn_robot.compute_forward_kinematics(dyn_state)
    return q
