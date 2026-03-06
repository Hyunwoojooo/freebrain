"""Forward and inverse kinematics for OpenManipulator-X (pure Python, no ROS)."""

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class KinematicsConfig:
    """DH-like kinematic parameters for the 4-DOF arm."""
    joint_origins: List[Tuple[float, float, float]]
    joint_axes: List[Tuple[float, float, float]]
    ee_offset: Tuple[float, float, float]
    jacobian_eps: float = 1e-4
    damping: float = 0.01


# OpenManipulator-X default config (from MJCF model)
OM_X_CONFIG = KinematicsConfig(
    joint_origins=[
        (0.012, 0.0, 0.017),
        (0.0, 0.0, 0.0595),
        (0.024, 0.0, 0.128),
        (0.124, 0.0, 0.0),
    ],
    joint_axes=[
        (0.0, 0.0, 1.0),  # joint1: yaw
        (0.0, 1.0, 0.0),  # joint2: pitch
        (0.0, 1.0, 0.0),  # joint3: pitch
        (0.0, 1.0, 0.0),  # joint4: pitch
    ],
    ee_offset=(0.126, 0.0, 0.0),
)


def _mat_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = (
                a[i][0] * b[0][j]
                + a[i][1] * b[1][j]
                + a[i][2] * b[2][j]
                + a[i][3] * b[3][j]
            )
    return out


def _eye4() -> List[List[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _trans(x: float, y: float, z: float) -> List[List[float]]:
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _rot_axis(axis: Tuple[float, float, float], angle: float) -> List[List[float]]:
    ax, ay, az = axis
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm == 0.0:
        return _eye4()
    ax /= norm
    ay /= norm
    az /= norm
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return [
        [t * ax * ax + c, t * ax * ay - s * az, t * ax * az + s * ay, 0.0],
        [t * ax * ay + s * az, t * ay * ay + c, t * ay * az - s * ax, 0.0],
        [t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def fk_position(
    joint_angles: Sequence[float],
    config: KinematicsConfig = OM_X_CONFIG,
) -> Tuple[float, float, float]:
    """Compute end-effector (x, y, z) from 4 arm joint angles."""
    if len(joint_angles) < len(config.joint_origins):
        return 0.0, 0.0, 0.0
    transform = _eye4()
    for idx, origin in enumerate(config.joint_origins):
        transform = _mat_mul(transform, _trans(*origin))
        transform = _mat_mul(transform, _rot_axis(config.joint_axes[idx], joint_angles[idx]))
    transform = _mat_mul(transform, _trans(*config.ee_offset))
    return transform[0][3], transform[1][3], transform[2][3]


def numeric_jacobian(
    joint_angles: Sequence[float],
    config: KinematicsConfig = OM_X_CONFIG,
) -> List[List[float]]:
    """Compute 3xN numeric Jacobian via finite differences."""
    base = fk_position(joint_angles, config)
    n = len(config.joint_origins)
    jacobian = [[0.0] * n for _ in range(3)]
    eps = config.jacobian_eps if abs(config.jacobian_eps) > 1e-12 else 1e-4
    for idx in range(n):
        perturbed = list(joint_angles)
        perturbed[idx] += eps
        p = fk_position(perturbed, config)
        jacobian[0][idx] = (p[0] - base[0]) / eps
        jacobian[1][idx] = (p[1] - base[1]) / eps
        jacobian[2][idx] = (p[2] - base[2]) / eps
    return jacobian


def _invert_3x3(m: List[List[float]]) -> Tuple[bool, List[List[float]]]:
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-9:
        return False, [[0.0] * 3 for _ in range(3)]
    inv_det = 1.0 / det
    return True, [
        [(e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det],
        [(f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det],
        [(d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det],
    ]


def cartesian_to_joint_delta(
    delta_xyz: Tuple[float, float, float],
    joint_angles: Sequence[float],
    config: KinematicsConfig = OM_X_CONFIG,
) -> List[float]:
    """Compute joint angle deltas for a desired Cartesian displacement (damped pseudo-inverse)."""
    jacobian = numeric_jacobian(joint_angles, config)
    n = len(config.joint_origins)
    # J * J^T (3x3)
    jj_t = [[0.0] * 3 for _ in range(3)]
    for r in range(3):
        for c in range(3):
            v = 0.0
            for j in range(n):
                v += jacobian[r][j] * jacobian[c][j]
            jj_t[r][c] = v
    damping_sq = config.damping * config.damping
    for idx in range(3):
        jj_t[idx][idx] += damping_sq
    ok, inv = _invert_3x3(jj_t)
    if not ok:
        return [0.0] * n
    dx, dy, dz = delta_xyz
    y0 = inv[0][0] * dx + inv[0][1] * dy + inv[0][2] * dz
    y1 = inv[1][0] * dx + inv[1][1] * dy + inv[1][2] * dz
    y2 = inv[2][0] * dx + inv[2][1] * dy + inv[2][2] * dz
    dq = [0.0] * n
    for idx in range(n):
        dq[idx] = jacobian[0][idx] * y0 + jacobian[1][idx] * y1 + jacobian[2][idx] * y2
    return dq


def ik_solve(
    target_xyz: Tuple[float, float, float],
    initial_angles: Sequence[float],
    config: KinematicsConfig = OM_X_CONFIG,
    max_iter: int = 100,
    tol: float = 1e-3,
) -> Tuple[bool, List[float]]:
    """Iterative IK solver using damped pseudo-inverse.

    Returns (converged, joint_angles).
    """
    angles = list(initial_angles[:len(config.joint_origins)])
    for _ in range(max_iter):
        ee = fk_position(angles, config)
        err = (target_xyz[0] - ee[0], target_xyz[1] - ee[1], target_xyz[2] - ee[2])
        dist = math.sqrt(err[0] ** 2 + err[1] ** 2 + err[2] ** 2)
        if dist < tol:
            return True, angles
        dq = cartesian_to_joint_delta(err, angles, config)
        for i in range(len(angles)):
            angles[i] += dq[i]
    return False, angles
