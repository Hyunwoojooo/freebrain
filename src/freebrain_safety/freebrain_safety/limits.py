"""Pure-Python safety limit checks (no ROS dependency)."""

from dataclasses import dataclass, field
import math

from .config import SafetyConfig, JOINT_NAMES, N_JOINTS


@dataclass
class SafetyCheckResult:
    all_ok: bool = True
    joint_limits_ok: bool = True
    velocity_ok: bool = True
    current_ok: bool = True
    workspace_ok: bool = True
    violations: list[str] = field(default_factory=list)


def check_joint_limits(
    positions: tuple[float, ...] | list[float],
    config: SafetyConfig,
) -> tuple[bool, list[str]]:
    violations = []
    for i in range(N_JOINTS):
        lo = config.joint_limits.lower[i]
        hi = config.joint_limits.upper[i]
        if positions[i] < lo or positions[i] > hi:
            violations.append(
                f"{JOINT_NAMES[i]} pos={positions[i]:.4f} outside [{lo:.4f}, {hi:.4f}]"
            )
    return (len(violations) == 0, violations)


def check_velocity_limits(
    velocities: tuple[float, ...] | list[float],
    config: SafetyConfig,
) -> tuple[bool, list[str]]:
    violations = []
    for i in range(N_JOINTS):
        if abs(velocities[i]) > config.max_velocities[i]:
            violations.append(
                f"{JOINT_NAMES[i]} vel={velocities[i]:.4f} exceeds ±{config.max_velocities[i]:.4f}"
            )
    return (len(violations) == 0, violations)


def check_torque_limits(
    torques: tuple[float, ...] | list[float],
    config: SafetyConfig,
) -> tuple[bool, list[str]]:
    violations = []
    for i in range(N_JOINTS):
        if abs(torques[i]) > config.max_torques[i]:
            violations.append(
                f"{JOINT_NAMES[i]} torque={torques[i]:.4f} exceeds ±{config.max_torques[i]:.4f}"
            )
    return (len(violations) == 0, violations)


def check_workspace(
    ee_position: tuple[float, float, float] | list[float],
    config: SafetyConfig,
) -> tuple[bool, list[str]]:
    x, y, z = ee_position[0], ee_position[1], ee_position[2]
    r = math.sqrt(x * x + y * y)
    violations = []
    if r > config.workspace_radius_max:
        violations.append(f"EE radius={r:.4f} exceeds max={config.workspace_radius_max:.4f}")
    if r < config.workspace_radius_min:
        violations.append(f"EE radius={r:.4f} below min={config.workspace_radius_min:.4f}")
    if z < config.workspace_z_min:
        violations.append(f"EE z={z:.4f} below min={config.workspace_z_min:.4f}")
    if z > config.workspace_z_max:
        violations.append(f"EE z={z:.4f} exceeds max={config.workspace_z_max:.4f}")
    return (len(violations) == 0, violations)


def check_all(
    positions: tuple[float, ...] | list[float],
    velocities: tuple[float, ...] | list[float],
    torques: tuple[float, ...] | list[float],
    ee_position: tuple[float, float, float] | list[float],
    config: SafetyConfig,
) -> SafetyCheckResult:
    result = SafetyCheckResult()

    ok, v = check_joint_limits(positions, config)
    result.joint_limits_ok = ok
    result.violations.extend(v)

    ok, v = check_velocity_limits(velocities, config)
    result.velocity_ok = ok
    result.violations.extend(v)

    ok, v = check_torque_limits(torques, config)
    result.current_ok = ok
    result.violations.extend(v)

    ok, v = check_workspace(ee_position, config)
    result.workspace_ok = ok
    result.violations.extend(v)

    result.all_ok = (
        result.joint_limits_ok
        and result.velocity_ok
        and result.current_ok
        and result.workspace_ok
    )
    return result
