"""Safety configuration and constants for OpenManipulator-X."""

from dataclasses import dataclass
import math


# 5-degree buffer in radians
_BUF_DEG5 = 5.0 * math.pi / 180.0  # 0.0873 rad
_BUF_GRIP = 0.005  # m (gripper is prismatic)

# Hardware joint limits (from MJCF / MoveIt config)
_HW_LOWER = (-3.14159, -1.5, -1.5, -1.7, -0.01)
_HW_UPPER = (3.14159, 1.5, 1.4, 1.97, 0.019)
_BUFFERS = (_BUF_DEG5, _BUF_DEG5, _BUF_DEG5, _BUF_DEG5, _BUF_GRIP)

JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "gripper")
N_JOINTS = 5


@dataclass(frozen=True)
class JointLimits:
    lower: tuple[float, ...]
    upper: tuple[float, ...]


@dataclass(frozen=True)
class SafetyConfig:
    joint_limits: JointLimits
    max_velocities: tuple[float, ...]       # rad/s (or m/s for gripper)
    max_torques: tuple[float, ...]          # Nm (or N for gripper)
    torque_spike_thresholds: tuple[float, ...]  # collision detection
    torque_spike_window: int                # moving average window size
    workspace_radius_max: float             # m
    workspace_radius_min: float             # m
    workspace_z_min: float                  # m
    workspace_z_max: float                  # m
    dt: float                               # update period (s)


@dataclass(frozen=True)
class StagePreset:
    velocity_scale: float
    workspace_scale: float
    torque_scale: float
    label: str


def default_config() -> SafetyConfig:
    """Default safety config with 5-degree buffer on joint limits."""
    soft_lower = tuple(hw - buf for hw, buf in zip(_HW_LOWER, _BUFFERS))
    soft_upper = tuple(hw + buf for hw, buf in zip(_HW_UPPER, _BUFFERS))
    # Note: soft limits are INSIDE hw limits (more restrictive)
    # lower gets +buffer (moves inward), upper gets -buffer (moves inward)
    soft_lower = tuple(hw + buf for hw, buf in zip(_HW_LOWER, _BUFFERS))
    soft_upper = tuple(hw - buf for hw, buf in zip(_HW_UPPER, _BUFFERS))

    return SafetyConfig(
        joint_limits=JointLimits(lower=soft_lower, upper=soft_upper),
        max_velocities=(4.8, 4.8, 4.8, 4.8, 0.02),  # rad/s, gripper m/s
        max_torques=(4.1, 4.1, 4.1, 4.1, 1.5),
        torque_spike_thresholds=(1.5, 1.5, 1.2, 1.0, 0.5),
        torque_spike_window=10,  # 100ms at 100Hz
        workspace_radius_max=0.35,  # m (slightly beyond max reach)
        workspace_radius_min=0.05,  # m (too close to base)
        workspace_z_min=0.0,        # m (table surface level)
        workspace_z_max=0.50,       # m
        dt=0.01,                    # 100Hz
    )


_STAGE_PRESETS = {
    0: StagePreset(velocity_scale=0.10, workspace_scale=0.6, torque_scale=0.3, label="Reflexive"),
    1: StagePreset(velocity_scale=0.30, workspace_scale=0.7, torque_scale=0.5, label="Reactive"),
    2: StagePreset(velocity_scale=0.50, workspace_scale=0.85, torque_scale=0.7, label="Adaptive"),
    3: StagePreset(velocity_scale=0.70, workspace_scale=1.0, torque_scale=0.9, label="Contextual"),
}


def stage_preset(stage: int) -> StagePreset:
    """Get safety preset for a developmental stage (0-3)."""
    if stage not in _STAGE_PRESETS:
        raise ValueError(f"Unknown stage {stage}, expected 0-3")
    return _STAGE_PRESETS[stage]


def effective_config(base: SafetyConfig, preset: StagePreset) -> SafetyConfig:
    """Apply stage preset scaling to a base config."""
    return SafetyConfig(
        joint_limits=base.joint_limits,
        max_velocities=tuple(v * preset.velocity_scale for v in base.max_velocities),
        max_torques=tuple(t * preset.torque_scale for t in base.max_torques),
        torque_spike_thresholds=base.torque_spike_thresholds,
        torque_spike_window=base.torque_spike_window,
        workspace_radius_max=base.workspace_radius_max * preset.workspace_scale,
        workspace_radius_min=base.workspace_radius_min,
        workspace_z_min=base.workspace_z_min,
        workspace_z_max=base.workspace_z_max * preset.workspace_scale,
        dt=base.dt,
    )
