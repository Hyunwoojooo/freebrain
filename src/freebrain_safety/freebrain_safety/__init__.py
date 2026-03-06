"""FreeBrain safety system — joint limits, collision detection, safety filtering."""

from .config import (
    SafetyConfig,
    StagePreset,
    JointLimits,
    default_config,
    stage_preset,
    effective_config,
    JOINT_NAMES,
    N_JOINTS,
)
from .limits import SafetyCheckResult, check_all
from .collision_detector import CollisionDetector
from .safety_filter import SafetyFilter

__all__ = [
    "SafetyConfig",
    "StagePreset",
    "JointLimits",
    "SafetyCheckResult",
    "CollisionDetector",
    "SafetyFilter",
    "default_config",
    "stage_preset",
    "effective_config",
    "check_all",
    "JOINT_NAMES",
    "N_JOINTS",
]
