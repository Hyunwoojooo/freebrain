"""FreeBrain motor control — kinematics, ros2_control client, motor node."""

from .kinematics import (
    KinematicsConfig,
    OM_X_CONFIG,
    fk_position,
    numeric_jacobian,
    cartesian_to_joint_delta,
    ik_solve,
)
from .ros2_control_client import (
    Ros2ControlClient,
    TrajectoryCommand,
    ARM_JOINT_NAMES,
)

__all__ = [
    "KinematicsConfig",
    "OM_X_CONFIG",
    "fk_position",
    "numeric_jacobian",
    "cartesian_to_joint_delta",
    "ik_solve",
    "Ros2ControlClient",
    "TrajectoryCommand",
    "ARM_JOINT_NAMES",
]
