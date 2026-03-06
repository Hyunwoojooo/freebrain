"""Fuzz test: random 1000 states, zero false-negatives."""

import math
import random
import pytest

from freebrain_safety.config import default_config, N_JOINTS
from freebrain_safety.limits import check_all


@pytest.fixture
def config():
    return default_config()


def test_fuzz_no_false_negatives(config):
    """Generate 1000 random states with known violations — check_all must catch them all."""
    random.seed(42)
    false_negatives = 0

    for _ in range(1000):
        # Generate positions that may violate limits
        positions = [random.uniform(-4.0, 4.0) for _ in range(N_JOINTS)]
        velocities = [random.uniform(-6.0, 6.0) for _ in range(N_JOINTS)]
        torques = [random.uniform(-5.0, 5.0) for _ in range(N_JOINTS)]
        ee_x = random.uniform(-0.5, 0.5)
        ee_y = random.uniform(-0.5, 0.5)
        ee_z = random.uniform(-0.2, 0.7)
        ee_pos = (ee_x, ee_y, ee_z)

        result = check_all(positions, velocities, torques, ee_pos, config)

        # Independently verify: check if there SHOULD be a violation
        has_pos_violation = any(
            positions[i] < config.joint_limits.lower[i] or positions[i] > config.joint_limits.upper[i]
            for i in range(N_JOINTS)
        )
        has_vel_violation = any(
            abs(velocities[i]) > config.max_velocities[i]
            for i in range(N_JOINTS)
        )
        has_torque_violation = any(
            abs(torques[i]) > config.max_torques[i]
            for i in range(N_JOINTS)
        )
        r = math.sqrt(ee_x ** 2 + ee_y ** 2)
        has_ws_violation = (
            r > config.workspace_radius_max
            or r < config.workspace_radius_min
            or ee_z < config.workspace_z_min
            or ee_z > config.workspace_z_max
        )

        expected_violation = has_pos_violation or has_vel_violation or has_torque_violation or has_ws_violation

        if expected_violation and result.all_ok:
            false_negatives += 1

        # Also check per-category consistency
        if has_pos_violation:
            assert not result.joint_limits_ok, f"Missed position violation: {positions}"
        if has_vel_violation:
            assert not result.velocity_ok, f"Missed velocity violation: {velocities}"
        if has_torque_violation:
            assert not result.current_ok, f"Missed torque violation: {torques}"
        if has_ws_violation:
            assert not result.workspace_ok, f"Missed workspace violation: ee={ee_pos}, r={r}"

    assert false_negatives == 0, f"{false_negatives} false negatives out of 1000"
