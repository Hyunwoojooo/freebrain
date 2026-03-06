"""Tests for safety limit checks."""

import math
import pytest

from freebrain_safety.config import default_config, stage_preset, effective_config, N_JOINTS
from freebrain_safety.limits import (
    check_joint_limits,
    check_velocity_limits,
    check_torque_limits,
    check_workspace,
    check_all,
)


@pytest.fixture
def config():
    return default_config()


@pytest.fixture
def stage0_config(config):
    return effective_config(config, stage_preset(0))


class TestJointLimits:
    def test_home_pose_ok(self, config):
        pos = [0.0] * N_JOINTS
        ok, violations = check_joint_limits(pos, config)
        assert ok
        assert violations == []

    def test_at_soft_limit_ok(self, config):
        # Exactly at soft upper limit should be ok
        pos = list(config.joint_limits.upper)
        ok, _ = check_joint_limits(pos, config)
        assert ok

    def test_beyond_soft_limit_fails(self, config):
        pos = [0.0] * N_JOINTS
        pos[0] = config.joint_limits.upper[0] + 0.001
        ok, violations = check_joint_limits(pos, config)
        assert not ok
        assert len(violations) == 1
        assert "joint1" in violations[0]

    def test_below_soft_limit_fails(self, config):
        pos = [0.0] * N_JOINTS
        pos[1] = config.joint_limits.lower[1] - 0.001
        ok, violations = check_joint_limits(pos, config)
        assert not ok
        assert "joint2" in violations[0]

    def test_multiple_violations(self, config):
        pos = [99.0] * N_JOINTS
        ok, violations = check_joint_limits(pos, config)
        assert not ok
        assert len(violations) == N_JOINTS


class TestVelocityLimits:
    def test_zero_velocity_ok(self, stage0_config):
        vel = [0.0] * N_JOINTS
        ok, _ = check_velocity_limits(vel, stage0_config)
        assert ok

    def test_at_max_ok(self, stage0_config):
        vel = list(stage0_config.max_velocities)
        ok, _ = check_velocity_limits(vel, stage0_config)
        assert ok

    def test_over_max_fails(self, stage0_config):
        vel = [0.0] * N_JOINTS
        vel[0] = stage0_config.max_velocities[0] + 0.001
        ok, violations = check_velocity_limits(vel, stage0_config)
        assert not ok
        assert "joint1" in violations[0]

    def test_negative_over_max_fails(self, stage0_config):
        vel = [0.0] * N_JOINTS
        vel[2] = -(stage0_config.max_velocities[2] + 0.001)
        ok, violations = check_velocity_limits(vel, stage0_config)
        assert not ok


class TestTorqueLimits:
    def test_zero_ok(self, config):
        torques = [0.0] * N_JOINTS
        ok, _ = check_torque_limits(torques, config)
        assert ok

    def test_over_max_fails(self, config):
        torques = [0.0] * N_JOINTS
        torques[0] = config.max_torques[0] + 0.1
        ok, violations = check_torque_limits(torques, config)
        assert not ok


class TestWorkspace:
    def test_home_pose_ee_ok(self, config):
        # Home pose EE at (0.286, 0, 0.2045)
        ok, _ = check_workspace((0.286, 0.0, 0.2045), config)
        assert ok

    def test_too_far_fails(self, config):
        ok, violations = check_workspace((0.5, 0.0, 0.2), config)
        assert not ok
        assert "radius" in violations[0]

    def test_too_close_fails(self, config):
        ok, violations = check_workspace((0.01, 0.0, 0.2), config)
        assert not ok
        assert "radius" in violations[0]

    def test_z_below_table_fails(self, config):
        ok, violations = check_workspace((0.2, 0.0, -0.05), config)
        assert not ok
        assert "z" in violations[0]

    def test_z_too_high_fails(self, config):
        ok, violations = check_workspace((0.2, 0.0, 0.6), config)
        assert not ok


class TestCheckAll:
    def test_all_ok(self, config):
        result = check_all(
            [0.0] * N_JOINTS,
            [0.0] * N_JOINTS,
            [0.0] * N_JOINTS,
            (0.286, 0.0, 0.2045),
            config,
        )
        assert result.all_ok
        assert result.joint_limits_ok
        assert result.velocity_ok
        assert result.current_ok
        assert result.workspace_ok
        assert result.violations == []

    def test_joint_violation_propagates(self, config):
        pos = [0.0] * N_JOINTS
        pos[0] = 99.0
        result = check_all(
            pos,
            [0.0] * N_JOINTS,
            [0.0] * N_JOINTS,
            (0.286, 0.0, 0.2045),
            config,
        )
        assert not result.all_ok
        assert not result.joint_limits_ok
        assert result.velocity_ok


class TestConfig:
    def test_soft_limits_inside_hw(self):
        cfg = default_config()
        # Soft limits should be more restrictive (inside) than HW
        # soft_lower > hw_lower, soft_upper < hw_upper
        for i in range(N_JOINTS):
            assert cfg.joint_limits.lower[i] > -3.15  # sanity
            assert cfg.joint_limits.upper[i] < 3.15

    def test_stage_presets(self):
        for s in range(4):
            p = stage_preset(s)
            assert 0 < p.velocity_scale <= 1.0

    def test_effective_config_scales(self):
        base = default_config()
        preset = stage_preset(0)  # 10% velocity
        eff = effective_config(base, preset)
        assert eff.max_velocities[0] == pytest.approx(base.max_velocities[0] * 0.1)

    def test_invalid_stage_raises(self):
        with pytest.raises(ValueError):
            stage_preset(99)
