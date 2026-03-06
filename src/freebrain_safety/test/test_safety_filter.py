"""Tests for SafetyFilter."""

import pytest

from freebrain_safety.config import default_config, stage_preset, effective_config, N_JOINTS
from freebrain_safety.safety_filter import SafetyFilter


@pytest.fixture
def filt():
    return SafetyFilter()


class TestClamp:
    def test_within_limits_unchanged(self, filt):
        target = [0.0] * N_JOINTS
        current = [0.0] * N_JOINTS
        safe, result = filt.filter_position_command(
            target, current, [0.0] * N_JOINTS, [0.0] * N_JOINTS, (0.286, 0, 0.2)
        )
        assert safe == target

    def test_exceeds_joint_limit_clamped(self, filt):
        cfg = filt.config
        target = [0.0] * N_JOINTS
        target[0] = 99.0  # way over
        current = [0.0] * N_JOINTS
        safe, result = filt.filter_position_command(
            target, current, [0.0] * N_JOINTS, [0.0] * N_JOINTS, (0.286, 0, 0.2)
        )
        # Should be clamped, and further limited by velocity
        assert safe[0] <= cfg.joint_limits.upper[0]

    def test_negative_limit_clamped(self, filt):
        target = [0.0] * N_JOINTS
        target[1] = -99.0
        current = [0.0] * N_JOINTS
        safe, _ = filt.filter_position_command(
            target, current, [0.0] * N_JOINTS, [0.0] * N_JOINTS, (0.286, 0, 0.2)
        )
        assert safe[1] >= filt.config.joint_limits.lower[1]


class TestVelocityLimit:
    def test_large_step_reduced(self, filt):
        cfg = filt.config
        # Try to jump 1.0 rad in one step (way over velocity limit)
        target = [1.0] * N_JOINTS
        current = [0.0] * N_JOINTS
        safe, _ = filt.filter_position_command(
            target, current, [0.0] * N_JOINTS, [0.0] * N_JOINTS, (0.286, 0, 0.2)
        )
        max_step = cfg.max_velocities[0] * cfg.dt
        assert abs(safe[0] - current[0]) <= max_step + 1e-9

    def test_small_step_unchanged(self, filt):
        cfg = filt.config
        max_step = cfg.max_velocities[0] * cfg.dt
        tiny = max_step * 0.5
        target = [tiny] + [0.0] * (N_JOINTS - 1)
        current = [0.0] * N_JOINTS
        safe, _ = filt.filter_position_command(
            target, current, [0.0] * N_JOINTS, [0.0] * N_JOINTS, (0.286, 0, 0.2)
        )
        assert safe[0] == pytest.approx(tiny, abs=1e-9)


class TestCollisionHold:
    def test_collision_holds_position(self):
        filt = SafetyFilter()
        current = [0.0] * N_JOINTS
        target = [0.001] * N_JOINTS
        normal_torque = [0.5] * N_JOINTS

        # Fill collision detector buffer with normal torques
        for _ in range(10):
            filt.filter_position_command(
                target, current, [0.0] * N_JOINTS, normal_torque, (0.286, 0, 0.2)
            )

        # Inject a spike
        spike_torque = [0.5, 0.5, 0.5, 0.5, 0.5]
        spike_torque[0] = 5.0  # big spike
        safe, result = filt.filter_position_command(
            target, current, [0.0] * N_JOINTS, spike_torque, (0.286, 0, 0.2)
        )
        # Should hold current position
        assert safe == current
        assert not result.current_ok


class TestStageChange:
    def test_stage_changes_velocity(self):
        filt = SafetyFilter()
        base = default_config()

        filt.set_stage(stage_preset(0))
        cfg0 = filt.config
        filt.set_stage(stage_preset(2))
        cfg2 = filt.config

        # Stage 2 should allow higher velocity than stage 0
        assert cfg2.max_velocities[0] > cfg0.max_velocities[0]

    def test_stage_changes_workspace(self):
        filt = SafetyFilter()
        filt.set_stage(stage_preset(0))
        r0 = filt.config.workspace_radius_max
        filt.set_stage(stage_preset(3))
        r3 = filt.config.workspace_radius_max
        assert r3 > r0


class TestCheckOnly:
    def test_check_only_safe(self, filt):
        # Stage 0 workspace_radius_max = 0.35 * 0.6 = 0.21, use EE within range
        result = filt.check_only(
            [0.0] * N_JOINTS, [0.0] * N_JOINTS, [0.0] * N_JOINTS, (0.15, 0, 0.2)
        )
        assert result.all_ok
