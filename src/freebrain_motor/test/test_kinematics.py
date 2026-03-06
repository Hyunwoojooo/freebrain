"""Tests for FK/IK kinematics (pure Python, no ROS)."""

import math
import pytest

from freebrain_motor.kinematics import (
    KinematicsConfig,
    OM_X_CONFIG,
    fk_position,
    numeric_jacobian,
    cartesian_to_joint_delta,
    ik_solve,
)


class TestFK:
    def test_home_pose(self):
        """Home pose (all zeros) should give known EE position."""
        x, y, z = fk_position([0.0, 0.0, 0.0, 0.0])
        assert x == pytest.approx(0.286, abs=0.001)
        assert y == pytest.approx(0.0, abs=0.001)
        assert z == pytest.approx(0.2045, abs=0.001)

    def test_joint1_rotation_changes_xy(self):
        """Rotating joint1 (yaw) should change x,y, preserve z and approximate reach."""
        x0, y0, z0 = fk_position([0.0, 0.0, 0.0, 0.0])
        x1, y1, z1 = fk_position([math.pi / 2, 0.0, 0.0, 0.0])
        r0 = math.sqrt(x0 ** 2 + y0 ** 2)
        r1 = math.sqrt(x1 ** 2 + y1 ** 2)
        # Reach approximately preserved (small deviation from non-coaxial joint origins)
        assert r0 == pytest.approx(r1, abs=0.02)
        assert z0 == pytest.approx(z1, abs=0.001)
        # At 90deg, x should be near 0, y positive
        assert abs(x1) < 0.02
        assert y1 > 0.2

    def test_joint2_changes_reach(self):
        """Rotating joint2 (pitch) should change EE reach and height."""
        x0, _, z0 = fk_position([0.0, 0.0, 0.0, 0.0])
        x1, _, z1 = fk_position([0.0, 0.5, 0.0, 0.0])
        # Different position
        assert (x0 != pytest.approx(x1, abs=0.01)) or (z0 != pytest.approx(z1, abs=0.01))

    def test_short_input_returns_zero(self):
        x, y, z = fk_position([0.0, 0.0])
        assert x == 0.0 and y == 0.0 and z == 0.0

    def test_symmetry_joint1(self):
        """+angle and -angle on joint1 should be symmetric in y."""
        _, y_pos, _ = fk_position([0.3, 0.0, 0.0, 0.0])
        _, y_neg, _ = fk_position([-0.3, 0.0, 0.0, 0.0])
        assert y_pos == pytest.approx(-y_neg, abs=0.001)


class TestJacobian:
    def test_shape(self):
        J = numeric_jacobian([0.0, 0.0, 0.0, 0.0])
        assert len(J) == 3
        assert len(J[0]) == 4

    def test_nonzero_at_home(self):
        J = numeric_jacobian([0.0, 0.0, 0.0, 0.0])
        # Joint1 (yaw) should affect y at home pose
        assert abs(J[1][0]) > 0.1
        # Joint2 (pitch) should affect x and z
        assert abs(J[0][1]) > 0.01 or abs(J[2][1]) > 0.01


class TestCartesianDelta:
    def test_zero_delta_gives_zero_dq(self):
        dq = cartesian_to_joint_delta((0.0, 0.0, 0.0), [0.0, 0.0, 0.0, 0.0])
        for v in dq:
            assert v == pytest.approx(0.0, abs=1e-8)

    def test_small_x_delta(self):
        dq = cartesian_to_joint_delta((0.01, 0.0, 0.0), [0.0, 0.0, 0.0, 0.0])
        # Should produce some joint motion
        assert any(abs(v) > 1e-6 for v in dq)

    def test_roundtrip(self):
        """Small delta_xyz → dq → FK should approximate original delta."""
        base = [0.0, 0.0, 0.0, 0.0]
        delta = (0.005, 0.003, -0.002)
        dq = cartesian_to_joint_delta(delta, base)
        ee_before = fk_position(base)
        new_angles = [base[i] + dq[i] for i in range(4)]
        ee_after = fk_position(new_angles)
        actual_delta = (ee_after[0] - ee_before[0], ee_after[1] - ee_before[1], ee_after[2] - ee_before[2])
        for i in range(3):
            assert actual_delta[i] == pytest.approx(delta[i], abs=0.002)


class TestIK:
    def test_home_pose_ik(self):
        """IK to home pose EE should return near-zero angles."""
        target = (0.286, 0.0, 0.2045)
        converged, angles = ik_solve(target, [0.0, 0.0, 0.0, 0.0])
        assert converged
        for a in angles:
            assert abs(a) < 0.01

    def test_reachable_target(self):
        """IK to a reachable target (derived from FK)."""
        seed = [0.3, -0.2, 0.4, -0.1]
        target = fk_position(seed)
        converged, angles = ik_solve(target, [0.0, 0.0, 0.0, 0.0])
        assert converged
        result = fk_position(angles)
        for i in range(3):
            assert result[i] == pytest.approx(target[i], abs=0.002)

    def test_unreachable_target(self):
        """IK to unreachable target should not converge."""
        target = (1.0, 0.0, 0.0)  # way beyond reach
        converged, _ = ik_solve(target, [0.0, 0.0, 0.0, 0.0], max_iter=50)
        assert not converged

    def test_various_poses(self):
        """Test IK roundtrip for several poses."""
        test_angles = [
            [0.5, 0.3, -0.2, 0.1],
            [-0.4, 0.0, 0.5, -0.3],
            [1.0, -0.5, 0.3, 0.2],
        ]
        for seed in test_angles:
            target = fk_position(seed)
            converged, result_angles = ik_solve(target, [0.0, 0.0, 0.0, 0.0])
            assert converged, f"Failed to converge for seed={seed}"
            result_pos = fk_position(result_angles)
            for i in range(3):
                assert result_pos[i] == pytest.approx(target[i], abs=0.003)
