"""Tests for JAX safety functions."""

import pytest

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from freebrain_safety.config import default_config, stage_preset

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")


@pytest.fixture
def params():
    from freebrain_safety.jax_safety import make_jax_params
    return make_jax_params(default_config(), stage_preset(1))


class TestJaxClip:
    def test_shape_preserved(self, params):
        from freebrain_safety.jax_safety import jax_clip_positions
        batch = 32
        target = jnp.zeros((batch, 5))
        current = jnp.zeros((batch, 5))
        result = jax_clip_positions(target, current, params, 0.01)
        assert result.shape == (batch, 5)

    def test_within_limits_unchanged(self, params):
        from freebrain_safety.jax_safety import jax_clip_positions
        target = jnp.zeros((5,))
        current = jnp.zeros((5,))
        result = jax_clip_positions(target, current, params, 0.01)
        assert jnp.allclose(result, target)

    def test_exceeds_limit_clipped(self, params):
        from freebrain_safety.jax_safety import jax_clip_positions
        target = jnp.ones((5,)) * 99.0
        current = jnp.zeros((5,))
        result = jax_clip_positions(target, current, params, 0.01)
        # Must be within joint limits
        assert jnp.all(result <= params.joint_upper)

    def test_jit_compiles(self, params):
        from freebrain_safety.jax_safety import jax_clip_positions
        target = jnp.zeros((5,))
        current = jnp.zeros((5,))
        # Call twice to verify JIT works
        r1 = jax_clip_positions(target, current, params, 0.01)
        r2 = jax_clip_positions(target, current, params, 0.01)
        assert jnp.allclose(r1, r2)


class TestJaxCheck:
    def test_safe_state(self, params):
        from freebrain_safety.jax_safety import jax_check_safety
        pos = jnp.zeros((5,))
        vel = jnp.zeros((5,))
        # Stage 1: workspace_radius_max = 0.35 * 0.7 = 0.245, use EE within range
        ee = jnp.array([0.15, 0.0, 0.2])
        assert bool(jax_check_safety(pos, vel, ee, params))

    def test_batch_safe(self, params):
        from freebrain_safety.jax_safety import jax_check_safety
        batch = 64
        pos = jnp.zeros((batch, 5))
        vel = jnp.zeros((batch, 5))
        ee = jnp.tile(jnp.array([0.15, 0.0, 0.2]), (batch, 1))
        result = jax_check_safety(pos, vel, ee, params)
        assert result.shape == (batch,)
        assert jnp.all(result)

    def test_unsafe_position(self, params):
        from freebrain_safety.jax_safety import jax_check_safety
        pos = jnp.ones((5,)) * 99.0
        vel = jnp.zeros((5,))
        ee = jnp.array([0.286, 0.0, 0.2])
        assert not bool(jax_check_safety(pos, vel, ee, params))


class TestJaxCost:
    def test_center_zero_cost(self, params):
        from freebrain_safety.jax_safety import jax_safety_cost
        pos = jnp.zeros((5,))
        vel = jnp.zeros((5,))
        ee = jnp.array([0.2, 0.0, 0.2])
        cost = jax_safety_cost(pos, vel, ee, params)
        assert float(cost) == pytest.approx(0.0, abs=1e-6)

    def test_near_limit_positive_cost(self, params):
        from freebrain_safety.jax_safety import jax_safety_cost
        # Place joint near its upper limit
        pos = jnp.array(list(params.joint_upper))
        vel = jnp.zeros((5,))
        ee = jnp.array([0.2, 0.0, 0.2])
        cost = jax_safety_cost(pos, vel, ee, params)
        assert float(cost) > 0.0

    def test_batch_shape(self, params):
        from freebrain_safety.jax_safety import jax_safety_cost
        batch = 16
        pos = jnp.zeros((batch, 5))
        vel = jnp.zeros((batch, 5))
        ee = jnp.tile(jnp.array([0.2, 0.0, 0.2]), (batch, 1))
        cost = jax_safety_cost(pos, vel, ee, params)
        assert cost.shape == (batch,)
