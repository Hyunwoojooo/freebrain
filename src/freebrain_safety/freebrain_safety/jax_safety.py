"""JAX-based safety functions for MJX parallel environments."""

from typing import NamedTuple

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from .config import SafetyConfig, StagePreset, effective_config


if HAS_JAX:
    class JaxSafetyParams(NamedTuple):
        """Safety parameters as a JAX-compatible NamedTuple (pytree leaf)."""
        joint_lower: jnp.ndarray       # (n_joints,)
        joint_upper: jnp.ndarray
        max_velocities: jnp.ndarray
        velocity_scale: jnp.ndarray    # scalar array
        workspace_radius_max: jnp.ndarray
        workspace_radius_min: jnp.ndarray
        workspace_z_min: jnp.ndarray
        workspace_z_max: jnp.ndarray


def make_jax_params(config: SafetyConfig, preset: StagePreset) -> "JaxSafetyParams":
    """Create JAX safety params from config + preset."""
    if not HAS_JAX:
        raise RuntimeError("JAX is not installed")
    eff = effective_config(config, preset)
    return JaxSafetyParams(
        joint_lower=jnp.array(config.joint_limits.lower),
        joint_upper=jnp.array(config.joint_limits.upper),
        max_velocities=jnp.array(config.max_velocities),
        velocity_scale=jnp.float32(preset.velocity_scale),
        workspace_radius_max=jnp.float32(eff.workspace_radius_max),
        workspace_radius_min=jnp.float32(eff.workspace_radius_min),
        workspace_z_min=jnp.float32(eff.workspace_z_min),
        workspace_z_max=jnp.float32(eff.workspace_z_max),
    )


def _jax_clip_positions(target, current, params, dt):
    """Clip target positions to joint limits and velocity limits.

    Args:
        target: (..., n_joints) target positions
        current: (..., n_joints) current positions
        params: JaxSafetyParams
        dt: timestep (static)

    Returns:
        Clipped positions, same shape as target.
    """
    clipped = jnp.clip(target, params.joint_lower, params.joint_upper)
    max_step = params.max_velocities * params.velocity_scale * dt
    delta = clipped - current
    delta = jnp.clip(delta, -max_step, max_step)
    return current + delta


def _jax_check_safety(positions, velocities, ee_positions, params):
    """Check if states are within safety bounds.

    Args:
        positions: (..., n_joints)
        velocities: (..., n_joints)
        ee_positions: (..., 3) end-effector xyz

    Returns:
        Boolean array (...,) — True if safe.
    """
    pos_ok = jnp.all(
        (positions >= params.joint_lower) & (positions <= params.joint_upper),
        axis=-1,
    )
    max_vel = params.max_velocities * params.velocity_scale
    vel_ok = jnp.all(jnp.abs(velocities) <= max_vel, axis=-1)
    r = jnp.sqrt(ee_positions[..., 0] ** 2 + ee_positions[..., 1] ** 2)
    z = ee_positions[..., 2]
    ws_ok = (
        (r <= params.workspace_radius_max)
        & (r >= params.workspace_radius_min)
        & (z >= params.workspace_z_min)
        & (z <= params.workspace_z_max)
    )
    return pos_ok & vel_ok & ws_ok


def _jax_safety_cost(positions, velocities, ee_positions, params):
    """Smooth safety penalty for reward shaping.

    Returns penalty (...,) — 0 when well within limits, >0 near boundaries.
    Uses quadratic penalty outside soft margin (10% from boundary).
    """
    joint_range = params.joint_upper - params.joint_lower
    margin = 0.1 * joint_range
    lower_dist = positions - (params.joint_lower + margin)
    upper_dist = (params.joint_upper - margin) - positions
    pos_cost = jnp.sum(
        jnp.where(lower_dist < 0, lower_dist ** 2, 0.0)
        + jnp.where(upper_dist < 0, upper_dist ** 2, 0.0),
        axis=-1,
    )
    max_vel = params.max_velocities * params.velocity_scale
    vel_margin = 0.1 * max_vel
    vel_excess = jnp.abs(velocities) - (max_vel - vel_margin)
    vel_cost = jnp.sum(jnp.where(vel_excess > 0, vel_excess ** 2, 0.0), axis=-1)
    r = jnp.sqrt(ee_positions[..., 0] ** 2 + ee_positions[..., 1] ** 2)
    z = ee_positions[..., 2]
    r_over = jnp.maximum(r - params.workspace_radius_max, 0.0) ** 2
    r_under = jnp.maximum(params.workspace_radius_min - r, 0.0) ** 2
    z_over = jnp.maximum(z - params.workspace_z_max, 0.0) ** 2
    z_under = jnp.maximum(params.workspace_z_min - z, 0.0) ** 2
    ws_cost = r_over + r_under + z_over + z_under

    return pos_cost + vel_cost + ws_cost


# JIT compile if JAX available
# dt is the only static arg (scalar python float); params is a NamedTuple pytree
if HAS_JAX:
    jax_clip_positions = jax.jit(_jax_clip_positions, static_argnums=(3,))
    jax_check_safety = jax.jit(_jax_check_safety)
    jax_safety_cost = jax.jit(_jax_safety_cost)
else:
    jax_clip_positions = _jax_clip_positions
    jax_check_safety = _jax_check_safety
    jax_safety_cost = _jax_safety_cost
