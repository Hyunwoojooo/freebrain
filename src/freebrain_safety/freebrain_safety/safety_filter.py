"""Safety filter: clips commands to safe ranges and detects collisions."""

from .config import SafetyConfig, StagePreset, effective_config, default_config, stage_preset, N_JOINTS
from .limits import SafetyCheckResult, check_all
from .collision_detector import CollisionDetector


class SafetyFilter:
    """Filters position commands to enforce safety constraints."""

    def __init__(
        self,
        config: SafetyConfig | None = None,
        preset: StagePreset | None = None,
    ) -> None:
        self._base_config = config or default_config()
        self._preset = preset or stage_preset(0)
        self._config = effective_config(self._base_config, self._preset)
        self._collision = CollisionDetector(self._base_config)

    @property
    def config(self) -> SafetyConfig:
        return self._config

    def set_stage(self, preset: StagePreset) -> None:
        self._preset = preset
        self._config = effective_config(self._base_config, self._preset)
        self._collision.reset()

    def filter_position_command(
        self,
        target: list[float] | tuple[float, ...],
        current_pos: list[float] | tuple[float, ...],
        current_vel: list[float] | tuple[float, ...],
        current_torques: list[float] | tuple[float, ...],
        ee_pos: tuple[float, float, float] | list[float],
    ) -> tuple[list[float], SafetyCheckResult]:
        """Filter a position command to be safe.

        Returns (safe_position, check_result).
        """
        cfg = self._config
        dt = cfg.dt
        safe = list(target)

        # 1) Clamp to joint limits
        for i in range(N_JOINTS):
            lo = cfg.joint_limits.lower[i]
            hi = cfg.joint_limits.upper[i]
            safe[i] = max(lo, min(hi, safe[i]))

        # 2) Velocity limiting: restrict step size
        for i in range(N_JOINTS):
            max_step = cfg.max_velocities[i] * dt
            delta = safe[i] - current_pos[i]
            if abs(delta) > max_step:
                safe[i] = current_pos[i] + max_step * (1.0 if delta > 0 else -1.0)

        # 3) Collision detection: hold position if collision detected
        collision, col_violations = self._collision.update(current_torques)
        if collision:
            safe = list(current_pos)

        # 4) Run full safety check on current state for status reporting
        result = check_all(current_pos, current_vel, current_torques, ee_pos, cfg)
        if collision:
            result.current_ok = False
            result.all_ok = False
            result.violations.extend(col_violations)

        return safe, result

    def check_only(
        self,
        positions: list[float] | tuple[float, ...],
        velocities: list[float] | tuple[float, ...],
        torques: list[float] | tuple[float, ...],
        ee_pos: tuple[float, float, float] | list[float],
    ) -> SafetyCheckResult:
        """Check safety without filtering. Also updates collision detector."""
        collision, col_violations = self._collision.update(torques)
        result = check_all(positions, velocities, torques, ee_pos, self._config)
        if collision:
            result.current_ok = False
            result.all_ok = False
            result.violations.extend(col_violations)
        return result
