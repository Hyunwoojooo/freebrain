"""Current-spike based collision detection using torque moving average."""

from collections import deque

from .config import SafetyConfig, JOINT_NAMES


class CollisionDetector:
    """Detects collisions by monitoring torque spikes against a moving average."""

    def __init__(self, config: SafetyConfig) -> None:
        self._config = config
        n = len(config.torque_spike_thresholds)
        self._window = config.torque_spike_window
        self._buffers: list[deque[float]] = [deque(maxlen=self._window) for _ in range(n)]
        self._sums: list[float] = [0.0] * n

    def update(self, torques: tuple[float, ...] | list[float]) -> tuple[bool, list[str]]:
        """Update with new torque reading. Returns (collision_detected, violations)."""
        violations = []
        n = len(self._config.torque_spike_thresholds)

        for i in range(n):
            buf = self._buffers[i]
            # Remove oldest value from running sum if buffer is full
            if len(buf) == self._window:
                self._sums[i] -= buf[0]
            buf.append(torques[i])
            self._sums[i] += torques[i]

            # Need at least half the window to have a meaningful average
            if len(buf) < self._window // 2:
                continue

            mean = self._sums[i] / len(buf)
            deviation = abs(torques[i] - mean)
            if deviation > self._config.torque_spike_thresholds[i]:
                violations.append(
                    f"{JOINT_NAMES[i]} torque spike: |{torques[i]:.3f} - mean {mean:.3f}| = "
                    f"{deviation:.3f} > {self._config.torque_spike_thresholds[i]:.3f}"
                )

        collision = len(violations) > 0
        return (collision, violations)

    def reset(self) -> None:
        """Clear all torque history."""
        n = len(self._config.torque_spike_thresholds)
        self._buffers = [deque(maxlen=self._window) for _ in range(n)]
        self._sums = [0.0] * n
