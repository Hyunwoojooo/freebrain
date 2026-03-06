"""ROS 2 safety monitoring node."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from freebrain_msgs.msg import SafetyStatus, DevelopmentalState

from .config import default_config, stage_preset, effective_config
from .safety_filter import SafetyFilter


class SafetyNode(Node):
    """Monitors joint states and publishes safety status at 100Hz."""

    def __init__(self) -> None:
        super().__init__("safety_node")

        self._filter = SafetyFilter()
        self._ee_pos = (0.286, 0.0, 0.2045)  # default home pose EE

        # Subscribers
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
        self.create_subscription(
            DevelopmentalState, "/developmental_state", self._stage_cb, 10
        )

        # Publisher
        self._pub = self.create_publisher(SafetyStatus, "/safety_status", 10)

        # Timer for periodic check (100Hz)
        self._latest_pos = [0.0] * 5
        self._latest_vel = [0.0] * 5
        self._latest_eff = [0.0] * 5
        self.create_timer(0.01, self._timer_cb)

        self.get_logger().info("Safety node started (Stage 0: Reflexive)")

    def _joint_cb(self, msg: JointState) -> None:
        n = min(len(msg.position), 5)
        self._latest_pos[:n] = list(msg.position[:n])
        if msg.velocity:
            nv = min(len(msg.velocity), 5)
            self._latest_vel[:nv] = list(msg.velocity[:nv])
        if msg.effort:
            ne = min(len(msg.effort), 5)
            self._latest_eff[:ne] = list(msg.effort[:ne])

    def _stage_cb(self, msg: DevelopmentalState) -> None:
        try:
            preset = stage_preset(msg.current_stage)
            self._filter.set_stage(preset)
            self.get_logger().info(f"Stage changed to {msg.current_stage}: {preset.label}")
        except ValueError:
            self.get_logger().warn(f"Unknown stage: {msg.current_stage}")

    def _timer_cb(self) -> None:
        result = self._filter.check_only(
            self._latest_pos, self._latest_vel, self._latest_eff, self._ee_pos
        )
        msg = SafetyStatus()
        msg.all_ok = result.all_ok
        msg.joint_limits_ok = result.joint_limits_ok
        msg.velocity_ok = result.velocity_ok
        msg.current_ok = result.current_ok
        msg.workspace_ok = result.workspace_ok
        self._pub.publish(msg)

        if not result.all_ok:
            self.get_logger().warn(
                f"Safety violation: {'; '.join(result.violations[:3])}", throttle_duration_sec=1.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = SafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
