"""ROS 2 motor control node — joint state tracking, FK, safety-filtered execution."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from freebrain_msgs.msg import SafetyStatus

from .kinematics import fk_position, OM_X_CONFIG
from .ros2_control_client import Ros2ControlClient, TrajectoryCommand, ARM_JOINT_NAMES


class MotorNode(Node):
    """Motor control node that tracks joint states and publishes EE position."""

    def __init__(self) -> None:
        super().__init__("motor_node")

        # Parameters
        self.declare_parameter("dry_run", False)
        self.declare_parameter("namespace", "")
        self.declare_parameter("arm_controller", "arm_controller")
        self.declare_parameter("gripper_controller", "gripper_action_controller")
        self.declare_parameter("publish_rate", 50.0)

        dry_run = self.get_parameter("dry_run").value
        namespace = self.get_parameter("namespace").value
        arm_ctrl = self.get_parameter("arm_controller").value
        grip_ctrl = self.get_parameter("gripper_controller").value
        rate = self.get_parameter("publish_rate").value

        # Hardware client
        self._client = Ros2ControlClient(
            node=self,
            namespace=namespace,
            arm_controller=arm_ctrl,
            gripper_controller=grip_ctrl,
            dry_run=dry_run,
        )

        # State tracking
        self._joint_positions = [0.0] * 5  # 4 arm + 1 gripper
        self._joint_velocities = [0.0] * 5
        self._joint_efforts = [0.0] * 5
        self._ee_position = (0.286, 0.0, 0.2045)  # home pose FK
        self._safety_ok = True

        # Subscribers
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
        self.create_subscription(SafetyStatus, "/safety_status", self._safety_cb, 10)

        # Publishers
        self._ee_pub = self.create_publisher(PointStamped, "/ee_position", 10)

        # Timer for EE position publishing
        period = 1.0 / rate if rate > 0 else 0.02
        self.create_timer(period, self._publish_ee)

        self.get_logger().info(f"Motor node started (dry_run={dry_run})")

    @property
    def client(self) -> Ros2ControlClient:
        return self._client

    @property
    def joint_positions(self) -> list:
        return list(self._joint_positions)

    @property
    def joint_velocities(self) -> list:
        return list(self._joint_velocities)

    @property
    def ee_position(self) -> tuple:
        return self._ee_position

    @property
    def safety_ok(self) -> bool:
        return self._safety_ok

    def _joint_cb(self, msg: JointState) -> None:
        # Map named joints to our 5-element arrays
        name_map = {
            "joint1": 0, "joint2": 1, "joint3": 2, "joint4": 3,
            "gripper_left_joint": 4, "gripper": 4,
        }
        for i, name in enumerate(msg.name):
            idx = name_map.get(name)
            if idx is None:
                continue
            if i < len(msg.position):
                self._joint_positions[idx] = msg.position[i]
            if msg.velocity and i < len(msg.velocity):
                self._joint_velocities[idx] = msg.velocity[i]
            if msg.effort and i < len(msg.effort):
                self._joint_efforts[idx] = msg.effort[i]

        # Update FK with arm joints (first 4)
        self._ee_position = fk_position(self._joint_positions[:4], OM_X_CONFIG)

    def _safety_cb(self, msg: SafetyStatus) -> None:
        self._safety_ok = msg.all_ok

    def _publish_ee(self) -> None:
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.point.x = self._ee_position[0]
        msg.point.y = self._ee_position[1]
        msg.point.z = self._ee_position[2]
        self._ee_pub.publish(msg)

    def send_arm_positions(self, positions: list, path_time: float = 1.0) -> bool:
        """Send arm positions if safety allows. Returns True if sent."""
        if not self._safety_ok:
            self.get_logger().warn("Safety violation active — command rejected")
            return False
        self._client.send_arm_positions(positions, path_time)
        return True

    def send_gripper(self, position: float, max_effort: float = 1.0) -> bool:
        """Send gripper command if safety allows."""
        if not self._safety_ok:
            self.get_logger().warn("Safety violation active — gripper command rejected")
            return False
        self._client.send_gripper(position, max_effort)
        return True

    def halt(self) -> None:
        """Emergency stop."""
        self._client.halt_all()


def main(args=None):
    rclpy.init(args=args)
    node = MotorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
