"""ROS 2 control client for OpenManipulator-X hardware interface."""

from dataclasses import dataclass
from typing import List

from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]


@dataclass(frozen=True)
class TrajectoryCommand:
    joint_names: List[str]
    positions: List[float]
    path_time: float


class Ros2ControlClient:
    """Thin wrapper over ros2_control trajectory + gripper + torque interfaces."""

    def __init__(
        self,
        node: Node,
        namespace: str = "",
        arm_controller: str = "arm_controller",
        gripper_controller: str = "gripper_action_controller",
        torque_service: str = "set_torque",
        dry_run: bool = False,
    ) -> None:
        self._node = node
        self._logger = node.get_logger()
        self._dry_run = dry_run
        self._ns = namespace.strip("/")

        arm_topic = self._join(arm_controller, "joint_trajectory")
        self._traj_pub = node.create_publisher(JointTrajectory, arm_topic, 10)

        gripper_action = self._join(gripper_controller, "gripper_cmd")
        self._gripper_client = ActionClient(node, GripperCommand, gripper_action)

        torque_srv = self._join(torque_service)
        self._torque_client = node.create_client(SetBool, torque_srv)

        self._gripper_pending = False

    def wait_for_interfaces(self, timeout_sec: float = 5.0) -> None:
        if self._dry_run:
            return
        if not self._gripper_client.wait_for_server(timeout_sec=timeout_sec):
            self._logger.warning(f"Gripper action unavailable after {timeout_sec:.1f}s")
        if not self._torque_client.wait_for_service(timeout_sec=timeout_sec):
            self._logger.warning(f"Torque service unavailable after {timeout_sec:.1f}s")

    def send_trajectory(self, cmd: TrajectoryCommand) -> None:
        if self._dry_run:
            self._logger.info(
                f"[dry_run] trajectory: joints={cmd.joint_names} "
                f"positions={[f'{p:.4f}' for p in cmd.positions]} "
                f"time={cmd.path_time:.2f}s"
            )
            return
        traj = JointTrajectory()
        traj.joint_names = list(cmd.joint_names)
        point = JointTrajectoryPoint()
        point.positions = list(cmd.positions)
        point.time_from_start = Duration(seconds=cmd.path_time).to_msg()
        traj.points = [point]
        self._traj_pub.publish(traj)

    def send_arm_positions(self, positions: List[float], path_time: float = 1.0) -> None:
        """Convenience: send 4-joint arm trajectory."""
        self.send_trajectory(TrajectoryCommand(
            joint_names=ARM_JOINT_NAMES,
            positions=positions[:4],
            path_time=path_time,
        ))

    def send_gripper(self, position: float, max_effort: float = 1.0) -> None:
        if self._gripper_pending:
            return
        if self._dry_run:
            self._logger.info(f"[dry_run] gripper: position={position:.4f} max_effort={max_effort:.1f}")
            return
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = max_effort
        self._gripper_pending = True
        future = self._gripper_client.send_goal_async(goal)
        future.add_done_callback(self._gripper_goal_done)

    def halt_all(self) -> None:
        if self._dry_run:
            self._logger.info("[dry_run] halt_all")
            return
        traj = JointTrajectory()
        traj.points = []
        self._traj_pub.publish(traj)

    def set_torque(self, enabled: bool) -> None:
        if self._dry_run:
            self._logger.info(f"[dry_run] torque: enabled={enabled}")
            return
        if not self._torque_client.service_is_ready():
            self._logger.warning("Torque service not ready")
            return
        request = SetBool.Request()
        request.data = enabled
        future = self._torque_client.call_async(request)
        future.add_done_callback(self._torque_done_callback)

    def _torque_done_callback(self, future) -> None:
        try:
            result = future.result()
            if not result.success:
                self._logger.error(f"Torque service failed: {result.message}")
        except Exception as exc:
            self._logger.error(f"Torque service call error: {exc}")

    def _gripper_goal_done(self, future) -> None:
        self._gripper_pending = False
        try:
            goal_handle = future.result()
        except Exception:
            return
        if not goal_handle.accepted:
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda _: None)

    def _join(self, *parts: str) -> str:
        cleaned = [p.strip("/") for p in parts if p]
        if not self._ns:
            return "/" + "/".join(cleaned)
        return "/" + "/".join([self._ns] + cleaned)
