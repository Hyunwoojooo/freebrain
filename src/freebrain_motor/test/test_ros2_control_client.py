"""Tests for Ros2ControlClient in dry_run mode (no actual ROS required)."""

import pytest
import rclpy
from rclpy.node import Node

from freebrain_motor.ros2_control_client import Ros2ControlClient, TrajectoryCommand, ARM_JOINT_NAMES


@pytest.fixture(scope="module")
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def dry_client(ros_context):
    node = Node("test_motor_client")
    client = Ros2ControlClient(node=node, dry_run=True)
    yield client
    node.destroy_node()


class TestDryRun:
    def test_send_trajectory(self, dry_client):
        cmd = TrajectoryCommand(
            joint_names=ARM_JOINT_NAMES,
            positions=[0.0, 0.0, 0.0, 0.0],
            path_time=1.0,
        )
        dry_client.send_trajectory(cmd)  # should not raise

    def test_send_arm_positions(self, dry_client):
        dry_client.send_arm_positions([0.1, 0.2, 0.3, 0.4], path_time=2.0)

    def test_send_gripper(self, dry_client):
        dry_client.send_gripper(0.01, max_effort=1.0)

    def test_halt_all(self, dry_client):
        dry_client.halt_all()

    def test_set_torque(self, dry_client):
        dry_client.set_torque(True)
        dry_client.set_torque(False)

    def test_wait_for_interfaces(self, dry_client):
        dry_client.wait_for_interfaces(0.1)  # dry_run returns immediately
