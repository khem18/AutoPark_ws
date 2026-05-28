"""
vins_pose_bridge.py
====================
Converts real-time VINS-Fusion odometry → /autopark/start_pose (Pose2D).

This replaces the perception_bridge camera-gated pose path.
VINS publishes nav_msgs/Odometry on /vins_estimator/odometry.
This node extracts x, y, yaw and republishes as geometry_msgs/Pose2D.

Add to autopark_system/autopark_system/ then register in setup.py:
    'vins_pose_bridge = autopark_system.vins_pose_bridge:main',

Also add a Node() entry in autopark_full_system.launch.py (see bottom of file).

YAML params (add under autopark_system/config/autopark_params.yaml):
  vins_pose_bridge:
    ros__parameters:
      odom_topic:        "/vins_estimator/odometry"
      start_pose_topic:  "/autopark/start_pose"
      x_offset:          0.0   # shift applied after VINS x  (metres)
      y_offset:          0.0   # shift applied after VINS y  (metres)
      yaw_offset_deg:    0.0   # e.g. 180.0 if VINS world frame is flipped vs planner
      publish_rate_hz:   10.0  # republish rate (capped to VINS rate)
"""

import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D


def quat_to_yaw(qx, qy, qz, qw) -> float:
    """Extract yaw (rotation around Z) from a quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class VinsPoseBridge(Node):
    def __init__(self):
        super().__init__('vins_pose_bridge')

        for name, default in [
            ('odom_topic',       '/vins_estimator/odometry'),
            ('start_pose_topic', '/autopark/start_pose'),
            ('x_offset',         0.0),
            ('y_offset',         0.0),
            ('yaw_offset_deg',   0.0),
            ('publish_rate_hz',  10.0),
        ]:
            self.declare_parameter(name, default)

        self.x_offset      = float(self.get_parameter('x_offset').value)
        self.y_offset      = float(self.get_parameter('y_offset').value)
        self.yaw_offset    = math.radians(float(self.get_parameter('yaw_offset_deg').value))
        self.latest_pose: Pose2D | None = None

        self.pose_pub = self.create_publisher(
            Pose2D,
            self.get_parameter('start_pose_topic').value,
            10,
        )

        self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self._on_odom,
            10,
        )

        rate = float(self.get_parameter('publish_rate_hz').value)
        self.create_timer(1.0 / rate, self._publish)

        self.get_logger().info(
            f'vins_pose_bridge ready\n'
            f'  {self.get_parameter("odom_topic").value} '
            f'-> {self.get_parameter("start_pose_topic").value}\n'
            f'  offsets: x={self.x_offset} y={self.y_offset} '
            f'yaw_offset_deg={math.degrees(self.yaw_offset):.1f}'
        )

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose
        q = p.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        pose = Pose2D()
        pose.x     = p.position.x + self.x_offset
        pose.y     = p.position.y + self.y_offset
        pose.theta = yaw + self.yaw_offset
        self.latest_pose = pose

    def _publish(self):
        if self.latest_pose is None:
            self.get_logger().warn(
                'No VINS odometry received yet — is VINS-Fusion running?',
                throttle_duration_sec=5.0,
            )
            return
        self.pose_pub.publish(self.latest_pose)
        self.get_logger().info(
            f'pose → x={self.latest_pose.x:.3f} '
            f'y={self.latest_pose.y:.3f} '
            f'yaw={math.degrees(self.latest_pose.theta):.1f}°',
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = VinsPoseBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
