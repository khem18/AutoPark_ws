"""
ego_pose_bridge.py
──────────────────
Subscribes to /ego_pose  (Float32MultiArray from lot_detector)
Publishes  to /autopark/start_pose  (Pose2D consumed by autopark_master)

Coordinate mapping
──────────────────
lot_detector ego frame:
    x_cm   = lateral distance  vehicle-centre → lot gate centre  (+right)
    y_cm   = forward distance  vehicle-nose   → lot gate          (+ahead)
    theta  = lot lane tilt angle in camera frame (deg)

autopark_master Pose2D (metres, radians):
    pose.x     = forward distance to gate   = y_cm / 100
    pose.y     = lateral offset to gate     = x_cm / 100
    pose.theta = vehicle heading vs lot     = radians(theta_deg + 180)
                 (+180 because master default_yaw=180 = facing INTO lot)

Adjust the mapping constants below if the planner coordinate frame differs.

Parameters (ROS2 params):
    min_confidence  (float, default 0.4)  – ignore ego_pose below this
    publish_rate_hz (float, default 10.0) – re-publish at fixed rate
"""

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose2D


class EgoPoseBridge(Node):

    def __init__(self):
        super().__init__('ego_pose_bridge')

        self.declare_parameter('min_confidence',  0.4)
        self.declare_parameter('publish_rate_hz', 10.0)

        self.min_conf  = float(self.get_parameter('min_confidence').value)
        rate_hz        = float(self.get_parameter('publish_rate_hz').value)

        # Latest converted pose (None until first confident observation)
        self._pose: Pose2D | None = None

        self.create_subscription(
            Float32MultiArray, '/ego_pose',
            self._on_ego_pose, 10)

        self._pub = self.create_publisher(
            Pose2D, '/autopark/start_pose', 10)

        self.create_timer(1.0 / rate_hz, self._publish)

        self.get_logger().info(
            f'ego_pose_bridge ready  '
            f'(min_conf={self.min_conf}, rate={rate_hz} Hz)')

    # ------------------------------------------------------------------
    def _on_ego_pose(self, msg: Float32MultiArray):
        """
        /ego_pose layout:
          [0] x_cm        lateral offset
          [1] y_cm        forward distance to gate
          [2] theta_deg   lot tilt
          [3] vx          lateral velocity  (cm/s)
          [4] vy          forward velocity  (cm/s)
          [5] omega       angular velocity  (deg/s)
          [6] confidence  1.0 = fresh fix,  0 = expired DR
        """
        if len(msg.data) < 7:
            return

        x_cm, y_cm, theta_deg = msg.data[0], msg.data[1], msg.data[2]
        confidence = msg.data[6]

        if confidence < self.min_conf:
            # ego_pose too stale — keep last good pose (or None)
            return

        pose          = Pose2D()
        pose.x        = float(y_cm)  / 100.0           # forward → planner X  (metres)
        pose.y        = float(x_cm)  / 100.0           # lateral → planner Y  (metres)
        pose.theta    = math.radians(theta_deg + 180.0) # heading into lot

        self._pose = pose

    # ------------------------------------------------------------------
    def _publish(self):
        if self._pose is not None:
            self._pub.publish(self._pose)


# ======================================================================
def main(args=None):
    rclpy.init(args=args)
    node = EgoPoseBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()