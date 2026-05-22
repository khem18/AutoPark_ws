import math
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge


class FlowDistanceNode(Node):
    """
    Simple optical-flow + IMU distance estimator.

    Publishes:
      /autopark/flow_distance : Float32MultiArray

    data layout:
      [0] vx_mps_est
      [1] distance_m_est_abs
      [2] yaw_rate_rad_s
      [3] flow_px_per_s
      [4] scale_m_per_px
      [5] valid_flag
    """

    def __init__(self):
        super().__init__("flow_distance_node")

        self.declare_parameter("image_topic", "/rear_cam/image_raw")
        self.declare_parameter("imu_topic", "/imu/data_raw")
        self.declare_parameter("output_topic", "/autopark/flow_distance")

        # Start rough. We will calibrate this with tape measure.
        self.declare_parameter("scale_m_per_px", 0.0005)

        # Ignore tiny image motion noise
        self.declare_parameter("deadband_px_per_s", 1.0)

        # Use center crop to avoid edge fisheye distortion
        self.declare_parameter("crop_ratio", 0.60)

        # Limit too large spikes
        self.declare_parameter("max_v_mps", 1.0)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)

        self.scale_m_per_px = float(self.get_parameter("scale_m_per_px").value)
        self.deadband_px_per_s = float(self.get_parameter("deadband_px_per_s").value)
        self.crop_ratio = float(self.get_parameter("crop_ratio").value)
        self.max_v_mps = float(self.get_parameter("max_v_mps").value)

        self.bridge = CvBridge()

        self.prev_gray = None
        self.prev_t = None

        self.distance_m = 0.0
        self.yaw_rate = 0.0

        self.create_subscription(Image, self.image_topic, self.on_image, 10)
        self.create_subscription(Imu, self.imu_topic, self.on_imu, 50)

        self.pub = self.create_publisher(Float32MultiArray, self.output_topic, 10)

        self.get_logger().info("flow_distance_node ready")
        self.get_logger().info("image_topic = " + self.image_topic)
        self.get_logger().info("imu_topic = " + self.imu_topic)
        self.get_logger().info("scale_m_per_px = " + str(self.scale_m_per_px))

    def on_imu(self, msg: Imu):
        self.yaw_rate = float(msg.angular_velocity.z)

    def crop_center(self, gray):
        h, w = gray.shape[:2]
        r = max(0.1, min(1.0, self.crop_ratio))

        cw = int(w * r)
        ch = int(h * r)

        x0 = (w - cw) // 2
        y0 = (h - ch) // 2

        return gray[y0:y0 + ch, x0:x0 + cw]

    def on_image(self, msg: Image):
        now = time.monotonic()

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            except Exception as exc:
                self.get_logger().warning("image convert failed: " + str(exc))
                return

        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        gray = self.crop_center(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_t = now
            self.publish(0.0, 0.0, 0.0, valid=False)
            return

        dt = now - self.prev_t
        if dt <= 0.001 or dt > 1.0:
            self.prev_gray = gray
            self.prev_t = now
            self.publish(0.0, 0.0, 0.0, valid=False)
            return

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            0.5,
            3,
            21,
            3,
            5,
            1.2,
            0,
        )

        fx = flow[..., 0]
        fy = flow[..., 1]

        # For forward/reverse distance, vertical image flow is often more useful.
        # If your camera direction is different, we can swap to fx later.
        flow_forward_px = float(np.median(fy))
        flow_px_per_s = flow_forward_px / dt

        if abs(flow_px_per_s) < self.deadband_px_per_s:
            vx = 0.0
        else:
            vx = flow_px_per_s * self.scale_m_per_px

        vx = max(-self.max_v_mps, min(self.max_v_mps, vx))

        self.distance_m += abs(vx) * dt

        self.prev_gray = gray
        self.prev_t = now

        self.publish(vx, self.distance_m, flow_px_per_s, valid=True)

    def publish(self, vx, dist, flow_px_per_s, valid):
        msg = Float32MultiArray()
        msg.data = [
            float(vx),
            float(dist),
            float(self.yaw_rate),
            float(flow_px_per_s),
            float(self.scale_m_per_px),
            1.0 if valid else 0.0,
        ]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FlowDistanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
