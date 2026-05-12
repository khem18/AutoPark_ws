"""
nv12_to_bgr_node.py
====================
Drop-in replacement for hobot_codec_republish on RDK X5.

hobot_codec often fails silently because the BPU pipeline isn't
ready when the node starts, leaving camera nodes completely isolated.
This node does the same NV12 → BGR8 conversion entirely in software
using OpenCV, which is always available.

Usage (parameters):
  sub_topic  – incoming  sensor_msgs/Image   (NV12, encoding='nv12')
  pub_topic  – outgoing  sensor_msgs/Image   (BGR8, encoding='bgr8')

One instance = one camera.  Launch two for side + rear.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class Nv12ToBgr(Node):
    def __init__(self):
        super().__init__('nv12_to_bgr')

        self.declare_parameter('sub_topic', '/cam/image_nv12')
        self.declare_parameter('pub_topic', '/cam/image_raw')

        sub_topic = self.get_parameter('sub_topic').get_parameter_value().string_value
        pub_topic = self.get_parameter('pub_topic').get_parameter_value().string_value

        # BEST_EFFORT + depth=1  →  never queue stale frames
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub = self.create_subscription(
            Image, sub_topic, self.callback, sensor_qos)
        self.pub = self.create_publisher(Image, pub_topic, 10)
        self.bridge = CvBridge()

        self.get_logger().info(
            f'nv12_to_bgr: {sub_topic}  →  {pub_topic}')

    def callback(self, msg: Image):
        try:
            # ── Decode NV12 ──────────────────────────────────────────────
            # NV12 layout: H rows of Y, then H/2 rows of interleaved UV.
            # Total height seen by OpenCV = H * 3/2.
            h = msg.height
            w = msg.width

            raw = np.frombuffer(msg.data, dtype=np.uint8)

            expected = h * w * 3 // 2
            if raw.size < expected:
                self.get_logger().warning(
                    f'Frame too small: got {raw.size}, expected {expected}')
                return

            yuv = raw[:expected].reshape((h * 3 // 2, w))
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

            # ── Re-publish as BGR8 ───────────────────────────────────────
            out = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
            out.header = msg.header   # preserve frame_id + stamp
            self.pub.publish(out)

        except Exception as exc:
            self.get_logger().error(f'nv12_to_bgr error: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = Nv12ToBgr()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
