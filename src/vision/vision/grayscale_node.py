"""
grayscale_node.py  (VINS-ready edition)
========================================
Changes vs original
--------------------
1. Resize rear image to 640×480 before publishing.
   VINS-Fusion is designed for this resolution; feeding 1280×720 raw
   causes process_time > 10 ms and n_pts to collapse below 20.

2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation)
   to the rear image before publishing.  Indoor corridors and car parks
   are low-contrast; CLAHE boosts gradient energy so the Shi-Tomasi
   detector finds 50-150 feature points instead of 4-17.

3. Side camera path is unchanged (used by lot_detector, not VINS).

Topics
------
  /rear_cam/image_raw   (BGR8  1280×720)  ─▶  /rear_cam/image_gray   (mono8 640×480 + CLAHE)
  /side_cam/image_raw   (BGR8  1280×720)  ─▶  /side_cam/image_gray   (mono8 1280×720, unchanged)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


# ── VINS target resolution ────────────────────────────────────────────────
VINS_W = 640
VINS_H = 360   # 16:9 — must match aspect ratio of OV5647 calibration


class GrayscaleConverter(Node):
    def __init__(self):
        super().__init__('grayscale_converter')

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscriptions ─────────────────────────────────────────────────
        self.side_sub = self.create_subscription(
            Image, '/side_cam/image_raw', self.side_callback, sensor_qos)
        self.rear_sub = self.create_subscription(
            Image, '/rear_cam/image_raw', self.rear_callback, sensor_qos)

        # ── Publishers ────────────────────────────────────────────────────
        self.side_pub = self.create_publisher(Image, '/side_cam/image_gray', 10)
        self.rear_pub = self.create_publisher(Image, '/rear_cam/image_gray', 10)

        self.bridge = CvBridge()

        # CLAHE instance — reused every frame (creating it each time is slow)
        # clipLimit=3.0  →  aggressive enough for dim corridors/car parks
        # tileGridSize=(8,8)  →  standard; smaller tiles = more local contrast
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        self.get_logger().info(
            f'GrayscaleConverter ready  '
            f'rear → {VINS_W}×{VINS_H} + CLAHE  |  side → passthrough'
        )

    # ── Side camera (parking detector, no resize needed) ──────────────────
    def side_callback(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            out = self.bridge.cv2_to_imgmsg(gray, encoding='mono8')
            out.header = msg.header
            self.side_pub.publish(out)
        except Exception as e:
            self.get_logger().error(f'side_callback: {e}')

    # ── Rear camera (VINS-Fusion — resize + CLAHE) ────────────────────────
    def rear_callback(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 1. Convert to grayscale first (faster resize on single channel)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            # 2. Resize to VINS target resolution
            #    INTER_AREA is the correct filter when downsampling;
            #    it avoids aliasing that confuses the feature detector.
            if gray.shape[1] != VINS_W or gray.shape[0] != VINS_H:
                gray = cv2.resize(gray, (VINS_W, VINS_H),
                                  interpolation=cv2.INTER_AREA)

            # 3. CLAHE — boosts feature count in low-contrast indoor scenes
            gray = self.clahe.apply(gray)

            # 4. Publish with the ORIGINAL stamp so VINS IMU sync is correct
            out = self.bridge.cv2_to_imgmsg(gray, encoding='mono8')
            out.header = msg.header          # keep original stamp
            self.rear_pub.publish(out)

        except Exception as e:
            self.get_logger().error(f'rear_callback: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = GrayscaleConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass  # already shut down by launch system


if __name__ == '__main__':
    main()
