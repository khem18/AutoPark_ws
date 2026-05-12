"""
vins_preflight_check.py
========================
Run this BEFORE launching VINS to verify:
  1. Image arrives at 640×480 (not 1280×720)
  2. Feature detector finds ≥ 50 points in the current scene
  3. IMU publishes at ≥ 150 Hz
  4. Image and IMU timestamps are within 50 ms of each other

Usage:
    ros2 run autopark_logic vins_preflight_check
    # or directly:
    python3 vins_preflight_check.py

Press Ctrl+C to stop.  Fix every ❌ before starting VINS.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
import cv2
import numpy as np
import time


MIN_FEATURES = 50
MIN_IMU_HZ   = 150
MAX_SYNC_GAP = 0.050   # seconds


class VinsPreflightCheck(Node):
    def __init__(self):
        super().__init__('vins_preflight_check')
        self.bridge = CvBridge()

        self.last_img_stamp = None
        self.last_imu_stamp = None
        self.imu_stamps     = []
        self.feature_counts = []
        self.img_sizes      = []

        self.create_subscription(Image, '/rear_cam/image_gray',
                                 self.on_image, 10)
        self.create_subscription(Imu, '/imu/data_raw',
                                 self.on_imu, 10)

        self.timer = self.create_timer(3.0, self.report)
        self.get_logger().info('Preflight check running — move the kart gently...')

    # ── Callbacks ─────────────────────────────────────────────────────────
    def on_image(self, msg: Image):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.last_img_stamp = t
        self.img_sizes.append((msg.width, msg.height))
        if len(self.img_sizes) > 30:
            self.img_sizes.pop(0)

        try:
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

            # Count Shi-Tomasi corners (same detector VINS uses)
            pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=300,
                qualityLevel=0.01,
                minDistance=20,
            )
            n = 0 if pts is None else len(pts)
            self.feature_counts.append(n)
            if len(self.feature_counts) > 30:
                self.feature_counts.pop(0)
        except Exception as e:
            self.get_logger().warning(f'image proc error: {e}')

    def on_imu(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.last_imu_stamp = t
        self.imu_stamps.append(t)
        if len(self.imu_stamps) > 400:
            self.imu_stamps.pop(0)

    # ── Report ────────────────────────────────────────────────────────────
    def report(self):
        print('\n' + '═' * 55)
        print('  VINS PREFLIGHT REPORT')
        print('═' * 55)

        # 1. Image size
        if self.img_sizes:
            w, h = self.img_sizes[-1]
            ok = (w == 640 and h == 480)
            mark = '✅' if ok else '❌'
            print(f'{mark}  Image size : {w}×{h}  (need 640×480)')
            if not ok:
                print('     ↳ Fix: check grayscale_node resize is applied')
        else:
            print('❌  Image size : no images received yet')
            print('     ↳ Fix: check /rear_cam/image_gray is publishing')

        # 2. Feature count
        if self.feature_counts:
            avg_n = np.mean(self.feature_counts)
            min_n = np.min(self.feature_counts)
            ok = avg_n >= MIN_FEATURES
            mark = '✅' if ok else '❌'
            print(f'{mark}  Features   : avg={avg_n:.0f}  min={min_n}  (need ≥{MIN_FEATURES})')
            if not ok:
                print('     ↳ Fix: increase clipLimit in CLAHE, or improve lighting')
                print('            lower keyframe_parallax in vins_config.yaml')
        else:
            print('❌  Features   : no images received')

        # 3. IMU rate
        if len(self.imu_stamps) >= 10:
            diffs = np.diff(self.imu_stamps[-50:])
            hz = 1.0 / np.mean(diffs) if np.mean(diffs) > 0 else 0
            ok = hz >= MIN_IMU_HZ
            mark = '✅' if ok else '❌'
            print(f'{mark}  IMU rate   : {hz:.1f} Hz  (need ≥{MIN_IMU_HZ} Hz)')
            if not ok:
                print('     ↳ Fix: mpu6050_node timer is 5 ms (200 Hz)')
                print('            check I2C bus speed; reduce other I2C traffic')
        else:
            print('❌  IMU rate   : waiting for more samples...')

        # 4. Timestamp sync
        if self.last_img_stamp and self.last_imu_stamp:
            gap = abs(self.last_img_stamp - self.last_imu_stamp)
            ok = gap < MAX_SYNC_GAP
            mark = '✅' if ok else '❌'
            print(f'{mark}  Stamp gap  : {gap*1000:.1f} ms  (need < {MAX_SYNC_GAP*1000:.0f} ms)')
            if not ok:
                print('     ↳ Fix: grayscale_node stamps image with get_clock().now()')
                print('            make sure mpu6050_node also uses get_clock().now()')
                print('            enable estimate_td: 1 in vins_config.yaml')
        else:
            print('⚠️   Stamp gap  : waiting for both streams...')

        # 5. Overall verdict
        print('─' * 55)
        img_ok  = bool(self.img_sizes and self.img_sizes[-1] == (640, 480))
        feat_ok = bool(self.feature_counts and np.mean(self.feature_counts) >= MIN_FEATURES)
        imu_ok  = len(self.imu_stamps) >= 10 and (1.0 / np.mean(np.diff(self.imu_stamps[-50:]))) >= MIN_IMU_HZ
        sync_ok = (self.last_img_stamp and self.last_imu_stamp and
                   abs(self.last_img_stamp - self.last_imu_stamp) < MAX_SYNC_GAP)

        if img_ok and feat_ok and imu_ok and sync_ok:
            print('🚀  ALL CHECKS PASSED — safe to launch VINS-Fusion')
        else:
            failed = [name for name, ok in [
                ('image-size', img_ok), ('features', feat_ok),
                ('imu-rate', imu_ok),   ('timestamp-sync', sync_ok)
            ] if not ok]
            print(f'🔴  NOT READY — fix: {", ".join(failed)}')
        print('═' * 55 + '\n')


def main(args=None):
    rclpy.init(args=args)
    node = VinsPreflightCheck()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
