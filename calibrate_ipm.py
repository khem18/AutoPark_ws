"""
calibrate_ipm.py — headless IPM calibration for AutoPark (no GUI needed)
=========================================================================
Works over SSH / without a local display.

How it works:
  1. Grabs one frame, applies resize+flip (same as lot_detector)
  2. Draws a numbered grid on the frame
  3. Publishes it on /calibration/frame  (view in rqt_image_view)
  4. You hover over the 4 yellow-line corners in rqt and type the coords here

Run:
    cd ~/AutoPark_ws && source install/setup.bash
    python3 calibrate_ipm.py

Then in rqt_image_view:
  - Select topic:  /calibration/frame
  - Tick the mouse checkbox (shows x,y as you hover)
  - Hover over each corner and type the coordinates when prompted
"""

import sys
import threading
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


LABELS = [
    'Bottom-Left  (lower end of LEFT  yellow line)',
    'Bottom-Right (lower end of RIGHT yellow line)',
    'Top-Left     (upper end of LEFT  yellow line, far away)',
    'Top-Right    (upper end of RIGHT yellow line, far away)',
]
COLORS_BGR = [
    (0, 255,   0),   # green
    (0,   0, 255),   # red
    (255, 200, 0),   # cyan-ish
    (255,   0, 255), # magenta
]


class CalibNode(Node):
    def __init__(self):
        super().__init__('ipm_calibrator')
        self.bridge = CvBridge()
        self.frame = None          # raw captured frame
        self.annotated = None      # frame with grid + clicked points

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.sub = self.create_subscription(
            Image, '/side_cam/image_raw', self._cb, qos)

        self.pub = self.create_publisher(Image, '/calibration/frame', 1)
        self.timer = self.create_timer(0.2, self._publish)   # 5 Hz

        self.get_logger().info('Waiting for /side_cam/image_raw …')

    # ── capture one frame ────────────────────────────────────────────
    def _cb(self, msg):
        if self.frame is not None:
            return
        try:
            enc = msg.encoding.lower()
            if enc in ('nv12', 'yuv420sp'):
                h, w = msg.height, msg.width
                raw = np.frombuffer(msg.data, dtype=np.uint8)
                yuv = raw[: h * w * 3 // 2].reshape((h * 3 // 2, w))
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
            else:
                bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # same transforms as lot_detector
            bgr = cv2.resize(bgr, (640, 480))
            bgr = cv2.flip(bgr, 1)
            self.frame = bgr
            self.annotated = self._draw_grid(bgr)
            self.get_logger().info('Frame captured — view /calibration/frame in rqt')
        except Exception as e:
            self.get_logger().error(f'{e}')

    # ── draw coordinate grid ─────────────────────────────────────────
    def _draw_grid(self, img):
        out = img.copy()
        # vertical lines every 80 px
        for x in range(0, 641, 80):
            cv2.line(out, (x, 0), (x, 480), (60, 60, 60), 1)
            cv2.putText(out, str(x), (x+2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        # horizontal lines every 60 px
        for y in range(0, 481, 60):
            cv2.line(out, (0, y), (640, y), (60, 60, 60), 1)
            cv2.putText(out, str(y), (2, y+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        return out

    def update_points(self, points):
        """Redraw with all clicked points so far."""
        if self.frame is None:
            return
        out = self._draw_grid(self.frame)
        for i, (px, py) in enumerate(points):
            cv2.circle(out, (int(px), int(py)), 7, COLORS_BGR[i], -1)
            cv2.circle(out, (int(px), int(py)), 9, (255, 255, 255), 2)
            cv2.putText(out, str(i+1), (int(px)+11, int(py)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS_BGR[i], 2)

        # live BEV preview once all 4 points entered
        if len(points) == 4:
            src = np.float32(points)
            dst = np.float32([
                [200.0, 480.0], [440.0, 480.0],
                [200.0,   0.0], [440.0,   0.0],
            ])
            M   = cv2.getPerspectiveTransform(src, dst)
            bev = cv2.warpPerspective(self.frame, M, (640, 480))
            bev_small = cv2.resize(bev, (213, 160))
            out[320:480, 427:640] = bev_small
            cv2.rectangle(out, (427, 320), (640, 480), (0, 255, 255), 2)
            cv2.putText(out, 'BEV preview', (430, 338),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        self.annotated = out

    def _publish(self):
        if self.annotated is None:
            return
        msg = self.bridge.cv2_to_imgmsg(self.annotated, encoding='bgr8')
        from builtin_interfaces.msg import Time
        self.pub.publish(msg)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = CalibNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # wait for frame
    print('\nWaiting for camera frame (up to 10 s)…')
    for _ in range(100):
        if node.frame is not None:
            break
        time.sleep(0.1)

    if node.frame is None:
        print('\nERROR: No frame received.')
        print('Make sure the camera launch is running:')
        print('  ros2 launch bringup dual_mipi_cam.launch.py')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    print(f'\nFrame size: 640×480  (resize+flip applied)')
    print('\n' + '='*60)
    print('STEP 1: Open rqt_image_view and select topic:')
    print('           /calibration/frame')
    print('        Tick the mouse-left checkbox to see x,y coords.')
    print()
    print('STEP 2: Hover over each corner and type the x,y below.')
    print('        The image has grid lines every 80px (H) / 60px (V).')
    print('='*60 + '\n')

    points = []
    for i in range(4):
        while True:
            try:
                raw = input(f'Point {i+1} — {LABELS[i]}\n  Enter x,y (e.g. 320,240): ').strip()
                x_str, y_str = raw.split(',')
                x, y = float(x_str.strip()), float(y_str.strip())
                if not (0 <= x <= 640 and 0 <= y <= 480):
                    print('  Out of range (0-640, 0-480). Try again.')
                    continue
                points.append((x, y))
                node.update_points(points)
                print(f'  ✓ Saved ({x:.0f}, {y:.0f})\n')
                break
            except (ValueError, KeyboardInterrupt):
                print('  Invalid — type two numbers separated by comma.')

    # print result
    bl, br, tl, tr = points
    print('\n' + '='*60)
    print('DONE — copy this into lot_detector.py  (~line 145):')
    print('='*60)
    print(f"""
        src_pts = np.float32([
            [{bl[0]:.1f}, {bl[1]:.1f}],   # Bottom-Left
            [{br[0]:.1f}, {br[1]:.1f}],   # Bottom-Right
            [{tl[0]:.1f}, {tl[1]:.1f}],   # Top-Left
            [{tr[0]:.1f}, {tr[1]:.1f}],   # Top-Right
        ])
""")
    print('Then rebuild:')
    print('  colcon build --packages-select autopark_logic && source install/setup.bash')
    print('='*60)

    # keep publishing for a moment so user can verify BEV preview
    print('\nKeeping /calibration/frame alive for 15 s so you can see the BEV preview…')
    time.sleep(15)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
