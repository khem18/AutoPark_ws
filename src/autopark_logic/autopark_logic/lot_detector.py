"""
lot_detector.py  — Yellow-line detection for AutoPark
=======================================================

BUG FIXES vs original
----------------------
1. HSV yellow range widened
   Original:  lower=[18, 40, 30]  upper=[45, 255, 255]
   Fixed:     lower=[13, 15, 15]  upper=[52, 255, 255]
   Reason: Faded/dirty paint on asphalt under dim parking-lot lighting
   has LOW saturation (as low as 15-20) and LOW value.  The old minimums
   of S=40 and V=30 silently rejected the line.

2. Median blur kernel reduced from 21 → 7
   Reason: A 21-pixel median completely erases yellow lines that are
   only 10-30 px wide after the BEV transform.  7 removes salt-and-
   pepper noise while preserving thin lines.

3. AWB de-coupled from detection
   Reason: Gray-world AWB shifts yellow toward neutral gray, lowering
   saturation in HSV and making detection fail.  AWB is now applied
   only to the display frame — the HSV mask always uses the raw frame.

4. Debug image publishers (ROS topics)
   /lot_detector/debug_bev      — bird's-eye view with detected boxes
   /lot_detector/debug_mask     — yellow HSV mask (white=detected)
   Enable with ROS param: show_debug_images:=true
   View with: ros2 run rqt_image_view rqt_image_view

5. IPM src_pts calibration helper (see CALIBRATION GUIDE below)

CALIBRATION GUIDE — IPM Source Points
--------------------------------------
The src_pts rectangle must trace where the yellow lines appear in the
*raw 640×480 side-camera image* (after the horizontal flip).

Step 1: Run with show_debug_images:=true and view /side_cam/image_raw.
Step 2: Pause on a frame where both lines are visible.
Step 3: Note the pixel coordinates of the four corners of the lane
        (the outer edges of both yellow lines):
          bottom-left  = outer-left line, lower end
          bottom-right = outer-right line, lower end
          top-left     = outer-left line, upper end
          top-right    = outer-right line, upper end
Step 4: Replace the four numbers in src_pts below.
Step 5: Re-run and verify the BEV debug image shows straight vertical lines.

Current src_pts are the original values — tune these first if lines are
visible in the raw image but invisible in the BEV mask.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import numpy as np
import math


class LotDetector(Node):
    def __init__(self):
        super().__init__('lot_detector')

        anti_lag_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            Image,
            '/side_cam/image_raw',
            self.listener_callback,
            anti_lag_qos
        )
        self.bridge = CvBridge()

        # --- PARAMETERS ---
        self.declare_parameter('show_debug_windows', False)
        self.declare_parameter('show_debug_images', False)   # NEW: publish debug topics

        self.show_debug_windows = bool(self.get_parameter('show_debug_windows').value)
        self.show_debug_images  = bool(self.get_parameter('show_debug_images').value)

        # --- PUBLISHERS ---
        self.metrics_pub = self.create_publisher(Float32MultiArray, '/parking_metrics', 10)

        if self.show_debug_images:
            self.debug_bev_pub  = self.create_publisher(Image, '/lot_detector/debug_bev',  1)
            self.debug_mask_pub = self.create_publisher(Image, '/lot_detector/debug_mask', 1)

        # --- AWB MEMORY (used for display only, NOT for detection) ---
        self.gain_b = 1.0
        self.gain_g = 1.0
        self.gain_r = 1.0

        self.get_logger().info(
            'LotDetector ready — fixed HSV range, blur=7, AWB display-only.'
        )
        if self.show_debug_images:
            self.get_logger().info(
                'Debug topics: /lot_detector/debug_bev  /lot_detector/debug_mask\n'
                '  View: ros2 run rqt_image_view rqt_image_view'
            )

    # ------------------------------------------------------------------
    def listener_callback(self, data):
        # --- AUTO-DECODE: handle both BGR8 (via nv12_to_bgr node) and raw NV12 ---
        if data.encoding.lower() in ('nv12', 'yuv420sp'):
            # NV12 layout: H rows Y  +  H/2 rows interleaved UV
            h, w = data.height, data.width
            raw = np.frombuffer(data.data, dtype=np.uint8)
            expected = h * w * 3 // 2
            if raw.size < expected:
                self.get_logger().warning(f'NV12 frame too small: {raw.size} < {expected}')
                return
            yuv = raw[:expected].reshape((h * 3 // 2, w))
            raw_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        else:
            raw_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        frame = cv2.resize(raw_image, (640, 480))
        frame = cv2.flip(frame, 1)              # RDK X5 camera mirror correction

        # ==============================================================
        # AWB — for DISPLAY only, never for the HSV mask
        # ==============================================================
        tiny = cv2.resize(frame, (64, 48))
        b_t, g_t, r_t = cv2.split(tiny)
        avg_b, avg_g, avg_r = np.mean(b_t), np.mean(g_t), np.mean(r_t)
        avg_all = (avg_b + avg_g + avg_r) / 3.0

        self.gain_b = 0.5*self.gain_b + 0.5*(avg_all / avg_b if avg_b > 0 else 1.0)
        self.gain_g = 0.5*self.gain_g + 0.5*(avg_all / avg_g if avg_g > 0 else 1.0)
        self.gain_r = 0.5*self.gain_r + 0.5*(avg_all / avg_r if avg_r > 0 else 1.0)

        b, g, r = cv2.split(frame)
        b = cv2.convertScaleAbs(b, alpha=self.gain_b)
        g = cv2.convertScaleAbs(g, alpha=self.gain_g)
        r = cv2.convertScaleAbs(r, alpha=self.gain_r)
        display_frame = cv2.merge([b, g, r])   # AWB version — display only

        # ==============================================================
        # IPM (Bird's Eye View) — applied to RAW frame for colour accuracy
        # ==============================================================
        # *** TUNE THESE POINTS — see CALIBRATION GUIDE at top of file ***
        src_pts = np.float32([
            [20.0,  380.0],   # Bottom-Left
            [620.0, 380.0],   # Bottom-Right
            [210.0, 210.0],   # Top-Left
            [430.0, 210.0],   # Top-Right
        ])
        dst_pts = np.float32([
            [200.0, 480.0],
            [440.0, 480.0],
            [200.0,   0.0],
            [440.0,   0.0],
        ])

        matrix    = cv2.getPerspectiveTransform(src_pts, dst_pts)
        bev_frame = cv2.warpPerspective(frame, matrix, (640, 480))   # raw colours

        # Floor mask — valid IPM region
        white_canvas = np.ones((480, 640), dtype=np.uint8) * 255
        floor_mask   = cv2.warpPerspective(white_canvas, matrix, (640, 480))
        floor_mask   = cv2.erode(floor_mask, np.ones((10, 10), np.uint8))

        # Display BEV (AWB version) — for overlay rendering only
        bev_display  = cv2.warpPerspective(display_frame, matrix, (640, 480))

        # ==============================================================
        # YELLOW LINE DETECTION
        # ==============================================================

        # FIX 2: blur kernel 7 (was 21) — small enough to keep thin lines
        blurred = cv2.medianBlur(bev_frame, 7)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # CLAHE on V channel — boosts faint lines in dim environments
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v     = clahe.apply(v)
        hsv_enhanced = cv2.merge([h, s, v])

        # FIX 1: wider HSV range — catches faded / low-saturation yellow paint
        #   Hue:        13–52  (was 18–45)
        #   Saturation: 15–255 (was 40–255)  ← key change
        #   Value:      15–255 (was 30–255)  ← key change
        lower_yellow = np.array([13,  15,  15])
        upper_yellow = np.array([52, 255, 255])
        yellow_mask  = cv2.inRange(hsv_enhanced, lower_yellow, upper_yellow)

        # Morphology — same as original
        kernel_eraser   = np.ones((7, 7),  np.uint8)
        kernel_vertical = np.ones((40, 5), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN,  kernel_eraser)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel_vertical)

        # Mask to valid floor region only
        yellow_mask = cv2.bitwise_and(yellow_mask, floor_mask)

        # ==============================================================
        # CONTOUR FILTERING — same geometry as original
        # ==============================================================
        contours, _ = cv2.findContours(
            yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_lines = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800:
                continue

            rect = cv2.minAreaRect(cnt)
            (cx, cy), (width, height), angle = rect
            length    = max(width, height)
            thickness = min(width, height)

            if thickness == 0:
                continue

            aspect_ratio = length / thickness
            box_area     = length * thickness
            fill_factor  = area / box_area if box_area > 0 else 0

            if length > 80 and thickness < 70 and aspect_ratio > 3.0 and fill_factor > 0.65:
                valid_lines.append({
                    'cx': cx, 'cy': cy,
                    'thickness': thickness, 'rect': rect
                })
                box = np.int32(cv2.boxPoints(rect))
                cv2.drawContours(bev_display, [box], 0, (0, 0, 255), 2)

        valid_lines = sorted(valid_lines, key=lambda r: r['cx'])
        overlay = bev_display.copy()

        line1 = None
        line2 = None

        # ==============================================================
        # MEASURE GAP & PUBLISH — unchanged from original
        # ==============================================================
        if len(valid_lines) >= 2:
            line1 = valid_lines[0]

            for l in valid_lines[1:]:
                if l['cx'] > (line1['cx'] + 15):
                    line2 = l
                    break

            if line2 is not None:
                box1 = cv2.boxPoints(line1['rect'])

                d01 = np.linalg.norm(box1[0] - box1[1])
                d12 = np.linalg.norm(box1[1] - box1[2])

                if d01 > d12:
                    vx = box1[1][0] - box1[0][0]
                    vy = box1[1][1] - box1[0][1]
                else:
                    vx = box1[2][0] - box1[1][0]
                    vy = box1[2][1] - box1[1][1]

                mag = np.sqrt(vx**2 + vy**2)
                if mag > 0:
                    vx, vy = vx / mag, vy / mag
                else:
                    vx, vy = 0, 1

                nx, ny = -vy, vx
                if nx < 0:
                    nx, ny = -nx, -ny

                dx = line2['cx'] - line1['cx']
                dy = line2['cy'] - line1['cy']

                center_distance = abs(dx * nx + dy * ny)
                offset1         = line1['thickness'] / 2
                offset2         = line2['thickness'] / 2
                pixel_gap       = center_distance - offset1 - offset2

                cm_per_pixel_x = 0.200
                cm_per_pixel_y = 0.247
                real_width_cm  = int(pixel_gap * cm_per_pixel_x)

                if real_width_cm > 0:
                    start_x = int(line1['cx'] + nx * offset1)
                    start_y = int(line1['cy'] + ny * offset1)
                    end_x   = int(start_x + nx * pixel_gap)
                    end_y   = int(start_y + ny * pixel_gap)

                    cv2.line(overlay, (start_x, start_y), (end_x, end_y),
                             (0, 255, 255), 3)

                    target_x = start_x + (end_x - start_x) // 2
                    target_y = start_y + (end_y - start_y) // 2

                    box2          = cv2.boxPoints(line2['rect'])
                    box1_sorted   = sorted(box1, key=lambda pt: pt[1])
                    box2_sorted   = sorted(box2, key=lambda pt: pt[1])

                    l1_bottom_y   = (box1_sorted[2][1] + box1_sorted[3][1]) / 2
                    l2_bottom_y   = (box2_sorted[2][1] + box2_sorted[3][1]) / 2
                    l1_bottom_x   = (box1_sorted[2][0] + box1_sorted[3][0]) / 2
                    l2_bottom_x   = (box2_sorted[2][0] + box2_sorted[3][0]) / 2

                    gate_x = int((l1_bottom_x + l2_bottom_x) / 2)
                    gate_y = int(max(l1_bottom_y, l2_bottom_y))

                    line_vx, line_vy = vx, vy
                    if line_vy > 0:
                        line_vx, line_vy = -line_vx, -line_vy

                    screen_angle_rad = math.atan2(line_vx, -line_vy)
                    tilt_degrees     = math.degrees(screen_angle_rad)

                    cv2.circle(overlay, (gate_x, gate_y), 8, (0, 0, 255), -1)
                    cv2.line(overlay, (target_x, target_y), (gate_x, gate_y),
                             (0, 255, 0), 1)

                    dist_fwd_cm  = (480 - gate_y) * cm_per_pixel_y
                    dist_side_cm = (320 - gate_x) * cm_per_pixel_x

                    cam_offset_fwd  = -64.0
                    cam_offset_side =  56.0

                    k_x   = int(dist_fwd_cm  + cam_offset_fwd)
                    k_y   = int(dist_side_cm + cam_offset_side)
                    k_tlt = int(tilt_degrees)

                    if 10 <= real_width_cm <= 100:
                        metrics_msg = Float32MultiArray()
                        metrics_msg.data = [
                            0.0, 0.0,
                            float(k_x), float(k_y), float(k_tlt)
                        ]
                        self.metrics_pub.publish(metrics_msg)

                    text_x = start_x + (end_x - start_x) // 2
                    text_y = start_y + (end_y - start_y) // 2 - 15
                    cv2.putText(overlay, f'WIDTH: {real_width_cm} cm',
                                (text_x - 70, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    if 80 <= real_width_cm <= 97:
                        cv2.putText(overlay, 'PERFECT SPOT!',
                                    (text_x - 70, text_y + 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    cv2.putText(overlay, f'WALL: X={k_x}cm, Y={k_y}cm',
                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(overlay, f'TILT: {k_tlt} DEG',
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # ==============================================================
        # ANOMALY DETECTION (unchanged, safety bypass still active)
        # ==============================================================
        safe_floor_mask = cv2.bitwise_and(
            floor_mask, cv2.bitwise_not(yellow_mask))
        avg_bgr = cv2.mean(
            cv2.medianBlur(bev_frame, 7), mask=safe_floor_mask)[:3]
        avg_background = np.full(bev_frame.shape, avg_bgr, dtype=np.uint8)

        diff      = cv2.absdiff(cv2.medianBlur(bev_frame, 7), avg_background)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, object_mask = cv2.threshold(diff_gray, 40, 255, cv2.THRESH_BINARY)
        object_mask = cv2.bitwise_and(object_mask, safe_floor_mask)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel_eraser)

        cv2.addWeighted(overlay, 0.4, bev_display, 0.6, 0, bev_display)

        # ==============================================================
        # DEBUG OUTPUT
        # ==============================================================
        if self.show_debug_images:
            # BEV overlay
            bev_msg = self.bridge.cv2_to_imgmsg(bev_display, encoding='bgr8')
            bev_msg.header = data.header
            self.debug_bev_pub.publish(bev_msg)

            # Mask — convert mono to BGR so rqt_image_view shows it clearly
            mask_bgr = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
            mask_msg = self.bridge.cv2_to_imgmsg(mask_bgr, encoding='bgr8')
            mask_msg.header = data.header
            self.debug_mask_pub.publish(mask_msg)

        if self.show_debug_windows:
            cv2.imshow('1. Normal img', frame)
            cv2.imshow('2. BEV img',    bev_display)
            cv2.imshow('3. Line Mask',  yellow_mask)
            cv2.imshow('4. Anomaly',    object_mask)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = LotDetector()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
