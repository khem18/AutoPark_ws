#!/usr/bin/env python3
"""
rear_cam_tracker.py  –  rear camera distance & tilt tracker

Distance formula:
  dist_cm = (_BEV_BOTTOM - wall_y) × CM_PER_PIXEL_Y
  wall_y = _TOP_MARGIN  → dist = MAX_VISIBLE_CM = ~240 cm  (wall just visible)
  wall_y = _BEV_BOTTOM  → dist = 0 cm                      (bumper touching wall)

wall_y is found by scanning a LIGHTLY-CLOSED yellow mask (15×5 kernel, max 7.5 px
bias ≈ 5 cm) for the topmost yellow pixel in each side-line column.
This replaces the previous fixed +HALF_K OBB-top correction that had up to 25 px
(17 cm) systematic error.

Only the LEFT and RIGHT slot-edge side lines are used for distance measurement
(cx within EDGE_TOL of _LEFT_MARGIN or _LEFT_MARGIN + _SLOT_W_PX).  Extra blobs
in the middle are filtered out → one clean cyan line in the debug view.

── BEV EXPANSION history ─────────────────────────────────────────────────────
  Round 1 (490):  top y 577→490, canvas 400×600→500×800, DEPTH 248→368 px
  Round 2 (370):  top y 490→370  (+120 px more sky / depth in source)
                  top x 220/970 → 130/1030  (wider far-end capture)
                  bottom x 70/1110 → 40/1150 (wider near-bumper capture)
                  BEV canvas 500×800 → 600×1000
                  _DEPTH_PX int(92×4)=368 → int(120×4)=480 px
                  MAX_VISIBLE_CM ~240 → ~330 cm (re-measure on rig!)
                  LEFT/RIGHT margins 76/424 → 126/474 px (more side room)
  Linear-regression calibration coefficients must be re-measured whenever
  SRC_POINTS change.
──────────────────────────────────────────────────────────────────────────────
"""

import json
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray

# ─── BEV geometry ─────────────────────────────────────────────────────────────
# SRC_POINTS: trapezoid in the source 1280×720 camera image.
#   Round-2 expansion: top pushed up to y=370 (was 490 → was 577).
#   Each ~10 px upward adds ~5–8 cm of real depth at the far end.
#   If the top corners clip the image edge (<0 or >1280) move them inward
#   by equal amounts and reduce MAX_VISIBLE_CM accordingly.
#
#   FIX: top was accidentally left at y=490 (the old value) even though
#   MAX_VISIBLE_CM was updated to 330 cm.  y=490 only captures the close
#   region of the slot; the back wall at ~165 cm was completely outside the
#   warped frame → detection always failed.  Corrected to y=370 as intended.
SRC_POINTS = np.float32([
    [220, 370], [970, 370],   # far edge  (top of BEV)   ← y=370 (was 490)
    [1110, 720], [70, 720],   # near edge (bottom of BEV) ← slightly wider
])

BEV_W, BEV_H  = 500, 800    # canvas px  (was 400×600; extra room for larger view)
_SCALE        = 4.0
_SLOT_W_PX    = int(87 * _SCALE)             # 348 px  – slot is still 87 cm wide
_DEPTH_PX     = int(92 * _SCALE)             # 368 px  (was 248; ~48 % deeper)
_LEFT_MARGIN  = (BEV_W - _SLOT_W_PX) // 2   #  76 px  ← left line cx
_TOP_MARGIN   = (BEV_H - _DEPTH_PX)  // 2   # 216 px  ← wall at max range
_BEV_BOTTOM   = _TOP_MARGIN + _DEPTH_PX      # 584 px  ← bumper level
_RIGHT_MARGIN = _LEFT_MARGIN + _SLOT_W_PX    # 424 px  ← right line cx

DST_POINTS = np.float32([
    [_LEFT_MARGIN,  _TOP_MARGIN],
    [_RIGHT_MARGIN, _TOP_MARGIN],
    [_RIGHT_MARGIN, _BEV_BOTTOM],
    [_LEFT_MARGIN,  _BEV_BOTTOM],
])

# ⚠️  MAX_VISIBLE_CM is an ESTIMATE for the round-2 SRC trapezoid (top y=370).
#     Re-measure physically: place an object at a known distance,
#     note wall_y in the debug view, then:
#       MAX_VISIBLE_CM = (known_dist_cm × _DEPTH_PX) / (_BEV_BOTTOM - observed_wall_y)
MAX_VISIBLE_CM   = 330.0                                  # was 240 cm (estimate; re-measure!)
CM_PER_PIXEL_Y   = MAX_VISIBLE_CM / _DEPTH_PX             # ≈ 0.688 cm/px
STOP_DISTANCE_CM = 20.0

# ─── Kernel heights ───────────────────────────────────────────────────────────
OBB_KERNEL_H  = 50    # for tilt OBB detection (reliable blob, high connectivity)
SCAN_KERNEL_H = 15    # for top-pixel scan    (max 7.5 px bias ≈ 5 cm)

# ─── Side-line edge filter ────────────────────────────────────────────────────
# Only use detections whose cx is within this many px of the slot edge lines.
# Removes false blobs in the middle of the slot.
EDGE_TOL = 40   # px

# ─── HSV thresholds ───────────────────────────────────────────────────────────
YELLOW_LO = np.array([15,  40,  30], dtype=np.uint8)
YELLOW_HI = np.array([45, 255, 255], dtype=np.uint8)

# ─── OBB quality thresholds ───────────────────────────────────────────────────
SIDE_MIN_LEN_PX   = 60
SIDE_MIN_ASPECT   = 3.0
SIDE_MIN_FILL     = 0.45
SIDE_MAX_VERT_DEG = 35.0

# ─── EMA ──────────────────────────────────────────────────────────────────────
EMA_TILT = 0.35
EMA_DIST = 0.40


# ─── Masks ────────────────────────────────────────────────────────────────────

def _base_yellow_mask(bgr_bev):
    """Blur → CLAHE → HSV threshold → OPEN (noise removal, no close yet)."""
    blurred = cv2.medianBlur(bgr_bev, 11)
    h, s, v = cv2.split(cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV))
    v       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(v)
    mask    = cv2.inRange(cv2.merge([h, s, v]), YELLOW_LO, YELLOW_HI)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def _obb_mask(base):
    """50×5 vertical close → reliable connected blobs for OBB tilt detection."""
    return cv2.morphologyEx(
        base, cv2.MORPH_CLOSE, np.ones((OBB_KERNEL_H, 5), np.uint8), iterations=1)


def _scan_mask(base):
    """
    15×5 vertical close → bridges small gaps for accurate top-pixel scan.
    Max morphological bias = SCAN_KERNEL_H/2 = 7.5 px ≈ 5 cm.
    """
    return cv2.morphologyEx(
        base, cv2.MORPH_CLOSE, np.ones((SCAN_KERNEL_H, 5), np.uint8), iterations=1)


# ─── Top-pixel scan ───────────────────────────────────────────────────────────

def find_top_y(smask, cx, half_width=20, min_pix=2):
    """
    Scan downward from _TOP_MARGIN in a column strip of width 2×half_width
    centered at cx.  Return the first row that has ≥ min_pix yellow pixels.
    Returns None if nothing found.

    Using smask (15×5 close) instead of raw mask for reliability while keeping
    bias ≤ 7.5 px (≈ 5 cm).
    """
    x0    = max(0,     cx - half_width)
    x1    = min(BEV_W, cx + half_width)
    strip = smask[_TOP_MARGIN:_BEV_BOTTOM, x0:x1]
    rows  = np.where(np.sum(strip > 0, axis=1) >= min_pix)[0]
    return (_TOP_MARGIN + int(rows[0])) if len(rows) > 0 else None


# ─── Side-line OBB detector ───────────────────────────────────────────────────

def detect_side_lines(obb_mask):
    """
    Detect left & right slot side lines.
    Returns [{cx, cy, tilt_deg, box_pts}, ...]

    Filters:
      • OBB quality (aspect, fill, near-vertical angle)
      • cx must be within EDGE_TOL of the left (_LEFT_MARGIN) or
        right (_RIGHT_MARGIN) slot edge  → removes middle false blobs
    """
    contours, _ = cv2.findContours(
        obb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 400:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), _ = rect
        length, thickness = max(w, h), min(w, h)
        if thickness < 3 or (length / thickness) < SIDE_MIN_ASPECT:
            continue
        if area / (length * thickness) < SIDE_MIN_FILL:
            continue

        # Long-axis angle (boxPoints method, works across OpenCV versions)
        box = cv2.boxPoints(rect)
        d01 = np.linalg.norm(box[1] - box[0])
        d12 = np.linalg.norm(box[2] - box[1])
        lv  = (box[1] - box[0]) if d01 >= d12 else (box[2] - box[1])
        if lv[1] > 0:
            lv = -lv
        lv /= np.linalg.norm(lv) + 1e-9

        if math.degrees(math.atan2(abs(lv[0]), abs(lv[1]) + 1e-9)) > SIDE_MAX_VERT_DEG:
            continue  # too horizontal → skip

        # ── Edge filter: must be near the left or right slot boundary ─────────
        on_left  = abs(cx - _LEFT_MARGIN)  < EDGE_TOL
        on_right = abs(cx - _RIGHT_MARGIN) < EDGE_TOL
        if not (on_left or on_right):
            continue

        tilt = math.degrees(math.atan2(lv[0], abs(lv[1]) + 1e-9))
        results.append({'cx':       int(cx),
                         'cy':       int(cy),
                         'tilt_deg': tilt,
                         'box_pts':  np.int32(box)})
    return results


# ─── Distance measurement ─────────────────────────────────────────────────────

def measure_distance(detections, smask):
    """
    For each detected side line, find the topmost yellow pixel in its column
    using the scan mask (15×5 close).  Falls back to OBB-corrected top if the
    scan finds nothing.

    dist_cm = (_BEV_BOTTOM - wall_y) × CM_PER_PIXEL_Y
    wall_y  = mean of per-line top_y values (one per detected side line)

    Returns (dist_cm, wall_y, wall_exited)
    """
    if not detections:
        return None, None, False

    top_ys      = []
    wall_exited = False

    for d in detections:
        ty = find_top_y(smask, d['cx'])

        if ty is None:
            # Fallback: OBB raw top + half scan-kernel correction
            raw_top = float(np.min(d['box_pts'][:, 1]))
            ty      = raw_top + SCAN_KERNEL_H // 2   # conservative correction

        top_ys.append(ty)
        if ty > _BEV_BOTTOM:
            wall_exited = True

    wall_y = float(np.mean(top_ys))

    # Wall not yet in BEV (car beyond max visible range)
    if wall_y < _TOP_MARGIN:
        return None, wall_y, wall_exited

    dist_cm = (_BEV_BOTTOM - wall_y) * CM_PER_PIXEL_Y
    return float(dist_cm), wall_y, wall_exited


# ─── Hough tilt fallback ──────────────────────────────────────────────────────

def tilt_from_hough(gray_bev):
    edges = cv2.Canny(gray_bev, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40,
                             minLineLength=50, maxLineGap=20)
    if lines is None:
        return None
    devs = []
    for x1, y1, x2, y2 in lines[:, 0]:
        ang = math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1) + 1e-6))
        if ang >= 60:
            devs.append((90 - ang) * (1 if (x2 - x1) >= 0 else -1))
    return float(np.mean(devs)) if devs else None


# ─── ROS 2 node ───────────────────────────────────────────────────────────────

class RearCamTracker(Node):

    def __init__(self):
        super().__init__('rear_cam_tracker')
        self.declare_parameter('debug_view', False)
        self.debug_view = bool(self.get_parameter('debug_view').value)

        self.bridge    = CvBridge()
        self.M_bev     = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
        self.active    = False
        self._tilt_ema = None
        self._dist_ema = None
        self._frame_count = 0   # diagnostic: total frames processed while active

        self.metrics_pub = self.create_publisher(
            Float32MultiArray, '/rear_parking_metrics', 1)
        self.debug_pub = self.create_publisher(
            Image, '/rear_cam_tracker/debug_image', 1)
        self.create_subscription(String, '/autopark/cmd_json', self._cmd_cb, 5)
        self.create_subscription(Image,  '/rear_cam/image_raw', self._image_cb, 1)
        self.get_logger().info(
            f'Ready. Max={MAX_VISIBLE_CM:.0f} cm. Stop≤{STOP_DISTANCE_CM:.0f} cm.')

    def _ema(self, prev, val, a):
        return val if (prev is None or val is None) else a * val + (1 - a) * prev

    def _cmd_cb(self, msg):
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if cmd.get('type') == 'rear_cam_activate':
            # Pre-measure: autopark_master sends this between Move2 and Move3
            self.active = True
            self._tilt_ema = self._dist_ema = None
            self._frame_count = 0
            self.get_logger().info('ACTIVATED (pre-measure for Move3)')
        elif cmd.get('type') == 'drive' and cmd.get('label') == 'rev_straight_d4':
            if not self.active:
                self.active = True
                self._tilt_ema = self._dist_ema = None
                self._frame_count = 0
                self.get_logger().info('ACTIVATED (rev_straight_d4)')
        elif cmd.get('type') == 'stop' and self.active:
            self.active = False
            self.get_logger().info('DEACTIVATED')

    def _image_cb(self, msg):
        if not self.active:
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(str(e), throttle_duration_sec=5.0)
            return

        bgr = cv2.flip(bgr, -1)

        self._frame_count += 1
        tilt_raw, dist_raw, wall_exited, dbg = self._process(bgr)

        self._tilt_ema = self._ema(self._tilt_ema, tilt_raw, EMA_TILT)
        self._dist_ema = self._ema(self._dist_ema, dist_raw, EMA_DIST)
        # Keep 1 decimal: round() to int truncates 0.4° tilt → 0° (sign lost).
        tilt_s = round(self._tilt_ema, 1) if self._tilt_ema is not None else None
        dist_s = round(self._dist_ema)    if self._dist_ema is not None else None

        # Diagnostic: every 25 frames log raw values so you can see detection
        # without the debug window.
        if self._frame_count % 25 == 1:
            self.get_logger().info(
                f'[diag] frame={self._frame_count}'
                f'  tilt_raw={tilt_raw}  dist_raw={dist_raw}'
                f'  tilt_ema={self._tilt_ema}  dist_ema={self._dist_ema}')

        stop = wall_exited or (dist_s is not None and dist_s <= STOP_DISTANCE_CM)

        if self.debug_view and dbg is not None:
            cv2.imshow('Rear Cam Tracker Debug', dbg)
            cv2.waitKey(1)

        out = Float32MultiArray()
        out.data = [
            float(tilt_s) if tilt_s is not None else float('nan'),
            float(dist_s) if dist_s is not None else float('nan'),
        ]
        self.metrics_pub.publish(out)

        if self.debug_pub.get_subscription_count() > 0 and dbg is not None:
            try:
                dm = self.bridge.cv2_to_imgmsg(dbg, 'bgr8')
                dm.header = msg.header
                self.debug_pub.publish(dm)
            except Exception:
                pass

        if wall_exited:
            self.get_logger().warn('WALL EXITED — car past wall!')
        elif dist_s is not None:
            flag = ' *** STOP ***' if stop else ''
            self.get_logger().info(f'tilt={tilt_s:+d}°  dist={dist_s} cm{flag}')
        else:
            self.get_logger().debug('dist=OUT OF RANGE')

    def _process(self, bgr):
        bev  = cv2.warpPerspective(bgr, self.M_bev, (BEV_W, BEV_H))
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)

        base      = _base_yellow_mask(bev)
        omask     = _obb_mask(base)   # 50×5 close → OBB / tilt
        smask     = _scan_mask(base)  # 15×5 close → top pixel scan / distance

        detections  = detect_side_lines(omask)
        tilt_deg    = (float(np.mean([d['tilt_deg'] for d in detections]))
                       if detections else None)
        tilt_source = 'BB'
        if tilt_deg is None:
            tilt_deg    = tilt_from_hough(gray)
            tilt_source = 'Hough'

        dist_cm, wall_y, wall_exited = measure_distance(detections, smask)
        
        if dist_cm is not None:
            # ── Distance calibration (SRC_POINTS top y=370) ──────────────────
            # OLD regression ((dist + 64.8571) / 1.8029) - 26 was fit to SRC top
            # y=490.  With y=370 the perspective warp is re-scaled; back-calculation
            # from a single measured point confirms the raw pixel-to-cm value is
            # already accurate:
            #   raw ≈ 111 cm,  real measured ≈ 111 cm  →  regression correction ≈ 0.
            # Use a simple scale factor (update once more calibration points exist):
            #   scale = real / raw  →  111 / 111 = 1.00  (effectively no-op for now)
            dist_cm = dist_cm * 1.00     # re-measure and adjust once more points known
            # Bound check and rounding
            dist_cm = max(0.0, dist_cm)
            dist_cm = int(round(dist_cm))

        # ── Debug overlay ──────────────────────────────────────────────────
        dbg = bev.copy()

        # Side-line OBBs (magenta)
        for d in detections:
            cv2.drawContours(dbg, [d['box_pts']], 0, (255, 0, 255), 2)
            cv2.circle(dbg, (d['cx'], d['cy']), 4, (255, 0, 255), -1)

        # ONE cyan line at the detected wall position
        if wall_y is not None:
            wy = int(wall_y)
            cv2.line(dbg, (_LEFT_MARGIN, wy), (_RIGHT_MARGIN, wy), (255, 255, 0), 2)
            cv2.putText(dbg, f'WALL y={wy}',
                        (_LEFT_MARGIN + 4, wy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # BEV slot boundary
        cv2.rectangle(dbg, (_LEFT_MARGIN, _TOP_MARGIN),
                      (_RIGHT_MARGIN, _BEV_BOTTOM), (255, 100, 0), 1)

        # Stop line (near BEV bottom = close distance)
        stop_y = int(_BEV_BOTTOM - STOP_DISTANCE_CM / CM_PER_PIXEL_Y)
        cv2.line(dbg, (_LEFT_MARGIN, stop_y), (_RIGHT_MARGIN, stop_y), (0, 0, 255), 1)
        cv2.putText(dbg, f'STOP {STOP_DISTANCE_CM:.0f}cm',
                    (_LEFT_MARGIN + 2, stop_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Metrics overlay
        g = (0, 200, 0)
        if tilt_deg is not None:
            cv2.putText(dbg, f'Tilt [{tilt_source}]: {tilt_deg:+.1f}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, g, 2)
        if wall_exited:
            cv2.putText(dbg, '*** WALL EXITED - STOP ***',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif dist_cm is not None:
            col = (0, 0, 255) if dist_cm <= STOP_DISTANCE_CM else g
            cv2.putText(dbg, f'Dist: {dist_cm} cm',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            if dist_cm <= STOP_DISTANCE_CM:
                cv2.putText(dbg, '*** STOP ***', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(dbg, f'Dist: OUT OF RANGE (>{MAX_VISIBLE_CM:.0f} cm)',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 0), 2)

        return tilt_deg, dist_cm, wall_exited, dbg


def main(args=None):
    rclpy.init(args=args)
    node = RearCamTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
