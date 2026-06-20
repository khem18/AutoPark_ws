import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Bool
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import numpy as np
import math
import time
from collections import deque


# ================================================================
#  EGO POSE ESTIMATOR  –  frame-to-frame differential odometry
#
#  Two modes:
#    LOT VISIBLE   → direct gate fix + velocity = Δgate / Δt
#    LOT HIDDEN    → optical flow on BEV + constant-velocity fallback
#
#  /ego_pose  →  [x_cm, y_cm, theta_deg, vx, vy, omega, confidence]
# ================================================================
class EgoPoseEstimator:

    LOCK_N = 5     # consecutive visible frames to declare "locked"
    DR_LIM = 3.0   # seconds before dead-reckoning is abandoned

    def __init__(self):
        self.x = self.y = self.theta = 0.0
        self.vx = self.vy = self.omega = 0.0

        self.prev_gx = self.prev_gy = self.prev_gt = None
        self.prev_gate_t  = None
        self.prev_cam_gray = None   # raw camera frame (sharp, for DR tracking)
        self.last_t        = None

        self.streak  = 0
        self.locked  = False
        self.traj_px = deque(maxlen=300)

    # ----------------------------------------------------------
    def update_visible(self, gx_cm, gy_cm, gtheta, t, cam_gray, gate_px=None):
        """
        Call every frame the lot IS visible.
        Δpose from consecutive gate frames → velocity.  Direct absolute fix.
        cam_gray: grayscale raw 1920×1080 frame (saved for DR optical flow).
        """
        if self.prev_gate_t is not None:
            dt = max(t - self.prev_gate_t, 1e-4)
            self.vx    = (gx_cm  - self.prev_gx) / dt
            self.vy    = (gy_cm  - self.prev_gy) / dt
            self.omega = (gtheta - self.prev_gt)  / dt

        self.x, self.y, self.theta = gx_cm, gy_cm, gtheta
        self.prev_gx, self.prev_gy, self.prev_gt = gx_cm, gy_cm, gtheta
        self.prev_gate_t   = t
        self.prev_cam_gray = cam_gray
        self.last_t        = t

        self.streak = min(self.streak + 1, self.LOCK_N + 2)
        self.locked = (self.streak >= self.LOCK_N)
        if gate_px:
            self.traj_px.append(gate_px)
        return self._pack('lot_fix', 1.0)

    # ----------------------------------------------------------
    def update_hidden(self, t, cam_gray, M_bev, cm_px_x, cm_px_y,
                      floor_y_start=350):
        """
        Call every frame the lot is NOT visible.

        Tracks floor features in the raw 1920×1080 camera frame (sharp,
        full-resolution), projects the tracked points through the existing
        BEV homography M_bev to get displacement in BEV pixels, then
        multiplies by CM_PX to convert to real cm.

        Why camera frame instead of BEV:
          - No medianBlur(21) destroying texture
          - 1920×1080 vs 640×480 → 9× more pixels to track
          - M_bev already handles the perspective-to-scale conversion

        floor_y_start: first camera row that is floor, not wall (~350 here).
        """
        self.streak = max(0, self.streak - 1)
        self.locked = (self.streak >= self.LOCK_N)

        if self.last_t is None or self.prev_gate_t is None:
            self.prev_cam_gray = cam_gray
            return None

        dt     = max(t - self.last_t, 1e-4)
        dr_age = t - self.prev_gate_t
        if dr_age > self.DR_LIM:
            return None

        # ── 1. Constant-velocity prior ──────────────────────────
        dx_cm  = self.vx    * dt
        dy_cm  = self.vy    * dt
        dtheta = self.omega * dt

        # ── 2. Raw-camera optical flow → BEV space → cm ─────────
        if self.prev_cam_gray is not None:
            try:
                h_cam = cam_gray.shape[0]
                # Restrict feature detection to floor strip only
                floor_prev = self.prev_cam_gray[floor_y_start:h_cam, :]
                floor_curr = cam_gray[floor_y_start:h_cam, :]

                pts = cv2.goodFeaturesToTrack(
                    floor_prev, maxCorners=200,
                    qualityLevel=0.01, minDistance=8, blockSize=7)

                if pts is not None and len(pts) >= 10:
                    # Restore full-frame y coordinates
                    pts_full = pts + np.array([[[0.0, float(floor_y_start)]]])

                    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_cam_gray, cam_gray, pts_full, None,
                        winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS |
                                  cv2.TERM_CRITERIA_COUNT, 15, 0.03))

                    good_prev = pts_full[status == 1]
                    good_curr = curr_pts[status == 1]

                    if len(good_prev) >= 8:
                        # Project both sets through BEV homography
                        def _to_bev(p2d):
                            p = p2d.reshape(-1, 1, 2).astype(np.float32)
                            return cv2.perspectiveTransform(p, M_bev).reshape(-1, 2)

                        bev_p = _to_bev(good_prev)
                        bev_c = _to_bev(good_curr)

                        # Keep only points landing inside BEV canvas
                        inside = ((bev_p[:, 0] > 0)   & (bev_p[:, 0] < 640) &
                                  (bev_p[:, 1] > 0)   & (bev_p[:, 1] < 480) &
                                  (bev_c[:, 0] > 0)   & (bev_c[:, 0] < 640) &
                                  (bev_c[:, 1] > 0)   & (bev_c[:, 1] < 480))

                        if inside.sum() >= 5:
                            mflow  = np.median((bev_c - bev_p)[inside], axis=0)
                            # Scene moves opposite to vehicle motion
                            of_dx  = -float(mflow[0]) * cm_px_x
                            of_dy  = -float(mflow[1]) * cm_px_y

                            # Trust OF more as velocity estimate ages
                            alpha  = min(0.7, 0.3 + 0.4 * (dr_age / self.DR_LIM))
                            dx_cm  = (1 - alpha) * dx_cm + alpha * of_dx
                            dy_cm  = (1 - alpha) * dy_cm + alpha * of_dy

            except Exception:
                pass   # keep constant-velocity if flow fails

        # ── 3. Integrate ────────────────────────────────────────
        self.x     += dx_cm
        self.y     += dy_cm
        self.theta += dtheta

        self.prev_cam_gray = cam_gray
        self.last_t        = t

        conf = max(0.0, 1.0 - dr_age / self.DR_LIM)
        return self._pack('cam_flow', conf)

    # ----------------------------------------------------------
    def _pack(self, mode, confidence):
        return dict(
            x=self.x, y=self.y, theta=self.theta,
            vx=self.vx, vy=self.vy, omega=self.omega,
            locked=self.locked, mode=mode, confidence=confidence,
            streak=self.streak,
        )


# ================================================================
#  LOT DETECTOR  –  original algorithm · 1920×1080 in · BEV 640×480
# ================================================================
class LotDetector(Node):

    def __init__(self):
        super().__init__('lot_detector')
        anti_lag_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.subscription = self.create_subscription(
            Image, '/side_cam/image_raw',
            self.listener_callback, anti_lag_qos)
        self.bridge = CvBridge()

        self.metrics_pub  = self.create_publisher(Float32MultiArray, '/parking_metrics', 10)
        self.obstacle_pub = self.create_publisher(Bool,              '/lot_obstacle',    10)
        self.pose_pub     = self.create_publisher(Float32MultiArray, '/ego_pose',        10)

        self.gain_b = self.gain_g = self.gain_r = 1.0
        self.estimator = EgoPoseEstimator()
        self.frame_cnt = 0

        # REVERTED: window auto-close-on-parking removed per request —
        # window now stays open continuously, same as before that change.
        self.get_logger().info(
            "Lot Detector 1920×1080 → BEV 640×480 | /ego_pose (differential)")

    # ================================================================
    def listener_callback(self, data):
        self.frame_cnt += 1
        t_now = time.monotonic()

        # ── 1. READ  →  1920 × 1080 ────────────────────────────────
        frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        if frame.shape[:2] != (480, 640):
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        frame = cv2.flip(frame, 1)

        if self.frame_cnt == 30:
            cv2.imwrite('/tmp/cal.png', frame)
            self.get_logger().info('Calibration frame → /tmp/cal.png')

        # ── DYNAMIC AWB ─────────────────────────────────────────────
        tiny = cv2.resize(frame, (64, 48))
        b_t, g_t, r_t = cv2.split(tiny)
        avg_all = (np.mean(b_t) + np.mean(g_t) + np.mean(r_t)) / 3.0
        def _g(ch):
            m = np.mean(ch); return avg_all / m if m > 0 else 1.0
        self.gain_b = 0.5*self.gain_b + 0.5*_g(b_t)
        self.gain_g = 0.5*self.gain_g + 0.5*_g(g_t)
        self.gain_r = 0.5*self.gain_r + 0.5*_g(r_t)
        b, g, r = cv2.split(frame)
        frame = cv2.merge([cv2.convertScaleAbs(b, alpha=self.gain_b),
                           cv2.convertScaleAbs(g, alpha=self.gain_g),
                           cv2.convertScaleAbs(r, alpha=self.gain_r)])

        # ── 2. BEV  (src 1920×1080 → dst 640×480) ──────────────────
        # Calibrate: run once, open /tmp/cal.png in GIMP, read coords
        # BL/BR = near parking line (larger y),  TL/TR = far line (smaller y, floor only)
         # SOURCE POINTS: Tracing the yellow lines
        src_pts = np.float32([
            [20.0, 380.0],   # Bottom-Left
            [620.0, 380.0],  # Bottom-Right
            [210.0, 210.0],  # Top-Left 
            [430.0, 210.0]   # Top-Right 
        ])

        # DESTINATION POINTS: Force lines to be perfectly straight vertical
        dst_pts = np.float32([
            [200.0, 480.0],  
            [440.0, 480.0],  
            [200.0, 0.0],    
            [440.0, 0.0]     
        ])

        matrix    = cv2.getPerspectiveTransform(src_pts, dst_pts)
        bev_frame = cv2.warpPerspective(frame, matrix, (640, 480))

        white_canvas = np.ones((480, 640), np.uint8) * 255
        floor_mask   = cv2.warpPerspective(white_canvas, matrix, (640, 480))
        floor_mask   = cv2.erode(floor_mask, np.ones((10, 10), np.uint8))

        # ── 3. YELLOW LINE DETECTION  (original) ───────────────────
        blurred_bev  = cv2.medianBlur(bev_frame, 21)
        # Grayscale of the raw camera frame — used for DR optical flow
        # (sharp, 1920×1080, much better features than the blurred BEV)
        cam_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv          = cv2.cvtColor(blurred_bev, cv2.COLOR_BGR2HSV)
        h, s, v      = cv2.split(hsv)
        clahe        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv_enh      = cv2.merge([h, s, clahe.apply(v)])

        yellow_mask  = cv2.inRange(hsv_enh,
                                   np.array([18, 40,  30]),
                                   np.array([45, 255, 255]))
        kernel_e     = np.ones((7, 7), np.uint8)
        kernel_v     = np.ones((40, 5), np.uint8)
        yellow_mask  = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN,  kernel_e)
        yellow_mask  = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel_v)
        yellow_mask  = cv2.bitwise_and(yellow_mask, floor_mask)

        contours, _  = cv2.findContours(
            yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_lines = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w_, h_), _ = rect
            length = max(w_, h_); thick = min(w_, h_)
            if thick == 0:
                continue
            ar = length / thick
            fill = area / (length * thick)
            if length > 80 and thick < 70 and ar > 3.0 and fill > 0.65:
                valid_lines.append(
                    {'cx': cx, 'cy': cy, 'thickness': thick, 'rect': rect})
                cv2.drawContours(bev_frame,
                                 [np.int32(cv2.boxPoints(rect))], 0, (0,0,255), 2)

        valid_lines = sorted(valid_lines, key=lambda r: r['cx'])
        overlay     = bev_frame.copy()
        line1 = line2 = None

        # ── 4. GAP + GATE  (original) ───────────────────────────────
        cm_per_pixel_x = 0.1825
        cm_per_pixel_y = 0.247

        if len(valid_lines) >= 2:
            line1 = valid_lines[0]
            for l in valid_lines[1:]:
                if l['cx'] > (line1['cx'] + 15):
                    line2 = l
                    break

        ego_state = None
        gate_px   = None
        got_gate  = False

        if line1 and line2:
            box1 = cv2.boxPoints(line1['rect'])
            d01  = np.linalg.norm(box1[0]-box1[1])
            d12  = np.linalg.norm(box1[1]-box1[2])
            if d01 > d12:
                vx = float(box1[1][0]-box1[0][0]); vy = float(box1[1][1]-box1[0][1])
            else:
                vx = float(box1[2][0]-box1[1][0]); vy = float(box1[2][1]-box1[1][1])

            mag = math.sqrt(vx**2 + vy**2)
            if mag > 0: vx, vy = vx/mag, vy/mag
            else:       vx, vy = 0.0, 1.0
            nx, ny = -vy, vx
            if nx < 0: nx, ny = -nx, -ny

            dx, dy  = line2['cx']-line1['cx'], line2['cy']-line1['cy']
            c_dist  = abs(dx*nx + dy*ny)
            o1, o2  = line1['thickness']/2, line2['thickness']/2
            px_gap  = c_dist - o1 - o2
            real_width_cm = int(px_gap * cm_per_pixel_x)

            if real_width_cm > 0:
                sx = int(line1['cx']+nx*o1); sy = int(line1['cy']+ny*o1)
                ex = int(sx+nx*px_gap);       ey = int(sy+ny*px_gap)
                cv2.line(overlay, (sx,sy), (ex,ey), (0,255,255), 3)

                tgt_x, tgt_y = (sx+ex)//2, (sy+ey)//2

                box2 = cv2.boxPoints(line2['rect'])
                b1s  = sorted(box1, key=lambda p: p[1])
                b2s  = sorted(box2, key=lambda p: p[1])
                gate_x = int(((b1s[2][0]+b1s[3][0])+(b2s[2][0]+b2s[3][0]))/4)
                gate_y = int(max((b1s[2][1]+b1s[3][1])/2, (b2s[2][1]+b2s[3][1])/2))
                gate_px = (gate_x, gate_y)

                lvx, lvy = vx, vy
                if lvy > 0: lvx, lvy = -lvx, -lvy
                tilt_deg = math.degrees(math.atan2(lvx, -lvy))

                cv2.circle(overlay, gate_px, 8, (0,0,255), -1)
                cv2.line(overlay, (tgt_x,tgt_y), gate_px, (0,255,0), 1)

                dist_out = (480 - gate_y) * cm_per_pixel_y
                dist_lng = (320 - gate_x) * cm_per_pixel_x
                k_y   = int(dist_lng  + (-4.0))
                k_x   = int(0.5915 * (dist_out + 22.5) + 42.96)
                k_tlt = int(tilt_deg)

                # ── FRAME-TO-FRAME EGO UPDATE ───────────────────
                ego_state = self.estimator.update_visible(
                    k_x, k_y, float(k_tlt), t_now, cam_gray, gate_px)
                got_gate = True

                if 75 <= real_width_cm <= 95:
                    msg = Float32MultiArray()
                    msg.data = [0.0, 0.0, float(k_x), float(k_y), float(k_tlt)]
                    self.metrics_pub.publish(msg)

                cv2.putText(overlay, f"WIDTH: {real_width_cm} cm",
                            (tgt_x-70, tgt_y-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                if 75 <= real_width_cm <= 95:
                    cv2.putText(overlay, "PERFECT SPOT!",
                                (tgt_x-70, tgt_y+30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(overlay, f"WALL: X={k_x}cm  Y={k_y}cm",
                            (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                cv2.putText(overlay, f"TILT: {k_tlt} DEG",
                            (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # ── LOT HIDDEN → optical flow integration ───────────────────
        if not got_gate:
            ego_state = self.estimator.update_hidden(
                t_now, cam_gray, matrix,
                cm_per_pixel_x, cm_per_pixel_y,
                floor_y_start=350)

        # ── 5. PUBLISH + DISPLAY EGO POSE ───────────────────────────
        if ego_state:
            pm = Float32MultiArray()
            pm.data = [float(ego_state[k]) for k in
                       ('x','y','theta','vx','vy','omega','confidence')]
            self.pose_pub.publish(pm)

            mode_tag = ego_state['mode']
            lk_tag   = " ★LOCKED" if ego_state['locked'] \
                       else f" ({ego_state['streak']}/{EgoPoseEstimator.LOCK_N})"
            cv2.putText(overlay,
                        f"EGO X:{ego_state['x']:+.1f} Y:{ego_state['y']:+.1f}"
                        f" θ:{ego_state['theta']:+.1f}°  [{mode_tag}]{lk_tag}",
                        (20,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,180), 2)
            cv2.putText(overlay,
                        f"Vx:{ego_state['vx']:+.1f} Vy:{ego_state['vy']:+.1f}"
                        f"cm/s  ω:{ego_state['omega']:+.1f}°/s"
                        f"  conf:{ego_state['confidence']:.2f}",
                        (20,115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,180), 1)

        # Trajectory trail
        traj = list(self.estimator.traj_px)
        for i in range(1, len(traj)):
            cv2.line(overlay, traj[i-1], traj[i], (255,100,0), 2)
        if traj:
            cv2.circle(overlay, traj[-1], 5, (0,150,255), -1)

        cv2.addWeighted(overlay, 0.4, bev_frame, 0.6, 0, bev_frame)

        # ── 6. OBSTACLE DETECTION  (original) ───────────────────────
        safe_mask   = cv2.bitwise_and(floor_mask, cv2.bitwise_not(yellow_mask))
        avg_bgr     = cv2.mean(blurred_bev, mask=safe_mask)[:3]
        avg_bg      = np.full(bev_frame.shape, avg_bgr, dtype=np.uint8)
        diff_gray   = cv2.cvtColor(cv2.absdiff(blurred_bev, avg_bg), cv2.COLOR_BGR2GRAY)
        _, obj_mask = cv2.threshold(diff_gray, 40, 255, cv2.THRESH_BINARY)
        obj_mask    = cv2.bitwise_and(obj_mask, safe_mask)
        obj_mask    = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, kernel_e)
        obj_cnts, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lot_blocked = False
        if line1 and line2:
            for ocnt in obj_cnts:
                if cv2.contourArea(ocnt) < 800:
                    continue
                M = cv2.moments(ocnt)
                if M["m00"] == 0: continue
                ocx = int(M["m10"]/M["m00"]); ocy = int(M["m01"]/M["m00"])
                x, y, w, h = cv2.boundingRect(ocnt)
                if line1['cx'] <= ocx <= line2['cx']:
                    lot_blocked = True
                    cv2.rectangle(bev_frame,(x,y),(x+w,y+h),(0,0,255),3)
                    cv2.putText(bev_frame,"!! OBSTACLE IN LOT !!",(x-20,y-12),
                                cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
                    cv2.rectangle(bev_frame,(0,0),(640,50),(0,0,200),-1)
                    cv2.putText(bev_frame,"DANGER: OBSTACLE INSIDE LOT — PARKING BLOCKED",
                                (10,35),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                else:
                    cv2.rectangle(bev_frame,(x,y),(x+w,y+h),(255,140,0),2)
                    cv2.putText(bev_frame,"OBJECT (outside lot)",(x-20,y-12),
                                cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,140,0),1)

        self.obstacle_pub.publish(Bool(data=lot_blocked))

        # ── 7. TILED DISPLAY  2×2  480×360 ──────────────────────────
        TILE_W, TILE_H = 480, 360
        panels = [
            (frame,                                        "1 CAMERA 1920x1080"),
            (bev_frame,                                    "2 BEV 640x480"),
            (cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR), "3 YELLOW MASK"),
            (cv2.cvtColor(obj_mask,    cv2.COLOR_GRAY2BGR), "4 OBSTACLE MASK"),
        ]
        tiles = []
        for img, label in panels:
            tile = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0,0), (TILE_W,26), (20,20,20), -1)
            cv2.putText(tile, label, (6,18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,230,230), 1, cv2.LINE_AA)
            tiles.append(tile)

        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        cv2.imshow("LotDetector", grid)
        cv2.waitKey(1)


# ================================================================
def main(args=None):
    rclpy.init(args=args)
    node = LotDetector()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
