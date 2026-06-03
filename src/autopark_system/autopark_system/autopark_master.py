"""
autopark_master.py  —  v_next4
Adds LED lifecycle to match flowchart:
  cam_check_request → check camera → send led:yellow/red
  parking done      → send led:green
  parking error     → send led:red
"""
import json, math, time, threading
from typing import Optional, Dict, Any, List, Tuple
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, Float32MultiArray
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Imu
from .planner_adapter import plan_from_start, result_to_dict


class AutoparkMaster(Node):

    def __init__(self):
        super().__init__("autopark_master")

        for name, default in [
            ("start_switch_topic",       "/autopark/start_switch"),
            ("cam_check_request_topic",  "/autopark/cam_check_request"),
            ("pose_topic",               "/autopark/start_pose"),
            ("parking_metrics_topic",    "/parking_metrics"),
            ("ultrasonic_topic",         "/autopark/ultrasonic"),
            ("command_topic",            "/autopark/cmd_json"),
            ("plan_topic",               "/autopark/plan_result"),
            ("slot_topic",               "/autopark/slot_info"),
            ("flow_distance_topic",      "/autopark/flow_distance"),
            ("imu_topic",                "/imu/data_raw"),
            ("esp32_steer_ready_topic",  "/autopark/esp32_steer_ready"),
            ("planner_mode",                   "right_only"),
            ("use_default_pose_when_missing",   False),
            ("default_start_x",                -1.28),
            ("default_start_y",                 0.70),
            ("default_start_yaw_deg",          180.0),
            # Camera check: False = always pass (for testing without camera)
            ("camera_check_enabled",           False),
            ("camera_pose_max_age_s",          3.0),
            ("speed_scale",                    0.01),
            ("max_command_steer_deg",          30.0),
            ("steer_ready_timeout_s",          7.0),
            ("steer_settle_pause_s",           0.3),
            ("steer_wait_fallback_s",          6.0),
            ("pause_between_commands",         1.2),
            ("forward_turn_speed_mps",         0.225),
            ("reverse_turn_speed_mps",         0.236),
            ("forward_straight_speed_mps",     0.225),
            ("reverse_straight_speed_mps",     0.236),
            ("straight_steer_threshold_deg",   3.0),
            ("max_segment_dist_m",             2.5),
            ("min_chunk_dist_m",               0.10),
            ("drive_time_min_s",               1.0),
            ("drive_time_max_s",               60.0),
            ("imu_arc_stop_enabled",           True),
            ("imu_arc_reversal_enabled",       True),
            ("imu_arc_reversal_min_deg",        10.0),  # min peak before reversal counts
            ("imu_arc_reversal_drop_deg",        3.0),  # drop from peak to trigger stop
            ("imu_arc_min_steer_deg",          8.0),
            ("imu_arc_stop_factor",            0.83),
            ("imu_arc_wait_before_check_s",    0.3),
            # ── Dynamic IMU arc stop factor for stuck/boosted arc speed ──────
            # When encoder_bridge boosts session_arc above threshold (passengers),
            # _wait_stop() switches to this smaller factor so the arc stops a bit
            # earlier — compensates for higher-speed overshoot.
            # NO-LOAD effect: NONE.
            #   session_arc stays at enc_rev_speed=0.040 < threshold(0.060)
            #   → imu_arc_stop_factor always used (unchanged).
            # PASSENGER effect: session_arc boosted to 0.10–0.15 > 0.060
            #   → imu_arc_stop_factor_stuck used instead.
            ("imu_arc_stop_factor_stuck",      0.78),   # slightly < imu_arc_stop_factor
            ("arc_stuck_speed_threshold_mps",  0.06),   # enc_rev_speed × 1.5 = 0.06
            ("flow_straight_stop_enabled",     False),
            ("flow_straight_stop_factor",      0.90),
            ("flow_straight_wait_before_check_s", 0.3),
            ("rear_ultrasonic_stop_m",         0.200),
            ("rear_us_indices",                [6]),
            ("drive_ultrasonic_safety",        True),
            ("ultrasonic_stop_m",              0.050),
            ("enable_stuck_detection",         True),
            # IMU straight heading correction (d1 / d4)
            ("imu_straight_correct_enabled",   True),
            ("imu_straight_correct_thresh_deg", 2.0),   # start correcting if drift exceeds this
            ("imu_straight_correct_gain",       1.5),   # P-gain: correction = -drift * gain
            ("imu_straight_correct_max_deg",    8.0),   # max steer correction allowed (deg)
            ("stuck_check_after_s",            3.0),
            ("stuck_gyro_min_rads",            0.006),
            ("stuck_flow_delta_min_m",         0.005),
            ("stuck_kick_speed_mps",           0.10),
            ("stuck_kick_duration_s",          0.5),
            ("stuck_retry_pause_s",            0.2),
            ("stuck_max_retries",              1),
            # ── Steer stuck kick — passengers increase steering friction ─────
            # When _wait_steer_ready() times out, overshoot the target angle to
            # break static friction, then return.  Repeats up to kick_retries times.
            ("steer_stuck_kick_enabled",       True),
            ("steer_stuck_kick_overshoot",     2.0),   # factor: 2.0 = 200 % of target angle
            ("steer_stuck_kick_hold_s",        0.4),   # seconds per kick phase
            ("steer_stuck_kick_retries",       2),     # max kick attempts before proceeding
            # kick_limit_deg: allows the kick to exceed max_command_steer_deg so
            # the ESP32 P-controller produces higher PWM (overcomes friction) even
            # when target is already at the hardware limit (e.g. 30°).
            # Set ≥ max_command_steer_deg * kick_overshoot (e.g. 30*2.0 = 60 → use 60).
            ("steer_stuck_kick_limit_deg",     60.0),
            # kick_drive_speed_mps: briefly drive the car at this speed during the
            # kick phases.  Rolling wheels reduce tire-ground friction and let the
            # steer servo reach target (dry-steering with passengers requires much
            # more torque than steering while moving).
            # encoder_bridge does NOT intercept these kick CMDs (no dist_m field).
            # Total motion per kick = 2 × kick_hold_s × kick_drive_speed_mps
            # (e.g. 2 × 0.4 × 0.08 = 6.4 cm per kick retry).
            ("steer_stuck_kick_drive_speed_mps", 0.08),
            # kick_timeout_s: per-retry timeout for Phase 3 (static wait after kick).
            # MUCH shorter than steer_ready_timeout_s (10s) — if the kick + motion
            # didn't get the steer there in 3s, it won't in 10s either.
            # Total worst-case kick time = retries × (2×hold_s + kick_timeout_s)
            # e.g. 3 × (0.8 + 3.0) = 11.4s  vs  3 × (0.8 + 10) = 32.4s before.
            ("steer_stuck_kick_timeout_s",        3.0),
            ("flow_valid_required",            False),
            ("send_arm_command_on_start",      False),
            ("disable_ultrasonic_block",       True),
            ("motor_start_delay_s",            1.2),
            ("d4_speed_mps",                   0.003),   # very slow for rear_us reaction time
            # Arc time budget when IMU is not available.
            # Set to slightly more than measured arc time (user: ~10s).
            ("arc_fallback_time_s",            12.0),
            ("min_clearance_m",                0.12),
            ("us_safety_start_delay_s",        0.50),   # grace period before US checks fire
            # ── Rear-camera Move3 correction ──────────────────────────────
            # Set rear_cam_enabled=False to fall back to planner d4 (old behaviour).
            ("rear_cam_enabled",          True),
            # rear_cam_stop_m: distance (m) from rear camera lens to the desired
            # stop point at the slot back wall.
            # = rear-cam offset from bumper (≈0.175 m) + desired bumper-to-wall gap (0.20 m)
            ("rear_cam_stop_m",           0.375),
            # Consecutive valid rear-cam frames needed before trusting the reading.
            ("rear_cam_stable_samples",   5),
            # Maximum time (s) to wait for stable rear-cam frames before giving up.
            ("rear_cam_timeout_s",        5.0),
            ("rear_cam_topic",            "/rear_parking_metrics"),
        ]:
            self.declare_parameter(name, default)

        def gp(n): return self.get_parameter(n).value

        self.planner_mode                  = str(gp("planner_mode"))
        self.use_default_pose_when_missing = bool(gp("use_default_pose_when_missing"))
        self.default_start_x               = float(gp("default_start_x"))
        self.default_start_y               = float(gp("default_start_y"))
        self.default_start_yaw_deg         = float(gp("default_start_yaw_deg"))
        self.camera_check_enabled          = bool(gp("camera_check_enabled"))
        self.camera_pose_max_age_s         = float(gp("camera_pose_max_age_s"))
        self.speed_scale                   = float(gp("speed_scale"))
        self.max_command_steer_deg         = float(gp("max_command_steer_deg"))
        self.steer_ready_timeout_s         = float(gp("steer_ready_timeout_s"))
        self.steer_settle_pause_s          = float(gp("steer_settle_pause_s"))
        self.steer_wait_fallback_s         = float(gp("steer_wait_fallback_s"))
        self.pause_between_commands        = float(gp("pause_between_commands"))
        self.forward_turn_speed_mps        = float(gp("forward_turn_speed_mps"))
        self.reverse_turn_speed_mps        = float(gp("reverse_turn_speed_mps"))
        self.forward_straight_speed_mps    = float(gp("forward_straight_speed_mps"))
        self.reverse_straight_speed_mps    = float(gp("reverse_straight_speed_mps"))
        self.straight_steer_threshold_deg  = float(gp("straight_steer_threshold_deg"))
        self.max_segment_dist_m            = float(gp("max_segment_dist_m"))
        self.min_chunk_dist_m              = float(gp("min_chunk_dist_m"))
        self.drive_time_min_s              = float(gp("drive_time_min_s"))
        self.drive_time_max_s              = float(gp("drive_time_max_s"))
        self.imu_arc_stop_enabled          = bool(gp("imu_arc_stop_enabled"))
        self.imu_arc_reversal_enabled      = bool(gp("imu_arc_reversal_enabled"))
        self.imu_arc_reversal_min_deg      = float(gp("imu_arc_reversal_min_deg"))
        self.imu_arc_reversal_drop_deg     = float(gp("imu_arc_reversal_drop_deg"))
        self._arc_peak: float              = 0.0
        self.imu_arc_min_steer_deg         = float(gp("imu_arc_min_steer_deg"))
        self.imu_arc_stop_factor           = float(gp("imu_arc_stop_factor"))
        self.imu_arc_wait_before_check_s   = float(gp("imu_arc_wait_before_check_s"))
        self.imu_arc_stop_factor_stuck     = float(gp("imu_arc_stop_factor_stuck"))
        self.arc_stuck_speed_threshold     = float(gp("arc_stuck_speed_threshold_mps"))
        self.flow_straight_stop_enabled    = bool(gp("flow_straight_stop_enabled"))
        self.flow_straight_stop_factor     = float(gp("flow_straight_stop_factor"))
        self.flow_straight_wait_before_check_s = float(gp("flow_straight_wait_before_check_s"))
        self.rear_ultrasonic_stop_m        = float(gp("rear_ultrasonic_stop_m"))
        self.rear_us_indices               = list(gp("rear_us_indices"))
        self.drive_ultrasonic_safety       = bool(gp("drive_ultrasonic_safety"))
        self.ultrasonic_stop_m             = float(gp("ultrasonic_stop_m"))
        self.enable_stuck_detection        = bool(gp("enable_stuck_detection"))
        self.imu_straight_correct_enabled  = bool(gp("imu_straight_correct_enabled"))
        self.imu_straight_correct_thresh_deg = float(gp("imu_straight_correct_thresh_deg"))
        self.imu_straight_correct_gain     = float(gp("imu_straight_correct_gain"))
        self.imu_straight_correct_max_deg  = float(gp("imu_straight_correct_max_deg"))
        self.stuck_check_after_s           = float(gp("stuck_check_after_s"))
        self.stuck_gyro_min_rads           = float(gp("stuck_gyro_min_rads"))
        self.stuck_flow_delta_min_m        = float(gp("stuck_flow_delta_min_m"))
        self.stuck_kick_speed_mps          = float(gp("stuck_kick_speed_mps"))
        self.stuck_kick_duration_s         = float(gp("stuck_kick_duration_s"))
        self.stuck_retry_pause_s           = float(gp("stuck_retry_pause_s"))
        self.stuck_max_retries             = int(gp("stuck_max_retries"))
        self.steer_stuck_kick_enabled      = bool(gp("steer_stuck_kick_enabled"))
        self.steer_stuck_kick_overshoot    = float(gp("steer_stuck_kick_overshoot"))
        self.steer_stuck_kick_hold_s       = float(gp("steer_stuck_kick_hold_s"))
        self.steer_stuck_kick_retries      = int(gp("steer_stuck_kick_retries"))
        self.steer_stuck_kick_limit_deg    = float(gp("steer_stuck_kick_limit_deg"))
        self.steer_stuck_kick_drive_speed  = float(gp("steer_stuck_kick_drive_speed_mps"))
        self.steer_stuck_kick_timeout_s    = float(gp("steer_stuck_kick_timeout_s"))
        self.flow_valid_required           = bool(gp("flow_valid_required"))
        self.send_arm_command_on_start     = bool(gp("send_arm_command_on_start"))
        self.disable_ultrasonic_block      = bool(gp("disable_ultrasonic_block"))
        self.motor_start_delay_s           = float(gp("motor_start_delay_s"))
        self.d4_speed_mps                  = float(gp("d4_speed_mps"))
        self.arc_fallback_time_s           = float(gp("arc_fallback_time_s"))
        self.min_clearance_m               = float(gp("min_clearance_m"))
        self.us_safety_start_delay_s       = float(gp("us_safety_start_delay_s"))
        self.rear_cam_enabled        = bool(gp("rear_cam_enabled"))
        self.rear_cam_stop_m         = float(gp("rear_cam_stop_m"))
        self.rear_cam_stable_samples = int(gp("rear_cam_stable_samples"))
        self.rear_cam_timeout_s      = float(gp("rear_cam_timeout_s"))

        # State
        self.latest_pose:       Optional[Pose2D] = None
        self.latest_pose_time:  float            = 0.0
        self.latest_us:         List[float]      = [9.9] * 8
        self.latest_case:       str              = self.planner_mode
        self.flow_distance_m:   Optional[float]  = None
        self.flow_valid:        bool             = False
        self.imu_gyro_z:        float            = 0.0
        self.imu_yaw_deg:       float            = 0.0
        self.last_imu_time:     float            = 0.0
        self.esp32_steer_ready:       bool       = False
        self.esp32_steer_ready_seen:  bool       = False
        self.busy = False
        self.lock = threading.Lock()
        # Rear-camera state (Move3 pre-measure)
        self.rear_cam_dist_m:   Optional[float] = None   # metres to slot back wall
        self.rear_cam_tilt_deg: Optional[float] = None   # yaw_rear in degrees
        self.rear_cam_fresh:    bool            = False
        self._rear_cam_count:   int             = 0      # frames since last activate

        # Encoder closed-loop result event — set when /enc_result arrives
        self.enc_result_event = threading.Event()

        # session_arc_rev_speed_ from encoder_bridge — updated by /enc_status
        # Used to detect when stuck boost has fired (for dynamic arc stop factor).
        self.session_arc_speed: float = 0.0

        # Subscriptions
        self.create_subscription(Bool,              gp("cam_check_request_topic"), self.on_cam_check_request, 10)
        self.create_subscription(Bool,              gp("start_switch_topic"),      self.on_start_switch,       10)
        self.create_subscription(Pose2D,            gp("pose_topic"),              self.on_pose,               10)
        self.create_subscription(Float32MultiArray, gp("parking_metrics_topic"),   lambda m: None,            10)
        self.create_subscription(Float32MultiArray, gp("ultrasonic_topic"),        self.on_ultrasonic,         10)
        self.create_subscription(String,            gp("slot_topic"),              self.on_slot_info,          10)
        self.create_subscription(Float32MultiArray, gp("flow_distance_topic"),     self.on_flow_distance,      20)
        self.create_subscription(Imu,               gp("imu_topic"),               self.on_imu,                50)
        self.create_subscription(Bool,              gp("esp32_steer_ready_topic"), self.on_esp32_steer_ready,  20)
        # v_next6: encoder_bridge publishes /enc_result when closed-loop straight move done
        self.create_subscription(String, "/enc_result", self.on_enc_result, 10)
        # Track encoder_bridge session_arc speed for dynamic arc stop factor
        self.create_subscription(String, "/enc_status", self.on_enc_status, 20)
        # Rear camera metrics for Move3 pre-measure
        self.create_subscription(Float32MultiArray, gp("rear_cam_topic"),
                                 self.on_rear_cam_metrics, 10)

        self.cmd_pub  = self.create_publisher(String, gp("command_topic"), 10)
        self.plan_pub = self.create_publisher(String, gp("plan_topic"),    10)

        self.get_logger().info(
            "autopark_master v_next4\n"
            f"  camera_check_enabled={self.camera_check_enabled}\n"
            f"  rear_us_stop={self.rear_ultrasonic_stop_m*1000:.0f}mm\n"
            f"  rear_cam_enabled={self.rear_cam_enabled}  "
            f"rear_cam_stop_m={self.rear_cam_stop_m:.3f}\n"
            "  Flowchart: cam_check_request → yellow/red LED → 2nd press → park → green LED")

    # ── Camera check request callback ─────────────────────────────────────
    def on_cam_check_request(self, msg: Bool):
        """
        Called when ESP32 transitions to BTN_WAITING_CAM (1st press happened).
        Check if camera has detected the slot origin and reply with LED color.
        """
        if not msg.data:
            return  # btn_state went back to idle — ignore

        self.get_logger().info("CAM CHECK REQUEST received")
        camera_ok = self._check_camera_origin()

        if camera_ok:
            self.get_logger().info("Camera: origin found → send YELLOW LED")
            self._cmd({"type": "led", "color": "yellow"})
        else:
            self.get_logger().warning("Camera: origin NOT found → send RED LED")
            self._cmd({"type": "led", "color": "red"})

    def _check_camera_origin(self) -> bool:
        """
        True if the camera (perception_bridge) has sent a valid, recent start pose.
        Returns True always if camera_check_enabled=False (testing mode).
        """
        if not self.camera_check_enabled:
            self.get_logger().info("  camera_check_enabled=False → bypass (always OK)")
            return True

        if self.latest_pose is None:
            self.get_logger().info("  No pose received yet → camera NOT ready")
            return False

        age = time.monotonic() - self.latest_pose_time
        if age > self.camera_pose_max_age_s:
            self.get_logger().info(
                f"  Pose too old ({age:.1f}s > {self.camera_pose_max_age_s}s) → NOT ready")
            return False

        self.get_logger().info(
            f"  Pose OK (age={age:.1f}s) x={self.latest_pose.x:.3f} "
            f"y={self.latest_pose.y:.3f} → camera READY")
        return True

    # ── Start switch = 2nd press (arm already sent by ESP32) ─────────────
    def on_start_switch(self, msg: Bool):
        """
        In the new flow, the start_switch topic fires when ESP32 confirms the 2nd press
        (btn_state changes to 'parking'). We treat this as the signal to begin planning.
        Note: serial_bridge may still publish on every start_switch status change,
        so we guard with the busy lock.
        """
        if not msg.data:
            return
        self.get_logger().info("START confirmed (2nd press) → begin parking")
        with self.lock:
            if self.busy:
                self.get_logger().warning("already busy — ignored")
                return
            self.busy = True
        threading.Thread(target=self._autopark_thread, daemon=True).start()

    # ── Remaining callbacks ───────────────────────────────────────────────
    def on_pose(self, msg: Pose2D):
        self.latest_pose      = msg
        self.latest_pose_time = time.monotonic()

    def on_ultrasonic(self, msg: Float32MultiArray):
        vals = list(msg.data)
        if len(vals) >= 8:
            self.latest_us = vals[:8]

    def on_flow_distance(self, msg: Float32MultiArray):
        data = [float(x) for x in msg.data]
        if len(data) >= 6:
            self.flow_distance_m = float(data[1])
            self.flow_valid      = bool(data[5] > 0.5)
        elif len(data) >= 2:
            self.flow_distance_m = float(data[1])
            self.flow_valid      = True

    def on_imu(self, msg: Imu):
        now = time.monotonic()
        if self.last_imu_time <= 0.0:
            self.last_imu_time = now; return
        dt = now - self.last_imu_time
        self.last_imu_time = now
        if 0.0 < dt < 0.2:
            wz = float(msg.angular_velocity.z)
            self.imu_gyro_z   = wz
            self.imu_yaw_deg += math.degrees(wz * dt)

    def on_slot_info(self, msg: String):
        try:
            c = str(json.loads(msg.data).get("case", self.planner_mode)).strip().lower()
            if c in ("left_only", "right_only", "both_sides"):
                self.latest_case = c
        except Exception:
            pass

    def on_esp32_steer_ready(self, msg: Bool):
        self.esp32_steer_ready      = bool(msg.data)
        self.esp32_steer_ready_seen = True

    def on_enc_result(self, msg):
        """encoder_bridge finished a closed-loop straight move — signal _wait_stop."""
        self.enc_result_event.set()
        self.get_logger().info(f"ENC_RESULT received: {msg.data}")

    def on_enc_status(self, msg: String):
        """
        Track encoder_bridge session_arc_rev_speed_ for dynamic arc stop factor.
        enc_status JSON: {"rc":..., "rd":..., "rrpm":..., "rspd":...,
                          "sess_fwd":..., "sess_rev":..., "sess_arc":...}
        Called ~10 Hz from encoder_bridge timer.
        """
        try:
            obj = json.loads(msg.data)
            arc = obj.get('sess_arc')
            if arc is not None:
                self.session_arc_speed = float(arc)
        except Exception:
            pass

    def on_rear_cam_metrics(self, msg: Float32MultiArray):
        """
        Receives [tilt_deg, dist_cm] from rearcam_tracker on /rear_parking_metrics.
        tilt_deg = yaw_rear (angle between car axis and slot lines, degrees)
        dist_cm  = distance from rear camera to the slot back wall (cm)
        NaN values mean the camera hasn't detected the lines yet.
        """
        data = list(msg.data)
        if len(data) < 2:
            return
        tilt   = float(data[0])
        dist_c = float(data[1])
        if math.isnan(tilt) or math.isnan(dist_c) or dist_c <= 0.0:
            return  # not yet a valid reading
        self.rear_cam_tilt_deg = tilt
        self.rear_cam_dist_m   = dist_c / 100.0
        self.rear_cam_fresh    = True
        self._rear_cam_count  += 1

    # ── Rear-camera Move3 helpers ─────────────────────────────────────────

    def _activate_rear_cam(self):
        """
        Ask rearcam_tracker to start processing frames.
        Resets frame counter so _wait_rear_cam_stable() counts from NOW.
        """
        self.rear_cam_fresh   = False
        self._rear_cam_count  = 0
        self._cmd({"type": "rear_cam_activate"})
        self.get_logger().info("Rear cam activated — waiting for stable reading")

    def _wait_rear_cam_stable(self) -> bool:
        """
        Block until rear_cam_stable_samples consecutive valid frames arrive,
        or rear_cam_timeout_s elapses.
        Returns True if a usable reading was obtained (even with fewer frames
        than requested if we at least got one valid frame before timeout).
        """
        deadline = time.monotonic() + self.rear_cam_timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            if self.rear_cam_fresh and self._rear_cam_count >= self.rear_cam_stable_samples:
                self.get_logger().info(
                    f"Rear cam stable: dist={self.rear_cam_dist_m:.3f}m "
                    f"tilt={self.rear_cam_tilt_deg:+.1f}°  "
                    f"(frames={self._rear_cam_count})")
                return True
            time.sleep(0.05)
        # Timeout — return True anyway if we have at least 1 valid frame
        if self.rear_cam_fresh:
            self.get_logger().warning(
                f"Rear cam: timeout but have 1 reading "
                f"dist={self.rear_cam_dist_m:.3f}m tilt={self.rear_cam_tilt_deg:+.1f}° "
                f"— using it")
            return True
        self.get_logger().warning("Rear cam: timeout with NO valid reading")
        return False

    def _compute_move3_from_rear_cam(self, motion: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override Move3 steer_deg and dist_m using the rear camera reading.

        Coordinate conventions (from diagram, page 101):
          y_rear     = distance from rear camera to slot back wall  (metres)
          yaw_rear   = tilt angle between car axis and slot axis    (degrees)
          a          = y_rear * cos(yaw_rear)  → depth component
          x_rear     = y_rear * sin(yaw_rear)  → lateral component

        Formulas:
          steer_move3 = -yaw_rear           (steer corrects angular misalignment)
          d3 = sqrt( (a − rear_cam_stop_m)² + x_rear² )
                                            (Euclidean distance to target)

        rear_cam_stop_m ≈ 0.375 m
          = camera-to-bumper offset (≈0.175 m) + desired bumper-to-wall gap (0.20 m)

        Falls back to the planner's original motion if data is missing.
        """
        if self.rear_cam_dist_m is None or self.rear_cam_tilt_deg is None:
            self.get_logger().warning("Move3 rear_cam: no data — keeping planner d4")
            return motion

        y_rear    = self.rear_cam_dist_m
        yaw_rad   = math.radians(self.rear_cam_tilt_deg)
        a         = y_rear * math.cos(yaw_rad)          # depth component
        x_rear    = y_rear * math.sin(yaw_rad)          # lateral component

        remaining_depth = a - self.rear_cam_stop_m
        d3 = math.sqrt(remaining_depth ** 2 + x_rear ** 2)

        # Clamp steer; note: no_imu_correct=True keeps is_turning=False so
        # the encoder (not IMU arc) still controls the stop regardless of steer.
        steer_deg = self.clamp(
            +self.rear_cam_tilt_deg,                  # +yaw_rear (corrected sign)
            -self.max_command_steer_deg,
            self.max_command_steer_deg)

        self.get_logger().info(
            f"Move3 rear_cam override: y_rear={y_rear:.3f}m  yaw={self.rear_cam_tilt_deg:+.1f}°\n"
            f"  a={a:.3f}m  x_rear={x_rear:.3f}m  "
            f"remaining_depth={remaining_depth:.3f}m\n"
            f"  → d3={d3:.3f}m  steer={steer_deg:+.1f}°")

        if d3 < 0.02:
            self.get_logger().warning(
                f"Move3: computed d3={d3:.4f}m is very small "
                f"(car may already be near target) — clamping to 0.05 m")
            d3 = 0.05

        updated = dict(motion)
        updated["dist_m"]    = round(d3, 4)
        updated["steer_deg"] = round(steer_deg, 2)
        return updated

    # ── Parking thread ────────────────────────────────────────────────────
    def _autopark_thread(self):
        try:
            self._run_autopark()
        except Exception as exc:
            self.get_logger().error(f"thread: {exc}")
            self._led("red")
            self._stop("thread_error")
        finally:
            with self.lock:
                self.busy = False

    def _run_autopark(self):
        self.get_logger().info("AUTOPARK START")

        if self.send_arm_command_on_start:
            self._cmd({"type": "arm"})
            time.sleep(0.2)

        pose = self._get_pose()
        if pose is None:
            self._led("red"); self._stop("no_start_pose"); return

        yaw_deg  = math.degrees(pose.theta)
        case     = (self.latest_case
                    if self.latest_case in ("left_only","right_only","both_sides")
                    else self.planner_mode)

        self.get_logger().info(
            f"Planning: x={pose.x:.3f} y={pose.y:.3f} yaw={yaw_deg:.1f}° case={case}")

        try:
            planned = plan_from_start(pose.x, pose.y, yaw_deg, case)
            result  = result_to_dict(planned)
        except Exception as exc:
            self.get_logger().error(f"planner: {exc}")
            self._led("red"); self._stop("planner_exception"); return

        if not result.get("success"):
            self.get_logger().warning(f"planner failed: {result.get('reason')}")
            self.plan_pub.publish(String(data=json.dumps(result)))
            self._led("red"); self._stop("planner_failed"); return

        motions = result.get("executable_motions") or result.get("motions", [])
        if not motions:
            self._led("red"); self._stop("no_motions"); return

        self.plan_pub.publish(String(data=json.dumps(result)))
        m = result.get("metrics", {})
        self.get_logger().info(
            f"Plan OK d1={m.get('d1_m',0):.3f} s23={m.get('s23_m',0):.3f} "
            f"d4={m.get('d4_m',0):.3f}")

        success = self._execute(motions)

        if success:
            # ── Parking done: send green LED ──────────────────────────────
            self._led("green")
            self.get_logger().info("PARKING DONE → Green LED 1s")
        else:
            self._led("red")
            self.get_logger().warning("PARKING FAILED → Red LED")

    # ── Execute motions ───────────────────────────────────────────────────
    def _execute(self, motions: List[Dict[str, Any]]) -> bool:
        total = len(motions)
        self.get_logger().info(f"EXECUTE {total} moves")

        for i, motion in enumerate(motions):
            seg   = i + 1
            label = str(motion.get("label", f"seg{seg}"))
            use_rear_us = bool(motion.get("use_rear_us", False))

            # ── Rear camera detection + steer/dist calculation (Move 3) ───
            # Flow (per page-101 diagram):
            #   1. Activate rear cam
            #   2. Block until back wall detected (camera within range)
            #   3. Compute steer = -yaw_rear  and  d3 from formula
            #   4. Steer settle (below) uses the computed steer_deg
            #   5. _wait_steer_ready() → steer_ready fires when servo at target
            #   6. Drive CMD sent → encoder_bridge drives d3 and publishes enc_result
            # If camera times out: fall back to planner steer=0 / d4.
            # During the actual drive, _wait_stop() also monitors real-time camera
            # distance as an additional stop guard.
            if motion.get("use_rear_cam", False) and self.rear_cam_enabled:
                self.get_logger().info(
                    f"Move3: activating rear cam — waiting for back-wall detection "
                    f"(timeout={self.rear_cam_timeout_s:.0f}s, "
                    f"need={self.rear_cam_stable_samples} frames)")
                self._activate_rear_cam()
                if self._wait_rear_cam_stable():
                    motion = self._compute_move3_from_rear_cam(motion)
                else:
                    self.get_logger().warning(
                        "Move3: rear cam timeout — using planner steer=0° / d4. "
                        "Tip: check BEV calibration (SRC_POINTS) or increase "
                        "rear_cam_timeout_s in YAML if wall is barely in range.")

            cmd   = self._motion_to_cmd(motion)
            dist  = float(cmd["target_dist_m"])
            dt    = float(cmd.get("master_duration", cmd["duration"]))  # pure travel time
            is_turning = abs(cmd["steer_deg"]) > self.straight_steer_threshold_deg

            # FIX: read no_imu_correct HERE — before steer_ready wait — so
            # Move1 (no_imu_correct=True) gets steer_ready + motor_start_delay
            # exactly like Move3. Previously this was read AFTER the steer_ready
            # check, causing Move1 to skip it and drive instantly without steering.
            no_imu_correct = bool(motion.get("no_imu_correct", False))
            if no_imu_correct:
                is_turning = False   # encoder stops it, not IMU arc (Move1 & Move3)

            self.get_logger().info(
                f"MOVE {seg}/{total} [{label}]  "
                f"gear={cmd['gear']} steer={cmd['steer_deg']:+.0f}°  "
                f"dist={dist:.3f}m  t={dt:.2f}s")

            # ── Steer settle ─────────────────────────────────────────────
            steer_active_hold = bool(cmd.get("steer_active_hold", False))
            self.get_logger().info(
                f"  steer_active_hold={steer_active_hold}  "
                f"({'straight=True keeps motor on' if steer_active_hold else 'arc=False locks then off'})")
            steer_gear = int(cmd["gear"]) or 1

            # All moves: autopark_master sends settle CMD via serial_bridge.
            # ESP32 fires steer_ready reliably through this channel.
            # encoder_bridge calls driveWithSteer() immediately after.
            self.esp32_steer_ready      = False
            self.esp32_steer_ready_seen = False
            # For no_imu_correct (Move1): settle CMD must outlast the full drive.
            # Current 21s expires 2s into the drive → spring briefly pulls steer off target.
            # Fix: extend to 50s so it covers settle (19s) + full drive (~19s) + margin.
            settle_extra = 48.0 if no_imu_correct else 2.0
            self._cmd({"type":"drive","gear":steer_gear,"speed_mps":0.0,
                       "steer_deg":cmd["steer_deg"],
                       "duration": self.steer_ready_timeout_s + self.steer_settle_pause_s + settle_extra,
                       "steer_active_hold": steer_active_hold})
            if not self._wait_steer_ready():
                self.get_logger().warning("  steer_ready timeout")
                # Steer stuck: kick overshoot to overcome passenger-weight friction.
                if self.steer_stuck_kick_enabled and abs(cmd["steer_deg"]) > 0.5:
                    settled = self._steer_stuck_kick(
                        cmd["steer_deg"], steer_gear, steer_active_hold, settle_extra)
                    if not settled:
                        self.get_logger().warning(
                            "  steer stuck: all kick retries failed — proceeding anyway")
            if self.steer_settle_pause_s > 0:
                self._sleep(self.steer_settle_pause_s)

            # ── Drive ────────────────────────────────────────────────────
            imu_start  = self.imu_yaw_deg
            flow_start = self.flow_distance_m
            # Clear encoder result event before sending drive cmd (straight moves)
            if not is_turning:
                self.enc_result_event.clear()
            self._cmd(cmd)

            # Motor-start delay: STRAIGHT moves only (Move1 + Move3).
            # Arc (Move2): IMU stops it; motor delay would cause overshoot.
            if not is_turning:
                motor_delay = self._wait_motor_start(flow_start)
                self.get_logger().info(
                    f"  Timer start after {motor_delay:.2f}s motor delay")
            else:
                self.get_logger().info("  Arc: no motor_start_delay (IMU/time stops arc)")

            stop, elapsed, retries = self._wait_stop(
                dt, dist, cmd["steer_deg"], is_turning, use_rear_us,
                imu_start, flow_start, seg, no_imu_correct=no_imu_correct,
                use_rear_cam=bool(motion.get("use_rear_cam", False) and self.rear_cam_enabled))

            imu_d = self._adelta(imu_start, self.imu_yaw_deg)
            self.get_logger().info(
                f"  LOG: stop={stop} elapsed={elapsed:.2f}s "
                f"imu_delta={imu_d:.2f}°  us={self.get_ultrasonic_min_m():.3f}m")

            self._stop(stop)

            if stop.startswith("ultrasonic_emergency"):
                self.get_logger().warning("ABORT — emergency US")
                return False
            if stop.startswith("stuck_failed"):
                self.get_logger().warning("ABORT — stuck")
                return False

            self._sleep(self.pause_between_commands)

        self._stop("parking_sequence_done")
        return True

    # ── Wait / sensor stop ────────────────────────────────────────────────
    def _wait_stop(self, drive_time_s, dist_m, steer_deg, is_turning,
                   use_rear_us, imu_start, flow_start, seg,
                   no_imu_correct: bool = False,
                   use_rear_cam:   bool = False) -> Tuple[str,float,int]:
        start_t = time.monotonic(); last_log = 0.0; retries = 0; stuck_done = False
        self._arc_peak = 0.0  # reset arc peak for reversal detection

        arc_trig: Optional[float] = None
        if is_turning and self.imu_arc_stop_enabled and abs(steer_deg) > 0.5:
            Rv = 1.335
            arc_trig = math.degrees((dist_m * self.imu_arc_stop_factor) / Rv)

        # Track whether we've already switched to the stuck-speed arc stop factor.
        arc_stuck_factor_applied: bool = False

        flow_trig: Optional[float] = None
        if not is_turning and self.flow_straight_stop_enabled and flow_start is not None:
            flow_trig = dist_m * self.flow_straight_stop_factor

        # IMU straight heading correction — only for straight moves (d1, d4).
        # If yaw drifts beyond imu_straight_correct_thresh_deg, send a corrective
        # steer command to nudge the car back on heading.
        # Move1 (gear=+1): sends type="drive" — passes serial_bridge (gear>0 not blocked).
        # Move3 (gear=-1, use_rear_cam): IMU correction disabled — the encoder_bridge
        #   drives with the rear-cam-computed steer_deg, and the car's wheel asymmetry
        #   is a hardware issue that the IMU correction cannot reliably fix without
        #   causing runaway drift accumulation. Use hardware calibration instead.
        # no_imu_correct only disables the IMU ARC stop — straight correction is separate.
        imu_straight_correct = (
            not is_turning
            and self.imu_straight_correct_enabled
            and not use_rear_cam   # Move3: encoder_bridge handles steer; IMU disabled
        )
        last_correction_t: float = 0.0
        current_correction_deg: float = 0.0

        while rclpy.ok():
            now = time.monotonic(); elapsed = now - start_t

            # Rear US primary (d4) — grace period prevents false-trigger at seg start
            if use_rear_us and elapsed >= self.us_safety_start_delay_s:
                if self.get_rear_us_m() <= self.rear_ultrasonic_stop_m:
                    return (f"rear_us_stop_seg_{seg}", elapsed, retries)

            # Emergency US — grace period prevents false-trigger from sensor noise at start
            if self.drive_ultrasonic_safety and elapsed >= self.us_safety_start_delay_s:
                if self.get_ultrasonic_min_m() < self.ultrasonic_stop_m:
                    return (f"ultrasonic_emergency_seg_{seg}", elapsed, retries)

            # IMU arc — dynamic stop factor when stuck boost is active
            # (encoder_bridge boosted session_arc above threshold → higher speed
            #  → switch to smaller factor so arc stops earlier / avoids overshoot).
            # Only fires once per arc move; NO effect on no-load autopark
            # (session_arc stays at enc_rev_speed=0.04 < threshold=0.06).
            if (arc_trig is not None
                    and not arc_stuck_factor_applied
                    and self.imu_arc_stop_factor_stuck < self.imu_arc_stop_factor
                    and self.session_arc_speed >= self.arc_stuck_speed_threshold):
                Rv_dyn = 1.335
                old_trig  = arc_trig
                arc_trig  = math.degrees(
                    (dist_m * self.imu_arc_stop_factor_stuck) / Rv_dyn)
                arc_stuck_factor_applied = True
                self.get_logger().info(
                    f"  ARC stuck-speed boost detected "
                    f"(sess_arc={self.session_arc_speed:.3f} m/s"
                    f" >= {self.arc_stuck_speed_threshold:.3f} m/s) "
                    f"→ arc_trig {old_trig:.1f}° → {arc_trig:.1f}° "
                    f"(factor {self.imu_arc_stop_factor:.2f}"
                    f" → {self.imu_arc_stop_factor_stuck:.2f})")
            if arc_trig and elapsed >= self.imu_arc_wait_before_check_s:
                cur_arc = abs(self._adelta(imu_start, self.imu_yaw_deg))
                if cur_arc >= arc_trig:
                    return (f"imu_arc_stop_seg_{seg}", elapsed, retries)
                # Arc reversal detection: if arc decreases from its peak, car rotated back.
                # Stop immediately and proceed to next move.
                if self.imu_arc_reversal_enabled:
                    if cur_arc > self._arc_peak:
                        self._arc_peak = cur_arc
                    elif (self._arc_peak >= self.imu_arc_reversal_min_deg
                            and (self._arc_peak - cur_arc) >= self.imu_arc_reversal_drop_deg):
                        self.get_logger().warning(
                            f"  IMU_ARC_REVERSAL peak={self._arc_peak:.1f}° "
                            f"current={cur_arc:.1f}° -> stop arc early")
                        return (f"imu_arc_reversal_seg_{seg}", elapsed, retries)

            # Flow straight
            if (flow_trig and self.flow_valid and self.flow_distance_m is not None
                    and elapsed >= self.flow_straight_wait_before_check_s):
                if abs((self.flow_distance_m or 0.0) - (flow_start or 0.0)) >= flow_trig:
                    return (f"flow_stop_seg_{seg}", elapsed, retries)

            # Encoder closed-loop done — straight moves only
            # encoder_bridge publishes /enc_result when it reaches the target distance.
            # This is the primary stop condition for straight moves when encoder is used.
            if not is_turning and self.enc_result_event.is_set():
                return (f"enc_result_stop_seg_{seg}", elapsed, retries)

            # Rear camera distance stop (Move3) — triggers when back wall enters
            # camera range and dist ≤ rear_cam_stop_m (default 37.5 cm from lens).
            # Fires BEFORE enc_result if the wall is reached sooner than planner d4.
            # Grace period = us_safety_start_delay_s avoids false trigger at start.
            if (use_rear_cam and elapsed >= self.us_safety_start_delay_s
                    and self.rear_cam_fresh and self.rear_cam_dist_m is not None):
                if self.rear_cam_dist_m <= self.rear_cam_stop_m:
                    self.get_logger().info(
                        f"  REAR_CAM_STOP: dist={self.rear_cam_dist_m*100:.1f}cm "
                        f"≤ {self.rear_cam_stop_m*100:.0f}cm target")
                    return (f"rear_cam_stop_seg_{seg}", elapsed, retries)

            # IMU straight heading correction
            if imu_straight_correct and elapsed >= 0.3:
                drift = self._adelta(imu_start, self.imu_yaw_deg)
                cmd_gear = 1 if seg == 1 else -1  # seg1=fwd_setup(+1), seg3=d4(-1)
                if abs(drift) >= self.imu_straight_correct_thresh_deg:
                    new_corr = self.clamp(
                        -drift * self.imu_straight_correct_gain,
                        -self.imu_straight_correct_max_deg,
                        +self.imu_straight_correct_max_deg)
                    if (abs(new_corr - current_correction_deg) >= 1.0
                            or now - last_correction_t >= 0.5):
                        current_correction_deg = new_corr
                        last_correction_t = now
                        if use_rear_cam and cmd_gear < 0:
                            # Move3: encoder_bridge drives directly; serial_bridge
                            # blocks gear<0 speed>0 CMDs.  Use steer_update so
                            # encoder_bridge.setLiveSteer() corrects the angle
                            # without aborting the running driveStraight().
                            self._cmd({"type": "steer_update",
                                       "steer_deg": round(current_correction_deg, 1)})
                        else:
                            # Move1: serial_bridge forwards gear>0 CMDs normally.
                            self._cmd({"type": "drive", "gear": cmd_gear,
                                       "speed_mps": self.speed_scale,
                                       "steer_deg": round(current_correction_deg, 1),
                                       "duration": drive_time_s,
                                       "steer_active_hold": True})
                        self.get_logger().info(
                            f"  IMU_CORR drift={drift:.1f}° -> steer={current_correction_deg:.1f}°")
                elif abs(drift) < self.imu_straight_correct_thresh_deg * 0.3 and current_correction_deg != 0.0:
                    current_correction_deg = 0.0
                    last_correction_t = now
                    if use_rear_cam and cmd_gear < 0:
                        self._cmd({"type": "steer_update", "steer_deg": 0.0})
                    else:
                        self._cmd({"type": "drive", "gear": cmd_gear,
                                   "speed_mps": self.speed_scale,
                                   "steer_deg": 0.0,
                                   "duration": drive_time_s,
                                   "steer_active_hold": True})
                    self.get_logger().info("  IMU_CORR heading recovered -> steer=0°")

            # Stuck
            if (self.enable_stuck_detection and not stuck_done
                    and elapsed >= self.stuck_check_after_s):
                stuck_done = True
                if is_turning and abs(self.imu_gyro_z) < self.stuck_gyro_min_rads:
                    if retries >= self.stuck_max_retries:
                        return (f"stuck_failed_seg_{seg}", elapsed, retries)
                    retries += 1; self._stop(f"stuck_retry_{retries}"); self._sleep(self.stuck_retry_pause_s)

            # Log
            if now - last_log >= 0.5:
                ru = f" rear_us={self.get_rear_us_m()*1000:.0f}mm" if use_rear_us else ""
                rc = (f" rcam={self.rear_cam_dist_m*100:.0f}cm"
                      if use_rear_cam and self.rear_cam_dist_m is not None else "")
                ar = ""
                if arc_trig:
                    ar = f" arc={abs(self._adelta(imu_start, self.imu_yaw_deg)):.1f}/{arc_trig:.1f}°"
                self.get_logger().info(
                    f"  DRIVE {elapsed:.2f}/{drive_time_s:.2f}s{ar}{ru}{rc}")
                last_log = now

            # Time fallback
            if elapsed >= drive_time_s:
                return (f"time_stop_seg_{seg}", elapsed, retries)

            time.sleep(0.04)

        return ("ros_shutdown", time.monotonic()-start_t, retries)

    # ── LED helper ────────────────────────────────────────────────────────
    def _led(self, color: str):
        self._cmd({"type": "led", "color": color})
        self.get_logger().info(f"LED → {color}")

    # ── Steer ready ───────────────────────────────────────────────────────
    def _wait_steer_ready(self, timeout_s: float = None) -> bool:
        t = self.steer_ready_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + t
        self.esp32_steer_ready      = False
        self.esp32_steer_ready_seen = False  # ensure fresh wait for THIS move's steer_ready
        while rclpy.ok() and time.monotonic() < deadline:
            if self.esp32_steer_ready: return True
            # Fallback: if ESP32 never acknowledged at all, stop early.
            # Only applies when fallback_s < t (i.e. normal full-timeout calls).
            if (not self.esp32_steer_ready_seen
                    and self.steer_wait_fallback_s < t
                    and time.monotonic() > deadline - t + self.steer_wait_fallback_s):
                return False
            time.sleep(0.04)
        return False

    def _wait_motor_start(self, flow_start) -> float:
        """Wait for car to start moving. Straight moves only."""
        t0 = time.monotonic()
        if self.flow_valid and flow_start is not None:
            deadline = t0 + 3.0
            while rclpy.ok() and time.monotonic() < deadline:
                if self.flow_distance_m is not None:
                    if abs((self.flow_distance_m or 0.0) - (flow_start or 0.0)) >= 0.003:
                        delay = time.monotonic() - t0
                        self.get_logger().info(f"  Motor start: flow detected after {delay:.2f}s")
                        return delay
                time.sleep(0.02)
        delay = self.motor_start_delay_s
        self._sleep(delay)
        self.get_logger().info(
            f"  Motor start: fixed delay {delay:.2f}s (set motor_start_delay_s in YAML)")
        return delay

    # ── Steer stuck kick ──────────────────────────────────────────────────
    def _steer_stuck_kick(self, target_deg: float, gear: int,
                          active_hold: bool, settle_extra: float) -> bool:
        """
        Called when _wait_steer_ready() times out (steer not reaching target).

        Root cause: passengers increase tire-ground friction ("dry steering").
        Two mechanisms work together:
          1. OVERSHOOT angle (2× target) → high steer PWM
          2. SLOW MOTION during kick → rolling wheels reduce tire friction

        Sequence per retry:
          Phase 1 — overshoot + motion:    kick_hold_s sec (breaks friction)
          Phase 2 — target   + motion:     kick_hold_s sec — polls steer_ready
                                           EXITS IMMEDIATELY when steer fires
          Phase 3 — target   + speed=0:    kick_timeout_s sec (short, NOT 10s)

        Worst-case time per retry = 2×kick_hold_s + kick_timeout_s
          e.g. 2×0.4 + 3.0 = 3.8s   vs.  previous 10s per retry

        Returns True if steer_ready fires at any point during any retry.
        """
        settle_dur = (self.steer_ready_timeout_s
                      + self.steer_settle_pause_s
                      + settle_extra)
        drv     = self.steer_stuck_kick_drive_speed
        k_tout  = self.steer_stuck_kick_timeout_s   # short per-retry timeout

        for attempt in range(self.steer_stuck_kick_retries):
            kick_deg = self.clamp(
                target_deg * self.steer_stuck_kick_overshoot,
                -self.steer_stuck_kick_limit_deg,
                self.steer_stuck_kick_limit_deg)

            self.get_logger().info(
                f"  steer stuck kick {attempt + 1}/{self.steer_stuck_kick_retries}: "
                f"overshoot {kick_deg:+.1f}° + drive {drv:.3f} m/s "
                f"(target={target_deg:+.1f}°)")

            # ── Phase 1: overshoot + motion — breaks static friction ───────
            self._cmd({"type": "drive", "gear": gear, "speed_mps": drv,
                       "steer_deg": kick_deg,
                       "duration": settle_dur, "steer_active_hold": active_hold})
            self._sleep(self.steer_stuck_kick_hold_s)

            # ── Phase 2: return to target + motion — POLL for steer_ready ──
            # Exit immediately when steer fires (no need to wait full timeout).
            self.esp32_steer_ready = False
            self.esp32_steer_ready_seen = False
            self._cmd({"type": "drive", "gear": gear, "speed_mps": drv,
                       "steer_deg": target_deg,
                       "duration": settle_dur, "steer_active_hold": active_hold})
            t_end = time.monotonic() + self.steer_stuck_kick_hold_s
            while rclpy.ok() and time.monotonic() < t_end:
                if self.esp32_steer_ready:
                    self.get_logger().info(
                        f"  steer stuck kick {attempt + 1}: "
                        f"reached target while moving ✓")
                    return True
                time.sleep(0.02)

            # ── Phase 3: stop motion, hold at target, short wait ───────────
            # Use kick_timeout_s (e.g. 3 s) instead of steer_ready_timeout_s (10 s).
            # If kick + motion didn't do it in 3 s at rest, retry.
            self.esp32_steer_ready = False
            self.esp32_steer_ready_seen = False
            self._cmd({"type": "drive", "gear": gear, "speed_mps": 0.0,
                       "steer_deg": target_deg,
                       "duration": settle_dur, "steer_active_hold": active_hold})
            if self._wait_steer_ready(k_tout):
                self.get_logger().info(
                    f"  steer stuck kick {attempt + 1}: "
                    f"reached target at rest ✓")
                return True
            self.get_logger().warning(
                f"  steer stuck kick {attempt + 1}: "
                f"steer_ready not fired (waited {k_tout:.1f}s)")

        return False  # all retries exhausted

    # ── Motion command ────────────────────────────────────────────────────
    def _motion_to_cmd(self, motion) -> Dict[str, Any]:
        gear  = int(motion.get("gear", -1))
        steer = float(motion.get("steer_deg", 0.0))
        dist  = 0.0
        for k in ("dist_m","target_dist_m","distance_m","dist"):
            if k in motion:
                try: dist = abs(float(motion[k]))
                except: pass
                break
        gear  = 1 if gear>0 else (-1 if gear<0 else 0)
        steer = self.clamp(steer, -self.max_command_steer_deg, self.max_command_steer_deg)
        turning = abs(steer) > self.straight_steer_threshold_deg
        use_rear_us_flag = bool(motion.get("use_rear_us", False))
        if gear == 0:
            cmd_speed = 0.0
        elif use_rear_us_flag:
            # d4: very slow so rear_us sensor has time to react.
            # Tune d4_speed_mps in YAML (0.02 m/s = slow crawl into slot).
            cmd_speed = min(self.d4_speed_mps, self.speed_scale)
        else:
            cmd_speed = min(0.09 if turning else 0.08, self.speed_scale)
        if gear == 0 or dist <= 0.0:
            dt = self.drive_time_min_s
        elif turning:
            # Arc stop hierarchy:
            #   1. imu_arc_stop (primary) — fires at 79.2° when IMU is live
            #   2. time_stop   (fallback) — dt must be > actual arc time
            # IMU available → give full drive_time_max_s; imu stops it.
            # IMU silent     → use arc_fallback_time_s (12s default).
            #   reverse_turn_speed_mps=0.501 → 4.01s (too short; car does ~45°)
            #   actual arc speed ≈ 0.25 m/s → arc needs ~10s for 90°
            imu_fresh = (self.last_imu_time > 0
                         and (time.monotonic() - self.last_imu_time) < 2.0)
            if imu_fresh and self.imu_arc_stop_enabled:
                # IMU is live AND arc stop is enabled: give full budget, IMU controls stop.
                # If imu_arc_stop_enabled=false, fall through to time fallback.
                dt = self.drive_time_max_s
            else:
                dt = self.clamp(self.arc_fallback_time_s,
                                self.drive_time_min_s, self.drive_time_max_s)
                reason = "IMU arc stop disabled" if imu_fresh else "IMU not available"
                self.get_logger().warning(
                    f"  {reason} — arc fallback: {dt:.1f}s "
                    f"(tune arc_fallback_time_s in YAML)")
        else:
            # Straight moves (Move 1 fwd_setup, Move 3 rev_straight_d4):
            # Encoder bridge controls the stop via /enc_result — NO timer needed.
            # Wheel diameter = 7 inches → circumference = 0.5585 m.
            # The ESP32 sends rightDistM in metres; driveStraight() stops when
            #   fabs(rightDistM - startDist) >= target_m  (pure distance, no time).
            #
            # drive_time_max_s here is only a SAFETY timeout (120 s) in case the
            # encoder fails completely — normal operation stops via enc_result_stop.
            dt = self.drive_time_max_s
            self.get_logger().info(
                f"  Straight: encoder-only stop (timer={dt:.0f}s safety-only  "
                f"dist={dist:.3f}m  wheel=7in)")
        # steer_active_hold: True for straight (motor holds 0°), False for arc (locks at -/+30°)
        # steer_active_hold: True for straight (motor holds 0°), False for arc.
        # For straight moves MUST be True — the return spring pulls to +30°.
        # Without it: steer reaches 0°, motor locks off, spring pulls wheel to ~30°
        # and drive command arrives needing err ≤ 8° → drive never starts.
        if "steer_active_hold" in motion:
            steer_active_hold = bool(motion["steer_active_hold"])
        else:
            steer_active_hold = not turning  # True for straight, False for arc
        # Straight moves: add motor_start_delay_s to ESP32 duration.
        # ESP32 must keep motor running until after autopark_master's timer fires.
        # Without fix: ESP32 stops at (dt+0.7s), master stops at (delay+dt) → 0.5s short.
        if not turning:
            esp32_duration = dt + self.motor_start_delay_s
        else:
            esp32_duration = dt
        # use_encoder=True: tells encoder_bridge to intercept this reverse move even when
        # |steer_deg| exceeds straight_steer_thresh (5°). Required for rear-cam Move3
        # which can have steer_deg = -yaw_rear (potentially > 5°).
        use_encoder = bool(motion.get("use_encoder", False))
        return {"type":"drive","gear":gear,"speed_mps":cmd_speed,
                "steer_deg":steer,"target_dist_m":dist,"duration":esp32_duration,
                "master_duration": dt,          # pure travel time for _wait_stop
                "steer_active_hold": steer_active_hold,
                "use_encoder": use_encoder}

    # ── Sensors ───────────────────────────────────────────────────────────
    def get_ultrasonic_min_m(self) -> float:
        vals = []
        for v in self.latest_us:
            try: x=float(v)
            except: continue
            if x<=0: continue
            if x>3: x/=100
            if x>5: continue
            vals.append(x)
        return min(vals) if vals else 9.9

    def get_rear_us_m(self) -> float:
        vals = []
        for i in self.rear_us_indices:
            if i < len(self.latest_us):
                try: x=float(self.latest_us[i])
                except: continue
                if x<=0: continue
                if x>3: x/=100
                if x>5: continue
                vals.append(x)
        return min(vals) if vals else 9.9

    def _get_pose(self) -> Optional[Pose2D]:
        if self.latest_pose is not None:
            return self.latest_pose
        if not self.use_default_pose_when_missing:
            return None
        p = Pose2D()
        p.x=self.default_start_x; p.y=self.default_start_y
        p.theta=math.radians(self.default_start_yaw_deg)
        self.get_logger().warning(f"using default pose ({p.x},{p.y},{math.degrees(p.theta):.0f}°)")
        return p

    def _cmd(self, obj): self.cmd_pub.publish(String(data=json.dumps(obj)))
    def _stop(self, why): self._cmd({"type":"stop","reason":why}); self.get_logger().warning(f"STOP:{why}")
    def _sleep(self, s):
        e=time.monotonic()+max(0.0,float(s))
        while rclpy.ok() and time.monotonic()<e: time.sleep(0.04)

    @staticmethod
    def _adelta(a, b):
        d=float(b)-float(a)
        while d>180: d-=360
        while d<-180: d+=360
        return d

    @staticmethod
    def clamp(v, lo, hi): return max(lo, min(hi, float(v)))


def main(args=None):
    rclpy.init(args=args)
    node = AutoparkMaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._led("off")
        node._stop("shutdown")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
