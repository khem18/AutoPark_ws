"""
autopark_master.py  —  v_next10
=============================================================================
Changes vs v_next9:
  FIX: kinetic centering motor not reacting after 3-move planner sequence.

  Root cause: encoder_bridge.DriveSerial owns /dev/ttyUSB0 and sends the
  final drive_/stop() at the end of Move 3 (driveStraight). The real ESP32
  firmware requires an explicit {"type":"arm"} before it will accept new
  drive commands after a stop. Kinetic centering never sent arm, so both
  KINETIC-A (forward) and KINETIC-B (reverse) received drive commands that
  the ESP32 silently ignored.

  Fixes applied:
  1. Send {"type":"arm"} at the start of _run_kinetic_centering() and wait
     kinetic_rearm_wait_s (default 0.5 s) for ESP32 to arm.
  2. Subscribe to /enc_busy → track self.enc_busy_from_encoder; log its
     value at kinetic start so any enc_busy leak is immediately visible.
  3. New param kinetic_rearm_wait_s (default 0.5).
"""
import json
import math
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, Float32MultiArray
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Imu

from .planner_adapter import plan_from_start, result_to_dict


def _yaw_from_imu(msg: Imu) -> float:
    """Extract yaw (radians) from IMU quaternion."""
    q = msg.orientation
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class AutoparkMaster(Node):
    def __init__(self):
        super().__init__("autopark_master")

        for name, default in [
            # Topics
            ("start_switch_topic",           "/autopark/start_switch"),
            ("esp32_status_topic",           "/autopark/esp32_status"),
            ("pose_topic",                   "/autopark/start_pose"),
            ("parking_metrics_topic",        "/parking_metrics"),
            ("ultrasonic_topic",             "/autopark/ultrasonic"),
            ("command_topic",                "/autopark/cmd_json"),
            ("plan_topic",                   "/autopark/plan_result"),
            ("cam_check_request_topic",      "/autopark/cam_check_request"),
            ("imu_topic",                    "/imu/data_raw"),
            ("enc_result_topic",             "/enc_result"),
            ("enc_status_topic",             "/enc_status"),

            # Planner / pose
            ("min_clearance_m",              0.12),
            ("disable_ultrasonic_block",     True),
            ("planner_mode",                 "both_sides"),
            ("use_default_pose_when_missing", True),
            ("default_start_x",              0.0),
            ("default_start_y",              0.70),
            ("default_start_yaw_deg",        180.0),

            # ── Speed & timing ─────────────────────────────────────────────
            # speed_scale: sent as speed_mps in drive commands.
            #   For encoder-intercepted moves (1, 3) the encoder_bridge drives
            #   at enc_fwd/rev_speed internally — this value is informational.
            #   For the arc (Move 2, serial_bridge handles) this IS the drive speed.
            ("speed_scale",                  0.03),   # ← was 0.01
            ("default_motion_seconds",       120.0),  # safety timeout per segment
            ("pause_between_commands",       1.20),
            ("steer_wait_seconds",           4.0),

            # Camera check
            ("camera_check_enabled",         True),
            ("camera_pose_max_age_s",        3.0),

            # ── Wheel speed bias ────────────────────────────────────────────
            # Positive = right steer bias.
            # Applied on steer≈0 moves:  gear=+1 → +bias,  gear=-1 → -bias.
            # If left wheel is physically slower the car drifts left going fwd
            # and right going rev.  Set bias ≈ 2–5° and retune as needed.
            ("steer_straight_bias_deg",      0.0),
            # Steer magnitude below which the move is considered "straight":
            ("steer_bias_apply_thresh_deg",  2.0),

            # ── IMU arc stopping (Move 2) ───────────────────────────────────
            ("imu_arc_stop_enabled",         True),
            # Fraction of full arc angle at which to send STOP.
            ("imu_arc_stop_factor",          0.89),
            # When encoder_bridge boosts session_arc past threshold (passenger load),
            # use this smaller factor so the faster arc still stops in the slot.
            ("imu_arc_stop_factor_stuck",    0.86),
            ("arc_stuck_speed_threshold_mps", 0.06),
            # Seconds to wait after arc starts before checking IMU delta.
            # Must exceed motor spin-up (~0.5-1 s).
            ("imu_arc_wait_before_check_s",  1.0),
            # Arc reversal detection: if arc angle drops from peak, car reversed.
            ("imu_arc_reversal_enabled",     True),
            ("imu_arc_reversal_min_deg",     10.0),
            ("imu_arc_reversal_drop_deg",     3.0),
            # Hard timeout if IMU stop never triggers:
            ("imu_arc_stop_timeout_s",       120.0),
            # Time-based fallback when IMU unavailable.  0.0 = auto-calculate.
            ("arc_fallback_time_s",          0.0),
            # Encoder distance as primary arc-stop backup when IMU = 0.
            ("imu_arc_enc_backup_enabled",   True),

            # ── Encoder result wait ─────────────────────────────────────────
            # Timeout for waiting on /enc_result after a Move 1 or 3 command:
            ("enc_result_timeout_s",         130.0),

            # ── Confirmed physical sensor layout (hardware owner, 1-based labels) ──
            #   x1, x2, x3 = front   → 0-based indices 0, 1, 2
            #   x4         = left    → 0-based index 3
            #   x5         = right   → 0-based index 4
            #   x6, x7, x8 = rear    → 0-based indices 5, 6, 7

            # ── Forward clearance guard (applied before each gear=+1 move) ────
            # If any sensor in us_fwd_indices reads below us_fwd_min_m,
            # the forward move is skipped entirely (car is already at front wall).
            # FIX (per request — disabled entirely): switching the sensor
            # set from x1/x2/x3 to x4/x5 was applied here in the Python
            # defaults, but the YAML override (which wins at launch) never
            # got the same update and stayed at [0,1,2] — so the guard kept
            # firing on the same unreliable front sensors regardless of
            # this file. Rather than keep chasing index-sync issues between
            # two files, the guard is now disabled outright. Move 1 will no
            # longer be skipped due to any ultrasonic reading.
            ("us_fwd_guard_enable",          False),   # disabled — was True
            ("us_fwd_indices",               [3, 4]),  # unused while disabled
            ("us_fwd_min_m",                 0.15),    # unused while disabled

            # ── Ultrasonic centering ────────────────────────────────────────
            # FIX: us_left_idx/us_right_idx were 1, 2 (→ x2, x3) — those are
            # FRONT sensors, not left/right, and read 150-260cm (open space)
            # in every observed log. KINETIC-A's "if L > 2.0 or R > 2.0: skip"
            # fired on almost every run → lateral centering silently never
            # executed. Correct sensors per confirmed hardware layout above.
            ("us_center_enable",             True),
            ("us_left_idx",                  3),    # x4 = left  (was 1 → x2, front)
            ("us_right_idx",                 4),    # x5 = right (was 2 → x3, front)
            ("us_deadband_m",                0.010),   # ±1cm tolerance around the target offset
            # FIX (corrected per request): the goal is NOT symmetric
            # centering (x4=x5). Target is x4-x5 = 5cm, held to within
            # ±us_deadband_m (1cm) — i.e. L-R should land in [4cm, 6cm].
            ("us_lateral_target_diff_m",     0.050),   # x4 - x5 target = 5cm
            ("us_steer_gain",                150.0),
            # x4=left, x5=right is now confirmed, so the original sign
            # convention (steer0 = -correction) should be correct by
            # default. Kept as a tunable in case the physical steer/servo
            # direction itself is reversed on this chassis — flip to -1.0
            # if correction still drives toward the closer wall instead of
            # away from it.
            ("lateral_correction_sign",      1.0),
            ("us_max_steer_deg",             15.0),

            # FIX: kinetic operates in a confirmed-narrow slot — committing
            # to the full us_max_steer_deg for any error >=10cm (which the
            # old gain=150 did for nearly every routine correction) risks
            # swinging the body into an obstacle the lateral-centering
            # check itself doesn't model (it only checks the centering
            # error, not how much actual clearance is available on the
            # tight side). Two changes:
            #  1. A separate, gentler gain just for kinetic's live steer
            #     calculation, so typical errors produce a proportional
            #     angle instead of immediately saturating.
            #  2. A clearance-aware cap: the allowed steer angle scales
            #     down as the tighter side's measured clearance shrinks
            #     below us_kinetic_clearance_ref_m, so the car only uses
            #     a large turn angle when there's actually room for it —
            #     not just because the raw proportional error is large.
            ("us_kinetic_steer_gain",        60.0),   # gentler than us_steer_gain
            ("us_kinetic_max_steer_deg",     10.0),   # lower ceiling for tight slots
            ("us_kinetic_clearance_ref_m",   0.30),   # full steer allowed at/above this clearance
            ("us_correction_dist_m",         0.12),
            ("us_correction_speed_mps",      0.06),
            ("us_max_attempts",              3),
            ("us_rear_idx",                  6),
            ("us_rear_target_m",             0.20),
            ("us_rear_tolerance_m",          0.05),   # ±5cm (reverted — confirmed correct value)
            ("us_depth_max_step_m",          0.04),
            ("us_depth_max_attempts",        4),
            ("us_stop_buffer_s",             0.20),
            ("us_stop_settle_s",             0.60),
            ("us_steer_wait_s",              2.0),

            # ── Kinetic centering ───────────────────────────────────────────
            ("us_kinetic_enable",            True),
            ("us_kinetic_fwd_max_m",         0.12),
            ("us_kinetic_rev_max_m",         0.25),
            ("us_kinetic_update_hz",         10.0),
            ("us_kinetic_settle_s",          0.40),
            # FIX: arm the ESP32 before kinetic so it accepts drive commands
            # after encoder_bridge's DriveSerial sent stop() at end of Move 3.
            ("kinetic_rearm_wait_s",          0.5),
            # FIX: emergency abort — independent of the rear-target window.
            # Checked against ALL 8 sensors, every loop tick, in both Phase A
            # and Phase B. Catches overshoot from sensor-update lag or motor
            # coast during the gear-switch settle pause, regardless of which
            # sensor (not just us_rear_idx) is closing in on an obstacle.
            ("us_kinetic_min_safe_m",         0.05),   # 5 cm hard floor
            ("us_kinetic_max_data_age_s",     0.30),   # treat stale US data as unsafe

            # FIX: step-and-check instead of one continuous creep/one deep
            # move. Each phase now: read sensor → compute remaining error →
            # drive only a FRACTION of that error → full stop → settle →
            # take a fresh reading → repeat. This means the car is
            # stationary (and the sensor reading is trustworthy) at the
            # moment each decision is made, instead of trying to react
            # mid-motion off a reading that may already be stale by the
            # time the command lands. Naturally converges and self-corrects
            # for speed/distance calibration error (we've seen commanded
            # 1.4600m actually land at 1.4724m) since each step re-measures
            # from the real current position rather than trusting an
            # open-loop distance estimate.
            # SUPERSEDED by the continuous-drive + 1s-recalc architecture
            # below (us_kinetic_recalc_period_s) — these fraction-based
            # discrete-step params are no longer read by the drive logic,
            # left declared only so old yaml overrides don't error out.
            ("us_kinetic_step_fraction",      0.25),   # drive 25% of remaining error per step
            ("us_kinetic_min_step_m",         0.02),   # don't bother stepping smaller than 2cm
            ("us_kinetic_max_step_lat_m",     0.06),   # cap a single lateral step
            ("us_kinetic_max_step_depth_m",   0.10),   # cap a single depth step
            # FIX (per request): Phase A no longer exits early on a distance
            # budget — the only backstops left are this attempt count and
            # the safety floor. Raised from 10 to 20 so it actually has
            # enough cycles to converge instead of running out of attempts
            # almost immediately. Phase B is unaffected in practice — it
            # still exits via its own reverse/forward budget checks first
            # in normal operation.
            ("us_kinetic_max_attempts",       20),     # was 10
            # FIX (per request): re-run lateral centering if depth correction
            # drags it back out of tolerance, instead of just detecting and
            # reporting the failure. Caps how many A→B outer passes to try.
            ("us_kinetic_outer_attempts",     3),

            # FIX (per request): full stop after each confirmed 1s drive
            # interval, BEFORE checking the next fresh ultrasonic reading.
            # The 1s is counted from when the encoder confirms the car
            # actually started moving (not from when the drive command was
            # sent), so PWM ramp-up / static-friction startup delay doesn't
            # eat into the timed window.
            ("us_kinetic_recalc_period_s",    1.0),    # drive duration once movement confirmed
            ("us_kinetic_min_enc_move_m",     0.01),   # 1cm — confirms real movement happened
            # FIX (per request): if EITHER side's ultrasonic reading shows
            # zero change for this many forward cycles (despite the encoder
            # confirming the wheels did turn), that corner is blocked — car
            # is out of room on that side. KINETIC-A [3] in the log showed
            # ΔL=+2.52cm (real movement) but ΔR=+0.00cm — continuing
            # forward there pushes the car further out of the slot rather
            # than helping. Triggers a brief reverse immediately (limit=1,
            # not waiting for a second occurrence) instead of continuing
            # to push forward.
            ("us_kinetic_unchanged_limit",    1),   # was 2
            # FIX (per request): if still blocked after one reverse, keep
            # reversing immediately rather than cycling back through a
            # forward attempt first. Caps how many consecutive reverses to
            # try before giving up on unstick and returning to normal
            # forward correction (which will just re-trigger unstick again
            # if still blocked — this cap only bounds a single burst).
            ("us_kinetic_unstick_max_retries", 3),
            ("us_kinetic_move_start_timeout_s", 2.0),  # max wait for encoder to confirm movement began
        ]:
            self.declare_parameter(name, default)

        def gp(n):
            return self.get_parameter(n).value

        # ── param bindings ────────────────────────────────────────────
        self.min_clearance_m           = float(gp("min_clearance_m"))
        self.disable_ultrasonic_block  = bool(gp("disable_ultrasonic_block"))
        self.planner_mode              = str(gp("planner_mode"))
        self.use_default_pose_when_missing = bool(gp("use_default_pose_when_missing"))
        self.default_start_x           = float(gp("default_start_x"))
        self.default_start_y           = float(gp("default_start_y"))
        self.default_start_yaw_deg     = float(gp("default_start_yaw_deg"))
        self.speed_scale               = float(gp("speed_scale"))
        self.default_motion_seconds    = float(gp("default_motion_seconds"))
        self.pause_between_commands    = float(gp("pause_between_commands"))
        self.steer_wait_seconds        = float(gp("steer_wait_seconds"))

        self.camera_check_enabled      = bool(gp("camera_check_enabled"))
        self.camera_pose_max_age_s     = float(gp("camera_pose_max_age_s"))

        self.steer_straight_bias_deg   = float(gp("steer_straight_bias_deg"))
        self.steer_bias_apply_thresh   = float(gp("steer_bias_apply_thresh_deg"))

        self.imu_arc_stop_enabled      = bool(gp("imu_arc_stop_enabled"))
        self.imu_arc_stop_factor       = float(gp("imu_arc_stop_factor"))
        self.imu_arc_stop_factor_stuck = float(gp("imu_arc_stop_factor_stuck"))
        self.arc_stuck_speed_threshold = float(gp("arc_stuck_speed_threshold_mps"))
        self.imu_arc_wait_s            = float(gp("imu_arc_wait_before_check_s"))
        self.imu_arc_reversal_enabled  = bool(gp("imu_arc_reversal_enabled"))
        self.imu_arc_reversal_min_deg  = float(gp("imu_arc_reversal_min_deg"))
        self.imu_arc_reversal_drop_deg = float(gp("imu_arc_reversal_drop_deg"))
        self.imu_arc_stop_timeout_s    = float(gp("imu_arc_stop_timeout_s"))
        self.arc_fallback_time_s       = float(gp("arc_fallback_time_s"))
        self.imu_arc_enc_backup        = bool(gp("imu_arc_enc_backup_enabled"))

        self.enc_result_timeout_s      = float(gp("enc_result_timeout_s"))

        self.us_fwd_guard_enable       = bool(gp("us_fwd_guard_enable"))
        self.us_fwd_indices            = list(gp("us_fwd_indices"))
        self.us_fwd_min_m              = float(gp("us_fwd_min_m"))

        self.us_center_enable          = bool(gp("us_center_enable"))
        self.us_left_idx               = int(gp("us_left_idx"))
        self.us_right_idx              = int(gp("us_right_idx"))
        self.us_deadband_m             = float(gp("us_deadband_m"))
        self.us_lateral_target_diff_m  = float(gp("us_lateral_target_diff_m"))
        self.us_steer_gain             = float(gp("us_steer_gain"))
        self.lateral_correction_sign   = float(gp("lateral_correction_sign"))
        self.us_max_steer_deg          = float(gp("us_max_steer_deg"))
        self.us_kinetic_steer_gain     = float(gp("us_kinetic_steer_gain"))
        self.us_kinetic_max_steer_deg  = float(gp("us_kinetic_max_steer_deg"))
        self.us_kinetic_clearance_ref_m = float(gp("us_kinetic_clearance_ref_m"))
        self.us_correction_dist_m      = float(gp("us_correction_dist_m"))
        self.us_correction_speed_mps   = float(gp("us_correction_speed_mps"))
        self.us_max_attempts           = int(gp("us_max_attempts"))
        self.us_rear_idx               = int(gp("us_rear_idx"))
        self.us_rear_target_m          = float(gp("us_rear_target_m"))
        self.us_rear_tolerance_m       = float(gp("us_rear_tolerance_m"))
        self.us_depth_max_step_m       = float(gp("us_depth_max_step_m"))
        self.us_depth_max_attempts     = int(gp("us_depth_max_attempts"))
        self.us_stop_buffer_s          = float(gp("us_stop_buffer_s"))
        self.us_stop_settle_s          = float(gp("us_stop_settle_s"))
        self.us_steer_wait_s           = float(gp("us_steer_wait_s"))
        self.us_kinetic_enable         = bool(gp("us_kinetic_enable"))
        self.us_kinetic_fwd_max_m      = float(gp("us_kinetic_fwd_max_m"))
        self.us_kinetic_rev_max_m      = float(gp("us_kinetic_rev_max_m"))
        self.us_kinetic_update_hz      = float(gp("us_kinetic_update_hz"))
        self.us_kinetic_settle_s       = float(gp("us_kinetic_settle_s"))
        self.kinetic_rearm_wait_s      = float(gp("kinetic_rearm_wait_s"))
        self.us_kinetic_min_safe_m     = float(gp("us_kinetic_min_safe_m"))
        self.us_kinetic_max_data_age_s = float(gp("us_kinetic_max_data_age_s"))
        self.us_kinetic_step_fraction   = float(gp("us_kinetic_step_fraction"))
        self.us_kinetic_min_step_m      = float(gp("us_kinetic_min_step_m"))
        self.us_kinetic_max_step_lat_m  = float(gp("us_kinetic_max_step_lat_m"))
        self.us_kinetic_max_step_depth_m = float(gp("us_kinetic_max_step_depth_m"))
        self.us_kinetic_max_attempts    = int(gp("us_kinetic_max_attempts"))
        self.us_kinetic_outer_attempts  = int(gp("us_kinetic_outer_attempts"))
        self.us_kinetic_recalc_period_s = float(gp("us_kinetic_recalc_period_s"))
        self.us_kinetic_min_enc_move_m  = float(gp("us_kinetic_min_enc_move_m"))
        self.us_kinetic_unchanged_limit = int(gp("us_kinetic_unchanged_limit"))
        self.us_kinetic_unstick_max_retries = int(gp("us_kinetic_unstick_max_retries"))
        self.us_kinetic_move_start_timeout_s = float(gp("us_kinetic_move_start_timeout_s"))

        # ── state ─────────────────────────────────────────────────────
        self.latest_pose: Optional[Pose2D] = None
        self.latest_pose_time: Optional[float] = None
        self.latest_us         = [9.9] * 8
        self.latest_us_time    = 0.0   # FIX: monotonic time of last US update
        self.latest_metrics    = []
        self.busy              = False
        # FIX: edge-detect btn_state transitions to prevent auto-relaunch
        # race (see on_esp32_status for full explanation)
        self.last_btn_state_seen = "idle"

        # IMU state
        # latest_yaw_rad: from quaternion (often identity on mpu6050_cpp → unreliable)
        # latest_gyro_z:  angular_velocity.z in rad/s — always valid on MPU6050
        self.latest_gyro_z     = 0.0       # rad/s (latest raw rate)
        self.imu_yaw_deg       = 0.0       # continuously integrated heading (deg)
        self.last_imu_time     = 0.0       # monotonic time of last IMU message
        self.imu_received      = False     # True once first message arrives
        self._arc_peak         = 0.0       # peak arc angle for reversal detection

        # Encoder bridge state
        self.enc_result_event  = threading.Event()
        self.enc_result_data   = None
        self.latest_enc_rd     = 0.0   # rightDistM from /enc_status
        self.enc_busy_from_encoder = False  # FIX: track enc_busy for kinetic diagnostics

        # ── Subscriptions ─────────────────────────────────────────────
        self.create_subscription(
            String, gp("esp32_status_topic"), self.on_esp32_status, 10)
        self.create_subscription(
            Bool,   gp("cam_check_request_topic"), self.on_cam_check_request, 10)
        self.create_subscription(
            Pose2D, gp("pose_topic"), self.on_pose, 10)
        self.create_subscription(
            Float32MultiArray, gp("parking_metrics_topic"), self.on_metrics, 10)
        self.create_subscription(
            Float32MultiArray, gp("ultrasonic_topic"), self.on_ultrasonic, 10)
        self.create_subscription(
            Imu,    gp("imu_topic"), self.on_imu, 10)
        self.create_subscription(
            String, gp("enc_result_topic"), self.on_enc_result, 10)
        self.create_subscription(
            String, gp("enc_status_topic"), self.on_enc_status, 10)
        # FIX: track enc_busy so kinetic start can log/warn if it leaks True
        self.create_subscription(
            Bool, "/enc_busy", self.on_enc_busy, 10)

        self.cmd_pub  = self.create_publisher(
            String, gp("command_topic"), 10)
        self.plan_pub = self.create_publisher(
            String, gp("plan_topic"), 10)

        self.get_logger().info(
            f"autopark_master v_next9  speed_scale={self.speed_scale}  "
            f"bias={self.steer_straight_bias_deg:+.1f}°  "
            f"imu_arc_stop={self.imu_arc_stop_enabled}  "
            f"kinetic_center={self.us_kinetic_enable}")

    # ── Subscribers ───────────────────────────────────────────────────

    def on_pose(self, msg):
        self.latest_pose = msg
        self.latest_pose_time = time.monotonic()

    def on_metrics(self, msg):
        self.latest_metrics = list(msg.data)

    def on_ultrasonic(self, msg):
        vals = list(msg.data)
        if len(vals) >= 8:
            self.latest_us      = vals[:8]
            self.latest_us_time = time.monotonic()   # FIX: stamp for staleness check

    def on_imu(self, msg: Imu):
        # ROOT CAUSE FIX: pre-integrate at full IMU rate (200 Hz) instead of
        # storing a snapshot.  The old code stored only latest_gyro_z and let
        # _wait_arc_imu re-integrate at 20 Hz (50 ms sleep), losing 90% of
        # samples — making the arc-angle estimate wildly wrong.
        #
        # This mirrors the working version (v_next4): every incoming message
        # immediately updates imu_yaw_deg.  _wait_arc_imu reads the already-
        # integrated value via _adelta(imu_start, self.imu_yaw_deg).
        now = time.monotonic()
        self.imu_received   = True
        if self.last_imu_time <= 0.0:
            self.last_imu_time = now
            return
        dt = now - self.last_imu_time
        self.last_imu_time  = now
        if 0.0 < dt < 0.2:          # guard against timer jumps
            wz = float(msg.angular_velocity.z)
            self.latest_gyro_z = wz
            self.imu_yaw_deg  += math.degrees(wz * dt)

    def on_enc_result(self, msg: String):
        self.enc_result_data = msg.data
        self.enc_result_event.set()

    def on_enc_status(self, msg: String):
        try:
            obj = json.loads(msg.data)
            self.latest_enc_rd = float(obj.get("rd", self.latest_enc_rd))
        except Exception:
            pass

    def on_enc_busy(self, msg: Bool):
        self.enc_busy_from_encoder = bool(msg.data)

    def on_cam_check_request(self, msg: Bool):
        if not msg.data:
            return
        if not self.camera_check_enabled:
            self.get_logger().info("CAM CHECK: disabled → LED yellow (always ready)")
            self._led("yellow")
            return
        pose_fresh = self._is_pose_fresh()
        if pose_fresh:
            self.get_logger().info(
                "CAM CHECK: pose fresh (age=%.2fs) → LED yellow" %
                (time.monotonic() - self.latest_pose_time))
            self._led("yellow")
        else:
            age_str = (
                "%.2fs" % (time.monotonic() - self.latest_pose_time)
                if self.latest_pose_time else "never"
            )
            self.get_logger().warning(
                f"CAM CHECK: pose NOT fresh (age={age_str}) → LED red")
            self._led("red")

    def _is_pose_fresh(self) -> bool:
        if self.latest_pose is None or self.latest_pose_time is None:
            return False
        return (time.monotonic() - self.latest_pose_time) <= self.camera_pose_max_age_s

    def on_esp32_status(self, msg: String):
        try:
            obj = json.loads(msg.data)
        except Exception:
            return
        btn_state = obj.get("btn_state", "")

        # FIX: auto-relaunch race. self.busy resets to False the instant
        # start_autopark() RETURNS — which happens right after sending the
        # final green-LED + parking_complete stop, with no wait. But the
        # ESP32 only resets its OWN btn_state back to "idle" ~1s later
        # (after its green-flash timer expires and _greenPendingDisarm
        # fires). For that ~1s gap the ESP32 keeps broadcasting a STALE
        # btn_state="parking", and the very next status message after
        # self.busy flips back to False was re-triggering a brand new
        # round with no button press — confirmed by logs showing
        # "STOP: parking_complete" and "launching parking thread" only
        # 17ms apart.
        #
        # Fix: only launch on the ACTUAL idle→parking edge, not just
        # "currently reporting parking". Stale repeated "parking" messages
        # from the tail of the previous round are now ignored because
        # self.last_btn_state_seen is already "parking" from earlier in
        # that same round.
        is_new_parking_edge = (
            btn_state == "parking" and self.last_btn_state_seen != "parking")
        self.last_btn_state_seen = btn_state

        if is_new_parking_edge and not self.busy:
            self.busy = True
            self.get_logger().info("btn_state=parking → launching parking thread")
            threading.Thread(target=self._parking_thread, daemon=True).start()

    def _parking_thread(self):
        try:
            self.start_autopark()
        except Exception as exc:
            self.get_logger().error(f"parking thread: {exc}")
            self._led("red")
            self.publish_stop("parking_exception")
        finally:
            self.busy = False

    # ─────────────────────────────────────────────────────────────────
    # Main parking sequence
    # ─────────────────────────────────────────────────────────────────

    def start_autopark(self):
        self.get_logger().info("START_AUTOPARK ENTERED")

        if not self.disable_ultrasonic_block:
            if self.latest_us and min(self.latest_us) < self.min_clearance_m:
                self.publish_stop("blocked_by_ultrasonic_before_start")
                return

        pose = self.latest_pose
        if pose is None:
            if not self.use_default_pose_when_missing:
                self.publish_stop("no_start_pose")
                return
            pose = Pose2D()
            pose.x     = self.default_start_x
            pose.y     = self.default_start_y
            pose.theta = math.radians(self.default_start_yaw_deg)
            self.get_logger().warning(
                "using default pose: " + str((pose.x, pose.y, pose.theta)))

        yaw_deg = math.degrees(pose.theta)
        self.get_logger().info(
            "AUTOPARK START")
        self.get_logger().info(
            "  Planning: x=%.3f y=%.3f yaw=%.1f° case=%s"
            % (pose.x, pose.y, yaw_deg, self.planner_mode))

        planned = plan_from_start(pose.x, pose.y, yaw_deg, self.planner_mode)
        result  = result_to_dict(planned)
        motions = result.get("motions", [])
        metrics = result.get("metrics", {})

        if not motions:
            self.plan_pub.publish(String(data=json.dumps(result)))
            self._led("red")
            self.publish_stop("planner_returned_empty_motion_sequence")
            return

        self.get_logger().info(
            "  Plan OK  d1={d1_m:.3f}  s23={s23_m:.3f}  d4={d4_m:.3f}".format(**metrics))

        if self.latest_metrics:
            result["parking_metrics"] = self.latest_metrics
        self.plan_pub.publish(String(data=json.dumps(result)))

        # ── Step 1: 3-move planner sequence ──────────────────────────
        self.execute_motions(motions, metrics)

        # ── Step 2: kinetic ultrasonic centering ──────────────────────
        kinetic_ok = True
        if self.us_center_enable:
            if self.us_kinetic_enable:
                kinetic_ok = self._run_kinetic_centering()
            else:
                self._run_lateral_centering()
                self._run_depth_correction()

        # FIX: the final LED/status used to always go green regardless of
        # whether kinetic centering actually landed inside its tolerance
        # window. A car parked 2.7cm from the rear wall (well outside the
        # 14-26cm target) was being reported identically to a clean park —
        # the only way to find out was to read the raw sensor dump after
        # the fact. Now a failed/out-of-tolerance finish is surfaced.
        if kinetic_ok:
            self.get_logger().info("PARKING DONE → Green LED 1s")
            self._led("green")
            self.publish_stop("parking_complete")
        else:
            self.get_logger().error(
                "PARKING DONE WITH WARNING — kinetic centering finished "
                "outside tolerance → Yellow LED 1s")
            self._led("yellow")
            self.publish_stop("parking_complete_out_of_tolerance")

    # ─────────────────────────────────────────────────────────────────
    # execute_motions  —  3-move encoder/IMU guided sequence
    # ─────────────────────────────────────────────────────────────────
    #
    #  Move 1  fwd_setup   gear=+1  any steer
    #    → encoder_bridge FORWARD MONITOR intercepts (gear>0, dist_ok)
    #    → waits for /enc_result SUCCESS
    #
    #  Move 2  rev_arc_90  gear=-1  steer=+30°
    #    → NOT intercepted by encoder (steer>thresh, no enc_force)
    #    → encoder_bridge starts arc_monitor for stuck detection
    #    → autopark_master monitors IMU yaw, sends STOP at arc target × factor
    #
    #  Move 3  rev_straight_d4  gear=-1  steer≈0°  use_encoder=True
    #    → encoder_bridge REVERSE DRIVE intercepts (enc_force=True)
    #    → waits for /enc_result SUCCESS
    # ─────────────────────────────────────────────────────────────────

    def execute_motions(self, motions, metrics: dict = None):
        metrics = metrics or {}
        n = len(motions)
        self.get_logger().info(f"EXECUTE {n} moves")

        R_m  = float(metrics.get("R_m",  1.335))
        s23  = float(metrics.get("s23_m", 0.0))

        for i, motion in enumerate(motions):
            label     = motion.get("label", f"move_{i+1}")
            gear      = int(motion.get("gear", 0))
            steer_raw = float(motion.get("steer_deg", 0.0))
            dist_m    = abs(float(motion.get("dist_m", 0.0)))
            act_hold  = bool(motion.get("steer_active_hold", False))
            use_enc   = bool(motion.get("use_encoder", False))

            # ── Determine move type ───────────────────────────────────
            is_straight = abs(steer_raw) <= self.steer_bias_apply_thresh
            is_arc      = (gear < 0) and not is_straight

            # ── Apply wheel-bias on straight moves ────────────────────
            if is_straight:
                steer_eff = steer_raw + (
                    +self.steer_straight_bias_deg if gear > 0
                    else -self.steer_straight_bias_deg
                )
            else:
                steer_eff = steer_raw

            dur_safety = float(self.default_motion_seconds)

            # ── Log MOVE X/N ──────────────────────────────────────────
            self.get_logger().info(
                f"MOVE {i+1}/{n} [{label}]  "
                f"gear={'+' if gear > 0 else ''}{gear}  "
                f"steer={steer_raw:+.0f}°  dist={dist_m:.3f}m  t={dur_safety:.2f}s")
            if abs(self.steer_straight_bias_deg) > 0.01 and is_straight:
                self.get_logger().info(
                    f"  steer_active_hold={act_hold}  "
                    f"(straight={is_straight} keeps motor {'on' if act_hold else 'off'})")

            # ── Forward clearance guard ───────────────────────────────
            # Skip this move entirely if a front sensor is too close.
            # Prevents stalling against the front wall during fwd_setup.
            if gear > 0 and self.us_fwd_guard_enable and self.latest_us:
                fwd_readings = [self.latest_us[idx]
                                for idx in self.us_fwd_indices
                                if idx < len(self.latest_us)]
                if fwd_readings:
                    fwd_min = min(fwd_readings)
                    if fwd_min < self.us_fwd_min_m:
                        self.get_logger().warning(
                            f"FWD GUARD: skip MOVE {i+1}/{n} [{label}] — "
                            f"front sensor {fwd_min*100:.1f}cm < "
                            f"threshold {self.us_fwd_min_m*100:.0f}cm")
                        continue
                    else:
                        self.get_logger().info(
                            f"FWD GUARD: OK {fwd_min*100:.1f}cm >= "
                            f"{self.us_fwd_min_m*100:.0f}cm — proceeding")

            # ── Pre-steer (wheels stopped) ────────────────────────────
            steer_cmd = {
                "type": "drive", "gear": 0,
                "speed_mps": 0.0, "steer_deg": steer_eff,
            }
            # Always hold steer during pre-steer (arc needs servo locked at 30°)
            steer_cmd["steer_active_hold"] = True
            self.cmd_pub.publish(String(data=json.dumps(steer_cmd)))
            self.get_logger().info(
                f"WAIT STEER: steer_deg={steer_eff:.1f} wait={self.steer_wait_seconds}s")
            time.sleep(self.steer_wait_seconds)

            # ── Build drive command (dist_m always forwarded) ─────────
            drive_cmd = {
                "type":      "drive",
                "gear":      gear,
                "speed_mps": self.speed_scale,
                "steer_deg": steer_eff,
                "duration":  dur_safety,
                "dist_m":    dist_m,      # ← enables encoder_bridge interception
            }
            # Arc: force steer_active_hold=True regardless of planner setting.
            # Without it, the servo spring returns to center mid-arc and the car stalls.
            if act_hold or is_arc:
                drive_cmd["steer_active_hold"] = True
            if use_enc:
                drive_cmd["use_encoder"] = True   # forces encoder intercept even for curved rev

            # Clear enc_result event before sending drive
            self.enc_result_event.clear()
            self.enc_result_data = None
            rd_before = self.latest_enc_rd

            # ── Send drive command ────────────────────────────────────
            self.cmd_pub.publish(String(data=json.dumps(drive_cmd)))

            # ── Wait for segment completion ───────────────────────────
            if is_arc:
                self._wait_arc_imu(s23, R_m, i, n, rd_before)
            else:
                self._wait_enc_result(dist_m, i, n, rd_before)

            # ── Segment done ──────────────────────────────────────────
            self.publish_stop(f"segment_pause")
            time.sleep(self.pause_between_commands)

        self.publish_stop("parking_sequence_done")
        self.get_logger().info("PARKING SEQUENCE DONE")

    # ── Wait helpers ──────────────────────────────────────────────────

    def _wait_enc_result(self, dist_m: float, seg_idx: int, n: int, rd_before: float):
        """
        Wait for /enc_result from encoder_bridge.
        Logs DRIVE progress every 500 ms using /enc_status rightDistM.
        Falls back to time-based if enc_result never arrives.
        """
        timeout  = self.enc_result_timeout_s
        t0       = time.monotonic()
        last_log = 0.0

        self.get_logger().info(
            f"DRIVE 0.00/{dist_m:.2f}m")

        while True:
            signalled = self.enc_result_event.wait(timeout=0.10)

            elapsed  = time.monotonic() - t0
            traveled = abs(self.latest_enc_rd - rd_before)

            # Progress every 500 ms
            if elapsed - last_log >= 0.5:
                self.get_logger().info(
                    f"DRIVE {traveled:.2f}/{dist_m:.2f}m")
                last_log = elapsed

            if signalled:
                raw = self.enc_result_data or ""
                try:
                    obj      = json.loads(raw)
                    status   = obj.get("enc_result", "?")
                    act_dist = float(obj.get("dist",   traveled))
                    act_tgt  = float(obj.get("target", dist_m))
                except Exception:
                    status, act_dist, act_tgt = "?", traveled, dist_m

                self.get_logger().info(
                    f"ENC_RESULT received: {raw}")
                self.get_logger().info(
                    f"LOG: stop=enc_result_stop_seg_{seg_idx + 1}  "
                    f"elapsed={elapsed:.2f}s  "
                    f"dist={act_dist:.4f}m  target={act_tgt:.4f}m")
                return

            if elapsed >= timeout:
                self.get_logger().warning(
                    f"enc_result TIMEOUT after {elapsed:.1f}s — continuing")
                return

    def _wait_arc_imu(self, s23_m, R_m, seg_idx, n, rd_before: float = 0.0):
        """
        Wait for arc (Move 2) to reach its target angle, then return so the
        caller can send STOP.

        Stop hierarchy (checked every 40 ms):
          1. IMU yaw  — primary.  imu_yaw_deg is pre-integrated at 200 Hz by
                        on_imu(); read via _adelta(imu_start, self.imu_yaw_deg).
          2. Encoder  — backup when IMU = 0.  encoder_bridge tracks right-wheel
                        distance; arc_stop when traveled >= s23 * stop_factor.
          3. Reversal — if arc angle drops from its peak, car reversed; stop early.
          4. Time     — final fallback if both IMU and encoder are unavailable.
        """
        arc_rad      = s23_m / max(R_m, 0.001)
        arc_deg_full = math.degrees(arc_rad)
        timeout      = self.imu_arc_stop_timeout_s

        # ── Stop-factor: normal vs. stuck/boosted arc speed ───────────────
        # When encoder_bridge boosts session_arc (passenger load), the car
        # moves faster → it needs to stop earlier to land in the slot.
        arc_trig_deg = arc_deg_full * self.imu_arc_stop_factor
        enc_stop_dist = s23_m * self.imu_arc_stop_factor

        # ── Time-based fallback duration ──────────────────────────────────
        # FIX: old code used s23/speed * 1.5 = ~70 s (full-arc time, no factor).
        #      Correct: s23 * factor / speed * 1.2 = ~28 s.
        if self.arc_fallback_time_s > 0:
            fallback_total = self.arc_fallback_time_s
        else:
            fallback_total = (
                s23_m * self.imu_arc_stop_factor
                / max(self.speed_scale, 0.001) * 1.2)

        # ── IMU freshness ─────────────────────────────────────────────────
        imu_fresh = (self.last_imu_time > 0
                     and (time.monotonic() - self.last_imu_time) < 2.0)

        self.get_logger().info(
            f"[IMU arc] arc={arc_deg_full:.1f}°  "
            f"trig={arc_trig_deg:.1f}°  factor={self.imu_arc_stop_factor}  "
            f"imu_fresh={imu_fresh}  fallback={fallback_total:.1f}s  "
            f"enc_backup={'ON' if self.imu_arc_enc_backup else 'OFF'}")

        # ── IMU disabled or not ready → immediate time fallback ───────────
        if not self.imu_arc_stop_enabled or not imu_fresh:
            reason = ("disabled" if not self.imu_arc_stop_enabled
                      else "IMU not ready/fresh")
            self.get_logger().warning(
                f"[IMU arc] {reason} — time-based fallback ({fallback_total:.1f}s)")
            time.sleep(min(fallback_total, timeout))
            return

        # ── Live monitoring loop ──────────────────────────────────────────
        imu_start  = self.imu_yaw_deg   # snapshot heading at arc start
        self._arc_peak = 0.0            # reset reversal tracker
        t0         = time.monotonic()
        last_log   = 0.0

        # Gyro-zero sustained tracking (backup if IMU stops mid-arc).
        # FIXED: old code checked abs(latest_gyro_z) < math.radians(0.5)/0.04
        # which equals 12.5 deg/s — fires even when car rotates at 4-5 deg/s!
        # New approach: track how much imu_yaw_deg actually changed over
        # GYRO_ZERO_SUSTAINED_S seconds.  If total yaw change < 1.0 deg in
        # 2 s the IMU is truly frozen; otherwise it is working fine.
        GYRO_ZERO_SUSTAINED_S    = 2.0
        gyro_zero_start          = None   # time when zero-streak began
        gyro_zero_yaw_at_start   = 0.0    # imu_yaw_deg snapshot when streak began
        time_based_deadline      = None

        # Stuck-speed arc factor (apply once per arc)
        arc_stuck_factor_applied = False

        while True:
            time.sleep(0.04)
            now     = time.monotonic()
            elapsed = now - t0

            # ── 1. Stuck-speed: switch to smaller factor if arc is boosted ─
            if (not arc_stuck_factor_applied
                    and self.imu_arc_stop_factor_stuck < self.imu_arc_stop_factor
                    and False):  # stub: arc_stuck speed check (requires encoder_bridge session_arc topic)
                old_trig     = arc_trig_deg
                arc_trig_deg = arc_deg_full * self.imu_arc_stop_factor_stuck
                enc_stop_dist = s23_m * self.imu_arc_stop_factor_stuck
                arc_stuck_factor_applied = True
                self.get_logger().info(
                    f"[IMU arc] stuck-boost detected "
                    f"(sess_arc={self.session_arc_rev_speed_:.3f} m/s) → "
                    f"trig {old_trig:.1f}° → {arc_trig_deg:.1f}°")

            # ── 2. IMU yaw arc stop (primary) ──────────────────────────────
            if elapsed >= self.imu_arc_wait_s:
                cur_arc = abs(self._adelta(imu_start, self.imu_yaw_deg))

                if cur_arc >= arc_trig_deg:
                    self.get_logger().info(
                        f"LOG: stop=imu_arc_stop_seg_{seg_idx + 1}  "
                        f"elapsed={elapsed:.2f}s  imu_arc={cur_arc:.2f}°")
                    return

                # Reversal detection
                if self.imu_arc_reversal_enabled:
                    if cur_arc > self._arc_peak:
                        self._arc_peak = cur_arc
                    elif (self._arc_peak >= self.imu_arc_reversal_min_deg
                            and (self._arc_peak - cur_arc) >= self.imu_arc_reversal_drop_deg):
                        self.get_logger().warning(
                            f"[IMU arc] reversal: peak={self._arc_peak:.1f}° "
                            f"cur={cur_arc:.1f}° → stop early")
                        return

            # ── 3. Encoder arc stop (backup when IMU = 0) ──────────────────
            if self.imu_arc_enc_backup:
                enc_traveled = abs(self.latest_enc_rd - rd_before)
                if enc_traveled >= enc_stop_dist:
                    self.get_logger().info(
                        f"LOG: stop=enc_arc_stop_seg_{seg_idx + 1}  "
                        f"elapsed={elapsed:.2f}s  "
                        f"enc={enc_traveled:.4f}/{enc_stop_dist:.4f}m")
                    return

            # ── 4. Gyro-zero sustained fallback ────────────────────────────
            if elapsed >= self.imu_arc_wait_s and time_based_deadline is None:
                # Track yaw change since the start of this potential zero-streak.
                # Fires fallback only if < 1.0 deg accumulated over 2 consecutive s.
                if gyro_zero_start is None:
                    gyro_zero_start        = now
                    gyro_zero_yaw_at_start = self.imu_yaw_deg
                else:
                    yaw_moved = abs(self._adelta(gyro_zero_yaw_at_start,
                                                 self.imu_yaw_deg))
                    if yaw_moved >= 1.0:
                        # IMU is accumulating — reset streak
                        if (now - gyro_zero_start) >= 0.5:
                            self.get_logger().info(
                                f"[IMU arc] IMU active "
                                f"(+{yaw_moved:.1f}° in "
                                f"{now - gyro_zero_start:.1f}s) — "
                                f"gyro-zero streak reset")
                        gyro_zero_start        = now
                        gyro_zero_yaw_at_start = self.imu_yaw_deg
                    elif (now - gyro_zero_start) >= GYRO_ZERO_SUSTAINED_S:
                        # < 1° yaw change in 2 s → IMU truly frozen
                        remaining = max(0.1, fallback_total - elapsed)
                        time_based_deadline = now + remaining
                        self.get_logger().warning(
                            f"[IMU arc] yaw moved only {yaw_moved:.2f}° in "
                            f"{now - gyro_zero_start:.1f}s after {elapsed:.1f}s — "
                            f"IMU frozen, time-based deadline in {remaining:.1f}s")

            # ── 5. Time-based deadline ──────────────────────────────────────
            if time_based_deadline is not None and now >= time_based_deadline:
                self.get_logger().warning(
                    f"[IMU arc] time-based stop at elapsed={elapsed:.1f}s")
                return

            # ── 6. Hard timeout ─────────────────────────────────────────────
            if elapsed >= timeout:
                self.get_logger().warning(
                    f"[IMU arc] hard timeout after {elapsed:.1f}s")
                return

            # ── 7. Progress log ─────────────────────────────────────────────
            if elapsed - last_log >= 0.5:
                cur_arc = abs(self._adelta(imu_start, self.imu_yaw_deg))
                enc_t = abs(self.latest_enc_rd - rd_before) if self.imu_arc_enc_backup else 0.0
                self.get_logger().info(
                    f"DRIVE {elapsed:.2f}/{timeout:.2f}s  "
                    f"imu={cur_arc:.1f}/{arc_trig_deg:.1f}°  "
                    f"enc={enc_t:.4f}/{enc_stop_dist:.4f}m")
                last_log = elapsed

    # ── FIX: kinetic emergency floor check ───────────────────────────────
    # Phase B (depth/reverse correction) was observed driving the car to
    # within ~2.7cm of the rear wall — well past the 14-26cm target window
    # — because the window check only watches us_rear_idx and only fires
    # the gear-flip once per 100ms loop tick.
    #
    # This is a pure collision-distance check, not a data-freshness check —
    # kept unconditionally regardless of recalculation cadence, since it's
    # about imminent physical danger, not about how old the reading is.
    #
    # FIX (per request): the standalone "ultrasonic data older than 0.3s"
    # abort has been removed. The architecture now recalculates from a
    # fresh ultrasonic reading every us_kinetic_recalc_period_s, gated on
    # confirmed encoder movement (not just elapsed wall-clock time) — see
    # _kinetic_lateral_steps / _kinetic_depth_steps. That recalculation
    # cadence is what keeps the data "fresh enough" now; a separate
    # data-age abort on top of it is redundant by design.
    #
    # FIX (per request): x1/x2/x3 (front) excluded entirely from this
    # check. Observed reading 4.66cm with nothing actually in front of the
    # car — unreliable/spurious on this hardware setup — and that false
    # reading incorrectly blocked a reverse move that would only have
    # IMPROVED the (nonexistent) front clearance issue. Floor check is now
    # restricted to the 3 trusted sensors: x4 (left, us_left_idx), x5
    # (right, us_right_idx), x7 (rear, us_rear_idx).
    #
    # Direction still matters within those 3: left/right are checked
    # regardless of gear (a side collision risk exists driving either
    # direction), but the rear sensor is only checked when actually
    # reversing (gear=-1) — a close rear reading shouldn't block driving
    # forward away from it. gear=None (direction not yet decided, e.g. the
    # very first per-cycle entry check) checks all 3 conservatively.
    def _kinetic_floor_ok(self, phase: str, gear: int = None) -> bool:
        us = self.latest_us
        if not us:
            return True

        check_idx = [self.us_left_idx, self.us_right_idx]
        if gear is None or gear < 0:
            check_idx.append(self.us_rear_idx)

        relevant = [(i, us[i]) for i in check_idx if i < len(us)]
        if not relevant:
            return True
        idx, min_val = min(relevant, key=lambda pair: pair[1])
        if min_val < self.us_kinetic_min_safe_m:
            self.get_logger().error(
                f"KINETIC SAFETY ABORT [{phase}]: sensor idx={idx} "
                f"= {min_val*100:.1f}cm < safety floor "
                f"{self.us_kinetic_min_safe_m*100:.1f}cm — STOPPING")
            self.publish_stop(f"kinetic_safety_abort_{phase}")
            return False

        return True

    # FIX: clearance-aware steer cap. The old code clamped the lateral
    # correction at a single fixed us_max_steer_deg regardless of how much
    # actual side clearance existed — meaning any error >=10cm committed to
    # the full turn angle even in a slot barely wider than the car. This
    # scales the allowed steer angle down as the TIGHTER side's measured
    # clearance shrinks below us_kinetic_clearance_ref_m, so a large turn
    # is only used when there's actually room for it.
    def _kinetic_clearance_capped_steer(self, err: float, L: float, R: float) -> float:
        raw = err * self.us_kinetic_steer_gain
        tight_side = min(L, R)
        clearance_scale = max(0.0, min(1.0, tight_side / self.us_kinetic_clearance_ref_m))
        effective_max = self.us_kinetic_max_steer_deg * clearance_scale
        correction = max(-effective_max, min(effective_max, raw))
        if abs(raw) > effective_max:
            self.get_logger().info(
                f"KINETIC-A: steer capped by clearance — tight_side={tight_side*100:.1f}cm "
                f"scale={clearance_scale:.2f} effective_max={effective_max:.1f}° "
                f"(raw correction would have been {raw:+.1f}°)")
        return -correction * self.lateral_correction_sign

    def _kinetic_lateral_steps(self) -> bool:
        """Lateral centering: drive → wait for encoder-confirmed movement
        start → drive exactly us_kinetic_recalc_period_s more from that
        confirmed point → full stop → settle → take a fresh ultrasonic
        reading → recompute → repeat until centred.

        FIX (per request): the car now fully stops after every timed drive
        interval, before the next ultrasonic check — not a continuous
        creep. The 1s window is timed from when the ENCODER confirms the
        car actually started moving, not from when the drive command was
        sent, so PWM ramp-up / static-friction startup delay doesn't eat
        into the timed driving window."""
        SPEED      = self.us_correction_speed_mps
        PERIOD     = self.us_kinetic_recalc_period_s
        MIN_ENC    = self.us_kinetic_min_enc_move_m
        START_TMO  = self.us_kinetic_move_start_timeout_s
        SETTLE     = self.us_kinetic_settle_s
        MAX_FWD    = self.us_kinetic_fwd_max_m
        total_fwd_est = 0.0
        prev_L, prev_R = None, None   # for explicit diff logging
        unchanged_streak = 0

        for attempt in range(self.us_kinetic_max_attempts):
            if not self._kinetic_floor_ok("A", gear=1):
                return False

            L   = self.latest_us[self.us_left_idx]
            R   = self.latest_us[self.us_right_idx]
            err = (L - R) / 2.0

            if prev_L is not None:
                d_l = (L - prev_L) * 100
                d_r = (R - prev_R) * 100
                # FIX (per request): EITHER side reading zero change counts
                # now, not just when BOTH do. Cycle 3 in the log had
                # ΔL=+2.52cm (real movement) but ΔR=+0.00cm — that side was
                # already blocked even though the car as a whole was still
                # moving (pivoting around the blocked corner). Previously
                # this required BOTH to be zero, so a one-sided block like
                # this was missed entirely and the car kept driving forward.
                is_unchanged = abs(d_l) < 0.01 or abs(d_r) < 0.01
                unchanged_streak = unchanged_streak + 1 if is_unchanged else 0
                unchanged = " [ONE SIDE UNCHANGED — that corner may be blocked]" \
                    if is_unchanged else ""
                self.get_logger().info(
                    f"KINETIC-A [{attempt+1}]: ΔL={d_l:+.2f}cm ΔR={d_r:+.2f}cm "
                    f"vs previous cycle{unchanged}")

            self.get_logger().info(
                f"KINETIC-A [{attempt+1}/{self.us_kinetic_max_attempts}]: "
                f"L={L:.3f}m R={R:.3f}m err={err*100:.1f}cm "
                f"deadband={self.us_deadband_m*100:.0f}cm")

            if L > 2.0 or R > 2.0:
                self.get_logger().warning("KINETIC-A: lateral sensors OOB — stop")
                return True   # nothing trustworthy to correct — treat as pass

            if abs(err) < self.us_deadband_m:
                self.get_logger().info(
                    f"KINETIC-A: centred after {attempt} cycle(s)")
                return True

            # FIX (per request): repeated ΔL=ΔR=0 while still commanding
            # forward (and the encoder confirmed wheel movement each time —
            # this isn't the "wheel may be stuck" no-movement case) means
            # the car has driven as far forward as the slot allows and is
            # now pressed against something ahead — wheels turning, body
            # not translating. Pushing forward again is pointless; back off
            # with a brief reverse instead, then retry from a fresh reading.
            if unchanged_streak >= self.us_kinetic_unchanged_limit:
                side_note = ""
                if prev_L is not None:
                    blocked_sides = []
                    if abs(d_l) < 0.01:
                        blocked_sides.append("left")
                    if abs(d_r) < 0.01:
                        blocked_sides.append("right")
                    if blocked_sides:
                        side_note = f" ({'/'.join(blocked_sides)} side blocked)"
                cycles_word = "cycle" if unchanged_streak == 1 else "cycles"
                self.get_logger().warning(
                    f"KINETIC-A [{attempt+1}]: {unchanged_streak} {cycles_word} "
                    f"with zero sensor change despite confirmed movement"
                    f"{side_note} — car is out of forward room. Reversing "
                    f"briefly to clear it.")

                # FIX (per request): keep reversing immediately as long as
                # the blockage persists, instead of doing one reverse then
                # routing back through a full forward-attempt cycle before
                # noticing it's still stuck. Each retry compares against
                # the reading from BEFORE that specific reverse — cleared
                # only once both sides show real movement again.
                pre_unstick_L, pre_unstick_R = L, R
                for unstick_try in range(self.us_kinetic_unstick_max_retries):
                    self.cmd_pub.publish(String(data=json.dumps({
                        "type": "drive", "gear": 0, "speed_mps": 0.0, "steer_deg": 0.0})))
                    time.sleep(self.us_steer_wait_s)
                    if not self._kinetic_floor_ok("A_unstick_pre", gear=-1):
                        return False
                    rd_before = self.latest_enc_rd
                    self.cmd_pub.publish(String(data=json.dumps({
                        "type": "drive", "gear": -1, "speed_mps": SPEED,
                        "steer_deg": 0.0,
                        "duration": round(START_TMO + PERIOD + 1.0, 2)})))
                    t_cmd = time.monotonic()
                    move_start = None
                    while time.monotonic() - t_cmd < START_TMO:
                        if not self._kinetic_floor_ok("A_unstick_waiting", gear=-1):
                            return False
                        if abs(self.latest_enc_rd - rd_before) >= MIN_ENC:
                            move_start = time.monotonic()
                            break
                        time.sleep(0.03)
                    if move_start is not None:
                        t_end = move_start + PERIOD
                        while time.monotonic() < t_end:
                            if not self._kinetic_floor_ok("A_unstick_driving", gear=-1):
                                return False
                            time.sleep(0.03)
                    self.publish_stop("kinetic_lateral_unstick_reverse")
                    time.sleep(SETTLE)

                    check_L = self.latest_us[self.us_left_idx]
                    check_R = self.latest_us[self.us_right_idx]
                    cd_l = (check_L - pre_unstick_L) * 100
                    cd_r = (check_R - pre_unstick_R) * 100
                    still_blocked = abs(cd_l) < 0.01 or abs(cd_r) < 0.01

                    self.get_logger().info(
                        f"KINETIC-A unstick [{unstick_try+1}/"
                        f"{self.us_kinetic_unstick_max_retries}]: "
                        f"ΔL={cd_l:+.2f}cm ΔR={cd_r:+.2f}cm vs pre-reverse — "
                        f"{'still blocked, reversing again' if still_blocked else 'cleared'}")

                    if not still_blocked:
                        break
                    pre_unstick_L, pre_unstick_R = check_L, check_R
                else:
                    self.get_logger().warning(
                        f"KINETIC-A: still blocked after "
                        f"{self.us_kinetic_unstick_max_retries} reverse attempts "
                        f"— giving up on unstick, returning to normal correction")

                unchanged_streak = 0
                prev_L, prev_R = None, None   # force a fresh diff baseline
                continue

            total_fwd_est += SPEED * PERIOD  # kept for logging/diagnostics only

            steer = self._kinetic_clearance_capped_steer(err, L, R)

            # Pre-steer toward the new correction angle (car is fully
            # stopped from the previous cycle's end, or this is cycle 1).
            self.cmd_pub.publish(String(data=json.dumps({
                "type": "drive", "gear": 0, "speed_mps": 0.0, "steer_deg": steer})))
            time.sleep(self.us_steer_wait_s)
            if not self._kinetic_floor_ok("A_post_steer", gear=1):
                return False

            # Issue the drive command with a generous watchdog duration —
            # we control the actual stop ourselves below, this is just a
            # safety ceiling in case our own stop call is ever delayed.
            rd_before = self.latest_enc_rd
            self.cmd_pub.publish(String(data=json.dumps({
                "type": "drive", "gear": 1, "speed_mps": SPEED, "steer_deg": steer,
                "duration": round(START_TMO + PERIOD + 1.0, 2)})))

            self.get_logger().info(
                f"KINETIC-A [{attempt+1}]: driving FWD steer={steer:+.1f}° "
                f"— waiting for encoder-confirmed movement start "
                f"(timeout {START_TMO:.1f}s)")

            # Wait for the encoder to confirm the car actually started
            # moving — counting the 1s drive window from command-send time
            # would include PWM ramp-up / stiction delay as "driving".
            t_cmd = time.monotonic()
            move_start = None
            while time.monotonic() - t_cmd < START_TMO:
                if not self._kinetic_floor_ok("A_waiting_move", gear=1):
                    return False
                if abs(self.latest_enc_rd - rd_before) >= MIN_ENC:
                    move_start = time.monotonic()
                    break
                time.sleep(0.03)

            if move_start is None:
                self.get_logger().warning(
                    f"KINETIC-A [{attempt+1}]: no encoder movement within "
                    f"{START_TMO:.1f}s of issuing drive — wheel may be stuck")
                self.publish_stop("kinetic_lateral_stuck")
                time.sleep(SETTLE)
                prev_L, prev_R = L, R
                continue

            # Drive for UP TO PERIOD seconds counted from confirmed start —
            # but cut short immediately if the live reading shows we've
            # already centred. FIX: a full blind PERIOD-second commitment
            # was observed covering 35-46cm of actual travel against an
            # expected ~6cm (speed*PERIOD) — by the time the NEXT cycle's
            # check ran, the car had already blown through the target by a
            # wide margin. Polling the live error during the drive itself
            # (not just at cycle boundaries) stops the car the instant the
            # target is reached, regardless of why a single cycle is
            # covering far more distance than expected.
            t_drive_end = move_start + PERIOD
            while time.monotonic() < t_drive_end:
                if not self._kinetic_floor_ok("A_driving", gear=1):
                    return False
                live_L = self.latest_us[self.us_left_idx]
                live_R = self.latest_us[self.us_right_idx]
                if live_L <= 2.0 and live_R <= 2.0:
                    live_err = (live_L - live_R) / 2.0
                    if abs(live_err) < self.us_deadband_m:
                        self.get_logger().info(
                            f"KINETIC-A [{attempt+1}]: centred mid-cycle "
                            f"(err={live_err*100:.1f}cm) — stopping early")
                        break
                time.sleep(0.03)

            # Full stop before checking the next fresh reading.
            self.publish_stop("kinetic_lateral_cycle_stop")
            time.sleep(SETTLE)

            prev_L, prev_R = L, R

        self.publish_stop("kinetic_lateral_max_attempts")
        self.get_logger().warning(
            f"KINETIC-A: max attempts ({self.us_kinetic_max_attempts}) "
            f"reached without centering")
        return False

    def _kinetic_depth_steps(self) -> bool:
        """Depth correction: same stop-after-each-confirmed-1s-drive
        architecture as lateral. Direction (reverse if too far, forward if
        too close) can flip between cycles; since the car is already fully
        stopped at the start of every cycle here, a direction change just
        needs a fresh pre-steer, no extra stop logic required."""
        SPEED      = self.us_correction_speed_mps
        PERIOD     = self.us_kinetic_recalc_period_s
        MIN_ENC    = self.us_kinetic_min_enc_move_m
        START_TMO  = self.us_kinetic_move_start_timeout_s
        SETTLE     = self.us_kinetic_settle_s
        MAX_REV    = self.us_kinetic_rev_max_m
        MAX_FWD    = self.us_kinetic_fwd_max_m

        target = self.us_rear_target_m
        tol    = self.us_rear_tolerance_m
        lo, hi = target - tol, target + tol

        total_rev_est = 0.0
        total_fwd_est = 0.0
        prev_rear      = None

        for attempt in range(self.us_kinetic_max_attempts):
            if not self._kinetic_floor_ok("B"):
                return False

            rear = self.latest_us[self.us_rear_idx]

            if prev_rear is not None:
                d_rear = (rear - prev_rear) * 100
                unchanged = " [UNCHANGED FROM LAST CYCLE — sensor data not updating]" \
                    if abs(d_rear) < 0.01 else ""
                self.get_logger().info(
                    f"KINETIC-B [{attempt+1}]: Δrear={d_rear:+.2f}cm "
                    f"vs previous cycle{unchanged}")

            self.get_logger().info(
                f"KINETIC-B [{attempt+1}/{self.us_kinetic_max_attempts}]: "
                f"rear={rear*100:.1f}cm target={target*100:.0f}±{tol*100:.0f}cm "
                f"window=[{lo*100:.0f}, {hi*100:.0f}]cm")

            if rear > 2.0:
                self.get_logger().warning("KINETIC-B: rear sensor OOB — stop")
                return True   # nothing trustworthy to correct — treat as pass

            if lo <= rear <= hi:
                self.get_logger().info(
                    f"KINETIC-B: depth OK after {attempt} cycle(s)")
                return True

            error = rear - target   # +ve = too far (reverse), -ve = too close (forward)
            gear  = -1 if error > 0 else 1

            if gear == -1 and total_rev_est >= MAX_REV:
                self.get_logger().warning(
                    f"KINETIC-B: reverse budget exhausted "
                    f"({total_rev_est*100:.1f}cm used) — error={error*100:+.1f}cm remains")
                return False
            if gear == 1 and total_fwd_est >= MAX_FWD:
                self.get_logger().warning(
                    f"KINETIC-B: forward budget exhausted "
                    f"({total_fwd_est*100:.1f}cm used) — error={error*100:+.1f}cm remains")
                return False

            # [FIX] Apply steer_straight_bias_deg during depth moves.
            # Without bias, left motor slower → car drifts right even
            # during Phase-B reverse/forward → lateral error accumulates
            # while depth is being corrected.
            depth_steer = (
                +self.steer_straight_bias_deg if gear > 0
                else -self.steer_straight_bias_deg
            )

            # Car is already fully stopped from the previous cycle (or this
            # is cycle 1) — pre-steer to straight before driving.
            self.cmd_pub.publish(String(data=json.dumps({
                "type": "drive", "gear": 0, "speed_mps": 0.0, "steer_deg": 0.0})))
            time.sleep(self.us_steer_wait_s)
            if not self._kinetic_floor_ok("B_post_steer", gear=gear):
                return False

            rd_before = self.latest_enc_rd
            self.cmd_pub.publish(String(data=json.dumps({
                "type": "drive", "gear": gear, "speed_mps": SPEED,
                "steer_deg": depth_steer,
                "duration": round(START_TMO + PERIOD + 1.0, 2)})))

            direction = "REV" if gear == -1 else "FWD"
            self.get_logger().info(
                f"KINETIC-B [{attempt+1}]: driving {direction} "
                f"(error={error*100:+.1f}cm) — waiting for encoder-confirmed "
                f"movement start (timeout {START_TMO:.1f}s)")

            t_cmd = time.monotonic()
            move_start = None
            while time.monotonic() - t_cmd < START_TMO:
                if not self._kinetic_floor_ok("B_waiting_move", gear=gear):
                    return False
                if abs(self.latest_enc_rd - rd_before) >= MIN_ENC:
                    move_start = time.monotonic()
                    break
                time.sleep(0.03)

            if move_start is None:
                self.get_logger().warning(
                    f"KINETIC-B [{attempt+1}]: no encoder movement within "
                    f"{START_TMO:.1f}s of issuing drive — wheel may be stuck")
                self.publish_stop("kinetic_depth_stuck")
                time.sleep(SETTLE)
                prev_rear = rear
                continue

            # Drive for UP TO PERIOD seconds counted from confirmed start —
            # but cut short immediately if the live reading shows we've
            # reached (or are about to blow past) the target window. FIX:
            # a full blind PERIOD-second commitment was observed covering
            # 35-46cm of actual travel against an expected ~6cm
            # (speed*PERIOD) — by the time the NEXT cycle's check ran, the
            # car had already reversed straight through the entire 15-25cm
            # window down to a near-collision 4.8cm before the floor check
            # caught it. Polling the live rear reading during the drive
            # itself (not just at cycle boundaries) stops the car the
            # instant it enters tolerance — or the instant it overshoots
            # past the window in the direction it's currently moving —
            # regardless of why a single cycle covers far more distance
            # than expected.
            t_drive_end = move_start + PERIOD
            while time.monotonic() < t_drive_end:
                if not self._kinetic_floor_ok("B_driving", gear=gear):
                    return False
                live_rear = self.latest_us[self.us_rear_idx]
                if live_rear <= 2.0:
                    if lo <= live_rear <= hi:
                        self.get_logger().info(
                            f"KINETIC-B [{attempt+1}]: reached tolerance "
                            f"mid-cycle (rear={live_rear*100:.1f}cm) — "
                            f"stopping early")
                        break
                    if gear == -1 and live_rear < lo:
                        self.get_logger().warning(
                            f"KINETIC-B [{attempt+1}]: overshot past window "
                            f"while reversing (rear={live_rear*100:.1f}cm < "
                            f"{lo*100:.0f}cm) — stopping early")
                        break
                    if gear == 1 and live_rear > hi:
                        self.get_logger().warning(
                            f"KINETIC-B [{attempt+1}]: overshot past window "
                            f"while creeping forward (rear={live_rear*100:.1f}cm "
                            f"> {hi*100:.0f}cm) — stopping early")
                        break
                time.sleep(0.03)

            self.publish_stop("kinetic_depth_cycle_stop")
            time.sleep(SETTLE)

            if gear == -1:
                total_rev_est += SPEED * PERIOD
            else:
                total_fwd_est += SPEED * PERIOD
            prev_rear = rear

        self.publish_stop("kinetic_depth_max_attempts")
        self.get_logger().warning(
            f"KINETIC-B: max attempts ({self.us_kinetic_max_attempts}) "
            f"reached without reaching tolerance")
        return False

    def _run_kinetic_centering(self) -> bool:
        """Returns True if the car finished within tolerance on both lateral
        and depth checks, False if it aborted early (safety) or finished
        out-of-tolerance (e.g. budget exhausted before reaching the rear
        target window).

        FIX (per request): rebuilt around stop-after-each-confirmed-cycle
        (see _kinetic_lateral_steps / _kinetic_depth_steps) — drive →
        wait for the encoder to confirm movement actually started → drive
        exactly us_kinetic_recalc_period_s more from that confirmed point
        → full stop → settle → take a fresh ultrasonic reading → recompute
        → repeat until within tolerance. The car is fully stationary every
        time a new sensor reading is taken and a new path is calculated.

        The 1s drive window is timed from CONFIRMED encoder movement, not
        from when the command was sent, so PWM ramp-up / static-friction
        startup delay doesn't eat into the timed driving interval. The
        absolute proximity floor (_kinetic_floor_ok) remains active
        throughout, polled every ~30ms while driving or waiting for
        movement to start — that's a collision-distance check, independent
        of this cycle structure."""
        time.sleep(0.50)

        # ── FIX: re-arm ESP32 before kinetic ─────────────────────────────
        # encoder_bridge's DriveSerial sends stop("distance_reached") at the
        # end of Move 3 via its own fd on /dev/ttyUSB0.  The real ESP32
        # firmware requires {"type":"arm"} before it will accept drive
        # commands after a stop.  Without this, ALL kinetic commands (both
        # forward KINETIC-A and reverse KINETIC-B) are silently ignored and
        # the motor does not react at all.
        if self.enc_busy_from_encoder:
            self.get_logger().warning(
                f"KINETIC START — enc_busy=True (leaked from Move 3!), "
                f"waiting 2s for encoder_bridge to release...")
            time.sleep(2.0)
        else:
            self.get_logger().info(
                f"KINETIC START — enc_busy={self.enc_busy_from_encoder} (OK)")

        self.get_logger().info(
            f"KINETIC ARM — sending arm to ESP32, wait={self.kinetic_rearm_wait_s}s")
        self.cmd_pub.publish(String(data=json.dumps({"type": "arm"})))
        time.sleep(self.kinetic_rearm_wait_s)

        self.get_logger().info(
            f"KINETIC CENTERING START (stop-after-cycle, "
            f"{self.us_kinetic_recalc_period_s:.1f}s drive per cycle, "
            f"encoder-confirmed start)")

        # FIX (per request): "x4 vs x5 difference must be ≤5cm before doing
        # x7" — not just checked once before Phase B starts (trivial, since
        # nothing moves between A finishing and B starting), but actually
        # held true at the END too. Confirmed bug: Phase B's reverse motion
        # drags the car laterally (imperfect straight-line tracking despite
        # steer_straight_bias_deg) — lat_err drifted from 3.9cm (Phase A's
        # own success) to 20.9cm by the time Phase B finished. Detecting
        # that after the fact and reporting FAIL (previous fix) is correct
        # but incomplete — this loop actually RE-RUNS lateral centering
        # afterward and tries again, up to us_kinetic_outer_attempts times,
        # until both axes hold simultaneously or the outer budget runs out.
        lat_ok = depth_ok = False
        L = R = rear = 0.0
        lat_err_cm = 0.0
        for outer in range(self.us_kinetic_outer_attempts):
            if outer > 0:
                self.get_logger().info(
                    f"KINETIC OUTER RETRY [{outer+1}/{self.us_kinetic_outer_attempts}]: "
                    f"lateral drifted out of tolerance after depth phase — "
                    f"re-centring laterally")

            lat_ok   = self._kinetic_lateral_steps()
            depth_ok = self._kinetic_depth_steps()
            # Depth phase always runs even if lateral fell short of
            # tolerance — matches prior behaviour where both phases always
            # got a chance.

            L    = self.latest_us[self.us_left_idx]
            R    = self.latest_us[self.us_right_idx]
            rear = self.latest_us[self.us_rear_idx]
            lat_err_cm = (L - R) / 2 * 100

            lat_final_ok = (L > 2.0 or R > 2.0) or (abs(lat_err_cm) / 100.0 < self.us_deadband_m)
            rear_lo = self.us_rear_target_m - self.us_rear_tolerance_m
            rear_hi = self.us_rear_target_m + self.us_rear_tolerance_m
            rear_final_ok = (rear > 2.0) or (rear_lo <= rear <= rear_hi)

            if lat_final_ok and rear_final_ok:
                break

            if not (lat_ok and depth_ok):
                # A genuine in-phase failure (budget exhausted, safety
                # abort, etc.) — retrying won't help, stop here.
                self.get_logger().warning(
                    f"KINETIC: phase failure (lat_ok={lat_ok} depth_ok={depth_ok}), "
                    f"not a cross-axis drift — not retrying")
                break

            self.get_logger().warning(
                f"KINETIC: both phases reported success individually, but "
                f"the FINAL combined state drifted out of tolerance "
                f"(lat_final_ok={lat_final_ok}, rear_final_ok={rear_final_ok}) "
                f"— cross-axis coupling between phases.")
        else:
            self.get_logger().warning(
                f"KINETIC: outer retry budget ({self.us_kinetic_outer_attempts}) "
                f"exhausted without both axes holding simultaneously")

        lat_final_ok = (L > 2.0 or R > 2.0) or (abs(lat_err_cm) / 100.0 < self.us_deadband_m)
        rear_lo = self.us_rear_target_m - self.us_rear_tolerance_m
        rear_hi = self.us_rear_target_m + self.us_rear_tolerance_m
        rear_final_ok = (rear > 2.0) or (rear_lo <= rear <= rear_hi)
        ok = lat_ok and depth_ok and lat_final_ok and rear_final_ok

        log_fn = self.get_logger().info if ok else self.get_logger().warning
        log_fn(
            f"KINETIC DONE — L={L:.3f}m R={R:.3f}m rear={rear:.3f}m "
            f"lat_err={lat_err_cm:.1f}cm  "
            f"{'OK' if ok else 'OUT-OF-TOLERANCE'}")

        return ok

    # ─────────────────────────────────────────────────────────────────
    # Legacy discrete centering (fallback when us_kinetic_enable=false)
    # ─────────────────────────────────────────────────────────────────

    def _run_lateral_centering(self):
        self.get_logger().info("LATERAL CENTERING START (legacy)")
        time.sleep(0.40)
        for attempt in range(self.us_max_attempts):
            L = self.latest_us[self.us_left_idx]
            R = self.latest_us[self.us_right_idx]
            if L > 2.0 or R > 2.0:
                break
            err = (L - R) / 2.0
            if abs(err) < self.us_deadband_m:
                break
            corr = max(-self.us_max_steer_deg,
                       min(self.us_max_steer_deg, err * self.us_steer_gain))
            corr *= self.lateral_correction_sign
            self._send_correction_move(1,  -corr, self.us_correction_dist_m)
            self._send_correction_move(-1, +corr, self.us_correction_dist_m)
            time.sleep(0.30)
        self.get_logger().info("LATERAL CENTERING DONE")

    def _run_depth_correction(self):
        self.get_logger().info("DEPTH CORRECTION START (legacy)")
        time.sleep(0.40)
        t, tol = self.us_rear_target_m, self.us_rear_tolerance_m
        for _ in range(self.us_depth_max_attempts):
            rear = self.latest_us[self.us_rear_idx]
            if rear > 2.0:
                break
            if t - tol <= rear <= t + tol:
                break
            if rear < t - tol:
                self._send_correction_move(
                    1, 0.0, min(t - tol - rear + 0.02, self.us_correction_dist_m))
            else:
                self._send_correction_move(
                    -1, 0.0, min(rear - t, self.us_depth_max_step_m))
            time.sleep(0.30)

    def _send_correction_move(self, gear: int, steer_deg: float, dist_m: float):
        speed    = self.us_correction_speed_mps
        duration = dist_m / max(speed, 0.01)
        self.cmd_pub.publish(String(data=json.dumps({
            "type": "drive", "gear": 0, "speed_mps": 0.0, "steer_deg": steer_deg})))
        time.sleep(self.us_steer_wait_s)
        self.cmd_pub.publish(String(data=json.dumps({
            "type": "drive", "gear": gear, "speed_mps": speed, "steer_deg": steer_deg})))
        time.sleep(max(0.05, duration - self.us_stop_buffer_s))
        self.publish_stop("correction_move_done")
        time.sleep(self.us_stop_settle_s)

    # ── Helpers ───────────────────────────────────────────────────────

    def _led(self, color: str):
        self.cmd_pub.publish(
            String(data=json.dumps({"type": "led", "color": color})))
        self.get_logger().info(f"LED → {color}")

    @staticmethod
    def _adelta(a: float, b: float) -> float:
        """Signed angle delta b-a, wrapped to (-180, +180]."""
        d = float(b) - float(a)
        while d >  180.0: d -= 360.0
        while d < -180.0: d += 360.0
        return d

    def publish_stop(self, reason):
        self.cmd_pub.publish(
            String(data=json.dumps({"type": "stop", "reason": reason})))
        self.get_logger().warning("STOP: " + reason)


def main(args=None):
    rclpy.init(args=args)
    node = AutoparkMaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop("shutdown")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
