"""
autopark_master.py  —  v_next9
=============================================================================
Changes vs v_next8 (zip version):
  1. speed_scale 0.01 → 0.03.
  2. dist_m forwarded in every drive command:
       Move 1 (fwd_setup, gear=+1)           → encoder_bridge FORWARD MONITOR
       Move 2 (rev_arc_90, gear=-1 steer=30°) → serial_bridge + arc_monitor (IMU stop)
       Move 3 (rev_straight_d4, gear=-1 steer=0°, use_encoder=True)
                                               → encoder_bridge REVERSE DRIVE
  3. Subscribes to /enc_result → waits for SUCCESS instead of time-sleep
     for Moves 1 and 3.
  4. Subscribes to /imu/data_raw → monitors yaw delta for arc (Move 2) stop.
  5. Subscribes to /enc_status  → live right-encoder distance for progress log.
  6. steer_straight_bias_deg param: +ve = right bias applied to steer=0 moves.
       gear=+1 straight: add +bias  (compensates left-slow leftward drift)
       gear=-1 straight: add -bias  (compensates left-slow rightward drift in rev)
  7. MOVE X/N logging + DRIVE dist/target progress at 500 ms intervals.
  8. Single encoder: encoder_bridge already uses rightDistM only (no change needed).
  9. Kinetic ultrasonic centering retained from previous version.
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
            # 0.51 → stop at ~51% of arc (gives physical car time to coast to end).
            ("imu_arc_stop_factor",          0.51),
            # Seconds to wait after arc starts before checking IMU delta:
            ("imu_arc_wait_before_check_s",  0.3),
            # Hard timeout if IMU stop never triggers:
            ("imu_arc_stop_timeout_s",       120.0),

            # ── Encoder result wait ─────────────────────────────────────────
            # Timeout for waiting on /enc_result after a Move 1 or 3 command:
            ("enc_result_timeout_s",         130.0),

            # ── Ultrasonic centering ────────────────────────────────────────
            ("us_center_enable",             True),
            ("us_left_idx",                  1),
            ("us_right_idx",                 2),
            ("us_deadband_m",                0.060),
            ("us_steer_gain",                150.0),
            ("us_max_steer_deg",             15.0),
            ("us_correction_dist_m",         0.12),
            ("us_correction_speed_mps",      0.06),
            ("us_max_attempts",              3),
            ("us_rear_idx",                  6),
            ("us_rear_target_m",             0.20),
            ("us_rear_tolerance_m",          0.06),
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
        self.imu_arc_wait_s            = float(gp("imu_arc_wait_before_check_s"))
        self.imu_arc_stop_timeout_s    = float(gp("imu_arc_stop_timeout_s"))

        self.enc_result_timeout_s      = float(gp("enc_result_timeout_s"))

        self.us_center_enable          = bool(gp("us_center_enable"))
        self.us_left_idx               = int(gp("us_left_idx"))
        self.us_right_idx              = int(gp("us_right_idx"))
        self.us_deadband_m             = float(gp("us_deadband_m"))
        self.us_steer_gain             = float(gp("us_steer_gain"))
        self.us_max_steer_deg          = float(gp("us_max_steer_deg"))
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

        # ── state ─────────────────────────────────────────────────────
        self.latest_pose: Optional[Pose2D] = None
        self.latest_pose_time: Optional[float] = None
        self.latest_us         = [9.9] * 8
        self.latest_metrics    = []
        self.busy              = False

        # IMU state
        # latest_yaw_rad: from quaternion (often identity on mpu6050_cpp → unreliable)
        # latest_gyro_z:  angular_velocity.z in rad/s — always valid on MPU6050
        self.latest_yaw_rad    = 0.0
        self.latest_gyro_z     = 0.0   # rad/s, used for arc yaw integration
        self.imu_received      = False

        # Encoder bridge state
        self.enc_result_event  = threading.Event()
        self.enc_result_data   = None
        self.latest_enc_rd     = 0.0   # rightDistM from /enc_status

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
            self.latest_us = vals[:8]

    def on_imu(self, msg: Imu):
        self.latest_yaw_rad = _yaw_from_imu(msg)
        # Store gyro Z rate — used for arc angle integration.
        # mpu6050_cpp publishes valid angular_velocity even when orientation
        # quaternion is identity (no fusion running), so this is the reliable source.
        self.latest_gyro_z  = msg.angular_velocity.z
        self.imu_received   = True

    def on_enc_result(self, msg: String):
        self.enc_result_data = msg.data
        self.enc_result_event.set()

    def on_enc_status(self, msg: String):
        try:
            obj = json.loads(msg.data)
            self.latest_enc_rd = float(obj.get("rd", self.latest_enc_rd))
        except Exception:
            pass

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
        if btn_state == "parking" and not self.busy:
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
        if self.us_center_enable:
            if self.us_kinetic_enable:
                self._run_kinetic_centering()
            else:
                self._run_lateral_centering()
                self._run_depth_correction()

        self.get_logger().info("PARKING DONE → Green LED 1s")
        self._led("green")
        self.publish_stop("parking_complete")

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
                self._wait_arc_imu(s23, R_m, i, n)
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

    def _wait_arc_imu(self, s23_m: float, R_m: float, seg_idx: int, n: int):
        """
        Wait for arc (Move 2) completion via gyroscope integration.

        Uses angular_velocity.z (gyro rate) integration — NOT orientation quaternion.
        The mpu6050_cpp node does not run orientation fusion, so the quaternion
        is always identity (yaw=0°).  The gyro rate is always valid.

        Stop criterion:
          |∫ gyro_z dt| >= arc_deg_full * imu_arc_stop_factor

        Falls back to time-based if:
          - imu_arc_stop_enabled = False
          - IMU topic not yet received
          - Gyro appears stuck at zero for the first imu_arc_wait_s seconds
        """
        arc_rad      = s23_m / max(R_m, 0.001)
        arc_deg_full = math.degrees(arc_rad)
        stop_deg     = arc_deg_full * self.imu_arc_stop_factor
        timeout      = self.imu_arc_stop_timeout_s

        self.get_logger().info(
            f"[IMU arc] arc={arc_deg_full:.1f}°  "
            f"stop_trig={stop_deg:.1f}°  "
            f"factor={self.imu_arc_stop_factor}  "
            f"(gyro integration)")

        if not self.imu_arc_stop_enabled or not self.imu_received:
            reason = "disabled" if not self.imu_arc_stop_enabled else "IMU not ready"
            fallback_s = s23_m / max(self.speed_scale, 0.01) * 1.5
            self.get_logger().warning(
                f"[IMU arc] {reason} — time-based fallback ({fallback_s:.1f}s)")
            time.sleep(min(fallback_s, timeout))
            return

        # ── Gyro integration ──────────────────────────────────────
        gyro_integral_rad = 0.0   # accumulates ∫ gyro_z dt
        t0       = time.monotonic()
        t_last   = t0
        last_log = 0.0

        self.get_logger().info(
            f"DRIVE 0.00/{timeout:.2f}s arc=0.0/{arc_deg_full:.1f}°")

        while True:
            time.sleep(0.05)
            now     = time.monotonic()
            dt      = now - t_last
            t_last  = now
            elapsed = now - t0

            # Integrate gyro Z (rad/s × s = rad)
            gyro_integral_rad += self.latest_gyro_z * dt
            delta_deg = math.degrees(abs(gyro_integral_rad))

            # ── Gyro-zero fallback ─────────────────────────────────
            # If gyro still reads ~0 after the warm-up delay, the IMU
            # might not be publishing useful data → time-based fallback.
            if elapsed >= self.imu_arc_wait_s and abs(gyro_integral_rad) < math.radians(0.5):
                # Only trigger this once (at the warm-up boundary)
                if abs(elapsed - self.imu_arc_wait_s) < 0.15:
                    fallback_s = max(0.1, s23_m / max(self.speed_scale, 0.01) * 1.5 - elapsed)
                    self.get_logger().warning(
                        f"[IMU arc] gyro reads ~0 after {elapsed:.1f}s — "
                        f"time-based fallback ({fallback_s:.1f}s remaining)")
                    time.sleep(fallback_s)
                    return

            if elapsed - last_log >= 0.5:
                self.get_logger().info(
                    f"DRIVE {elapsed:.2f}/{timeout:.2f}s "
                    f"arc={delta_deg:.1f}/{arc_deg_full:.1f}°")
                last_log = elapsed

            # ── IMU stop condition (after warm-up delay) ───────────
            if elapsed >= self.imu_arc_wait_s and delta_deg >= stop_deg:
                self.get_logger().info(
                    f"LOG: stop=imu_arc_stop_seg_{seg_idx + 1}  "
                    f"elapsed={elapsed:.2f}s  "
                    f"imu_delta={delta_deg:.2f}°  "
                    f"gyro_z={math.degrees(self.latest_gyro_z):.1f}°/s")
                return

            if elapsed >= timeout:
                self.get_logger().warning(
                    f"[IMU arc] timeout after {elapsed:.1f}s — "
                    f"imu_delta={delta_deg:.2f}° < {stop_deg:.1f}°")
                return

    # ─────────────────────────────────────────────────────────────────
    # Kinetic ultrasonic centering (post-parking)
    # ─────────────────────────────────────────────────────────────────
    #
    # Phase A: forward creep, streaming steer updates from L/R US.
    # Phase B: reverse straight to rear-US target depth.
    # ─────────────────────────────────────────────────────────────────

    def _run_kinetic_centering(self):
        SPEED    = self.us_correction_speed_mps
        DT       = 1.0 / max(self.us_kinetic_update_hz, 1.0)
        MAX_FWD  = self.us_kinetic_fwd_max_m
        MAX_REV  = self.us_kinetic_rev_max_m
        SETTLE   = self.us_kinetic_settle_s
        KEEPALIVE = round(DT * 6, 2)

        time.sleep(0.50)
        self.get_logger().info("KINETIC CENTERING START")

        # ── Phase A: lateral ─────────────────────────────────────────
        L   = self.latest_us[self.us_left_idx]
        R   = self.latest_us[self.us_right_idx]
        err = (L - R) / 2.0

        self.get_logger().info(
            f"KINETIC-A: L={L:.3f}m R={R:.3f}m err={err*100:.1f}cm "
            f"deadband={self.us_deadband_m*100:.0f}cm")

        if L > 2.0 or R > 2.0:
            self.get_logger().warning("KINETIC-A: lateral sensors OOB — skip")
        elif abs(err) < self.us_deadband_m:
            self.get_logger().info("KINETIC-A: already centred — skip")
        else:
            correction = max(-self.us_max_steer_deg,
                             min(self.us_max_steer_deg, err * self.us_steer_gain))
            steer0 = -correction   # gear=+1: neg = left steer compensates right excess

            self.cmd_pub.publish(String(data=json.dumps({
                "type": "drive", "gear": 0, "speed_mps": 0.0, "steer_deg": steer0})))
            time.sleep(self.us_steer_wait_s)

            steps = max(1, int(MAX_FWD / (SPEED * DT)))
            for step in range(steps):
                L   = self.latest_us[self.us_left_idx]
                R   = self.latest_us[self.us_right_idx]
                err = (L - R) / 2.0

                if abs(err) < self.us_deadband_m:
                    self.get_logger().info(
                        f"KINETIC-A: centred step={step} err={err*100:.1f}cm")
                    break

                corr  = max(-self.us_max_steer_deg,
                            min(self.us_max_steer_deg, err * self.us_steer_gain))
                steer = -corr

                self.cmd_pub.publish(String(data=json.dumps({
                    "type": "drive", "gear": 1, "speed_mps": SPEED,
                    "steer_deg": steer, "duration": KEEPALIVE})))

                if step % 5 == 0:
                    self.get_logger().info(
                        f"KINETIC-A [{step:02d}]: L={L:.3f} R={R:.3f} "
                        f"err={err*100:.1f}cm steer={steer:+.1f}°")
                time.sleep(DT)

            self.publish_stop("kinetic_fwd_done")
            time.sleep(SETTLE)

        # ── Phase B: depth ────────────────────────────────────────────
        rear   = self.latest_us[self.us_rear_idx]
        target = self.us_rear_target_m
        tol    = self.us_rear_tolerance_m
        hi     = target + tol

        self.get_logger().info(
            f"KINETIC-B: rear={rear:.3f}m target={target:.2f}±{tol:.2f}m")

        if rear > 2.0:
            self.get_logger().warning("KINETIC-B: rear sensor OOB — skip")
        elif rear > hi:
            self.cmd_pub.publish(String(data=json.dumps({
                "type": "drive", "gear": 0, "speed_mps": 0.0, "steer_deg": 0.0})))
            time.sleep(self.us_steer_wait_s)

            steps = max(1, int(MAX_REV / (SPEED * DT)))
            for step in range(steps):
                rear = self.latest_us[self.us_rear_idx]
                if rear <= hi:
                    self.get_logger().info(
                        f"KINETIC-B: depth OK step={step} rear={rear*100:.1f}cm")
                    break
                self.cmd_pub.publish(String(data=json.dumps({
                    "type": "drive", "gear": -1, "speed_mps": SPEED,
                    "steer_deg": 0.0, "duration": KEEPALIVE})))
                if step % 5 == 0:
                    self.get_logger().info(
                        f"KINETIC-B [{step:02d}]: rear={rear:.3f}m")
                time.sleep(DT)

            self.publish_stop("kinetic_rev_done")
            time.sleep(SETTLE)

        elif rear < target - tol:
            creep = min(target - tol - rear + 0.02, self.us_depth_max_step_m)
            self._send_correction_move(gear=1, steer_deg=0.0, dist_m=creep)
        else:
            self.get_logger().info("KINETIC-B: depth within window — skip")

        L    = self.latest_us[self.us_left_idx]
        R    = self.latest_us[self.us_right_idx]
        rear = self.latest_us[self.us_rear_idx]
        self.get_logger().info(
            f"KINETIC DONE — L={L:.3f}m R={R:.3f}m rear={rear:.3f}m "
            f"lat_err={(L-R)/2*100:.1f}cm")

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
