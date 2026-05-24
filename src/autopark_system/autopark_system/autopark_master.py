"""
autopark_master.py  —  v_next2
=============================================================
Drive stop hierarchy (slippery floor, no drive encoder):

  TURNING moves  (|steer| > imu_arc_min_steer_deg):
    PRIMARY STOP  → IMU yaw integration → arc length ≥ target × arc_stop_factor
    FALLBACK      → time (dist_m / calibrated_speed)
    EMERGENCY     → ultrasonic < 25 mm

  STRAIGHT moves  (|steer| ≤ imu_arc_min_steer_deg):
    PRIMARY STOP  → optical flow distance delta ≥ target × flow_stop_factor
    FALLBACK      → time
    EMERGENCY     → ultrasonic < 25 mm

Stuck detection (both types):
  TURNING:  |gyro_z| < stuck_gyro_min_rads for > stuck_check_after_s → stuck
  STRAIGHT: flow_delta < stuck_flow_delta_min for > stuck_check_after_s → stuck

IMU arc formula (bicycle kinematics, rear axle):
  R = wheelbase / tan(steer_rad)
  arc_dist = R × |Δyaw_rad|

Why this works on slippery floor:
  - Wheel slip changes arc time, but the car still rotates physically
  - IMU measures the actual rotation → arc distance is real regardless of slip
  - Flow measures actual ground distance for straight segments
  - Time is the last resort when sensors fail (overcautious but safe)
"""

import json
import math
import time
import threading
from typing import Optional, Dict, Any, List, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, Float32MultiArray
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Imu

from .planner_adapter import plan_from_start, result_to_dict

WHEELBASE_M = 0.739  # matches VehicleSpec in planner


def arc_dist_from_yaw_delta(delta_yaw_deg: float, steer_deg: float) -> float:
    """
    Compute arc distance at rear axle from IMU yaw change.
    Returns 0.0 if steer is too small to be reliable.
    """
    steer_rad = math.radians(abs(steer_deg))
    if steer_rad < math.radians(0.5):
        return 0.0
    R = WHEELBASE_M / math.tan(steer_rad)
    return R * math.radians(abs(delta_yaw_deg))


class AutoparkMaster(Node):

    def __init__(self):
        super().__init__("autopark_master")

        for name, default in [
            # Topics
            ("start_switch_topic",       "/autopark/start_switch"),
            ("pose_topic",               "/autopark/start_pose"),
            ("parking_metrics_topic",    "/parking_metrics"),
            ("ultrasonic_topic",         "/autopark/ultrasonic"),
            ("command_topic",            "/autopark/cmd_json"),
            ("plan_topic",               "/autopark/plan_result"),
            ("slot_topic",               "/autopark/slot_info"),
            ("flow_distance_topic",      "/autopark/flow_distance"),
            ("imu_topic",                "/imu/data_raw"),
            ("esp32_steer_ready_topic",  "/autopark/esp32_steer_ready"),

            # Planner / pose
            ("planner_mode",                    "right_only"),
            ("use_default_pose_when_missing",   True),
            ("default_start_x",                 0.0),
            ("default_start_y",                 0.70),
            ("default_start_yaw_deg",           180.0),
            ("allow_fallback_path",             False),

            # Pre-start safety
            ("disable_ultrasonic_block",        True),
            ("min_clearance_m",                 0.12),

            # Speed / steer limits
            ("speed_scale",                     0.10),
            ("max_command_steer_deg",           22.0),

            # Steer settle gate
            ("steer_ready_timeout_s",           7.0),
            ("steer_settle_pause_s",            0.3),
            ("steer_wait_fallback_s",           4.0),
            ("pause_between_commands",          0.8),

            # Calibrated speeds (used for time fallback only)
            ("forward_turn_speed_mps",          0.055),
            ("reverse_turn_speed_mps",          0.076),
            ("forward_straight_speed_mps",      0.0675),
            ("reverse_straight_speed_mps",      0.0775),
            ("straight_steer_threshold_deg",    3.0),

            # Chunk splitting
            ("max_segment_dist_m",              0.50),
            ("min_chunk_dist_m",                0.05),
            ("drive_time_min_s",                1.0),
            ("drive_time_max_s",                60.0),

            # ── IMU ARC STOP (turning moves) ──────────────────────────
            # Enable IMU yaw-based arc distance stop for turning chunks.
            # Only active when |steer_deg| > imu_arc_min_steer_deg.
            # Stop is triggered when arc_distance >= target × arc_stop_factor.
            # arc_stop_factor < 1.0 stops early to allow for motor coasting.
            # Set arc_stop_factor = 1.0 to stop exactly at target.
            ("imu_arc_stop_enabled",            True),
            ("imu_arc_min_steer_deg",           8.0),   # below this → use flow
            ("imu_arc_stop_factor",             0.92),  # stop at 92% of target arc
            ("imu_arc_wait_before_check_s",     0.3),   # ignore first 300 ms (motor ramp)

            # ── FLOW STRAIGHT STOP (straight / low-steer moves) ───────
            # Enable optical flow distance stop for straight chunks.
            # Only active when |steer_deg| <= imu_arc_min_steer_deg.
            # Requires flow_valid = True from flow_distance_node.
            # If flow is invalid, falls back to timed stop.
            ("flow_straight_stop_enabled",      True),
            ("flow_straight_stop_factor",       0.90),  # stop at 90% of target
            ("flow_straight_wait_before_check_s", 0.3),

            # ── STUCK DETECTION ───────────────────────────────────────
            # Turning stuck: no rotation detected.
            # Straight stuck: no flow movement detected.
            ("enable_stuck_detection",          True),
            ("stuck_check_after_s",             1.5),   # wait this long before declaring stuck
            ("stuck_gyro_min_rads",             0.015), # rad/s — below = stuck during turn
            ("stuck_flow_delta_min_m",          0.005), # m — below = stuck during straight
            ("stuck_kick_speed_mps",            0.10),
            ("stuck_kick_duration_s",           0.5),
            ("stuck_retry_pause_s",             0.2),
            ("stuck_max_retries",               2),

            # Ultrasonic emergency
            ("drive_ultrasonic_safety",         True),
            ("ultrasonic_stop_m",               0.025),

            # Flow / IMU role (for compatibility)
            ("flow_valid_required",             False),
            ("send_arm_command_on_start",       False),
        ]:
            self.declare_parameter(name, default)

        def gp(n):
            return self.get_parameter(n).value

        self.planner_mode                   = str(gp("planner_mode"))
        self.use_default_pose_when_missing  = bool(gp("use_default_pose_when_missing"))
        self.default_start_x                = float(gp("default_start_x"))
        self.default_start_y                = float(gp("default_start_y"))
        self.default_start_yaw_deg          = float(gp("default_start_yaw_deg"))
        self.allow_fallback_path            = bool(gp("allow_fallback_path"))
        self.disable_ultrasonic_block       = bool(gp("disable_ultrasonic_block"))
        self.min_clearance_m                = float(gp("min_clearance_m"))
        self.speed_scale                    = float(gp("speed_scale"))
        self.max_command_steer_deg          = float(gp("max_command_steer_deg"))
        self.steer_ready_timeout_s          = float(gp("steer_ready_timeout_s"))
        self.steer_settle_pause_s           = float(gp("steer_settle_pause_s"))
        self.steer_wait_fallback_s          = float(gp("steer_wait_fallback_s"))
        self.pause_between_commands         = float(gp("pause_between_commands"))
        self.forward_turn_speed_mps         = float(gp("forward_turn_speed_mps"))
        self.reverse_turn_speed_mps         = float(gp("reverse_turn_speed_mps"))
        self.forward_straight_speed_mps     = float(gp("forward_straight_speed_mps"))
        self.reverse_straight_speed_mps     = float(gp("reverse_straight_speed_mps"))
        self.straight_steer_threshold_deg   = float(gp("straight_steer_threshold_deg"))
        self.max_segment_dist_m             = float(gp("max_segment_dist_m"))
        self.min_chunk_dist_m               = float(gp("min_chunk_dist_m"))
        self.drive_time_min_s               = float(gp("drive_time_min_s"))
        self.drive_time_max_s               = float(gp("drive_time_max_s"))

        self.imu_arc_stop_enabled           = bool(gp("imu_arc_stop_enabled"))
        self.imu_arc_min_steer_deg          = float(gp("imu_arc_min_steer_deg"))
        self.imu_arc_stop_factor            = float(gp("imu_arc_stop_factor"))
        self.imu_arc_wait_before_check_s    = float(gp("imu_arc_wait_before_check_s"))

        self.flow_straight_stop_enabled         = bool(gp("flow_straight_stop_enabled"))
        self.flow_straight_stop_factor          = float(gp("flow_straight_stop_factor"))
        self.flow_straight_wait_before_check_s  = float(gp("flow_straight_wait_before_check_s"))

        self.enable_stuck_detection         = bool(gp("enable_stuck_detection"))
        self.stuck_check_after_s            = float(gp("stuck_check_after_s"))
        self.stuck_gyro_min_rads            = float(gp("stuck_gyro_min_rads"))
        self.stuck_flow_delta_min_m         = float(gp("stuck_flow_delta_min_m"))
        self.stuck_kick_speed_mps           = float(gp("stuck_kick_speed_mps"))
        self.stuck_kick_duration_s          = float(gp("stuck_kick_duration_s"))
        self.stuck_retry_pause_s            = float(gp("stuck_retry_pause_s"))
        self.stuck_max_retries              = int(gp("stuck_max_retries"))

        self.drive_ultrasonic_safety        = bool(gp("drive_ultrasonic_safety"))
        self.ultrasonic_stop_m              = float(gp("ultrasonic_stop_m"))
        self.flow_valid_required            = bool(gp("flow_valid_required"))
        self.send_arm_command_on_start      = bool(gp("send_arm_command_on_start"))

        # State
        self.latest_pose:       Optional[Pose2D] = None
        self.latest_us:         List[float]      = [9.9] * 8
        self.latest_metrics:    List[float]      = []
        self.latest_case:       str              = self.planner_mode

        self.flow_distance_m:   Optional[float]  = None
        self.flow_valid:        bool             = False
        self.latest_flow_data:  List[float]      = []
        self.last_flow_time:    float            = 0.0

        # IMU — integrated yaw (absolute, for logging)
        self.imu_yaw_rad:       float = 0.0
        self.imu_yaw_deg:       float = 0.0
        self.imu_gyro_z:        float = 0.0   # current rad/s
        self.last_imu_time:     float = 0.0

        self.esp32_steer_ready:        bool = False
        self.esp32_steer_ready_seen:   bool = False

        self.busy = False
        self.lock = threading.Lock()

        # Subscriptions
        self.create_subscription(Bool,              gp("start_switch_topic"),      self.on_start_switch,       10)
        self.create_subscription(Pose2D,            gp("pose_topic"),              self.on_pose,               10)
        self.create_subscription(Float32MultiArray, gp("parking_metrics_topic"),   self.on_metrics,            10)
        self.create_subscription(Float32MultiArray, gp("ultrasonic_topic"),        self.on_ultrasonic,         10)
        self.create_subscription(String,            gp("slot_topic"),              self.on_slot_info,          10)
        self.create_subscription(Float32MultiArray, gp("flow_distance_topic"),     self.on_flow_distance,      20)
        self.create_subscription(Imu,               gp("imu_topic"),               self.on_imu,                50)
        self.create_subscription(Bool,              gp("esp32_steer_ready_topic"), self.on_esp32_steer_ready,  20)

        self.cmd_pub  = self.create_publisher(String, gp("command_topic"), 10)
        self.plan_pub = self.create_publisher(String, gp("plan_topic"),    10)

        self.get_logger().info(
            "autopark_master v_next2 ready\n"
            "  TURNING  stop: IMU arc distance (primary) + time (fallback)\n"
            "  STRAIGHT stop: optical flow distance (primary) + time (fallback)\n"
            "  EMERGENCY:     ultrasonic < {:.0f} mm".format(self.ultrasonic_stop_m * 1000)
        )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def on_pose(self, msg: Pose2D):
        self.latest_pose = msg

    def on_metrics(self, msg: Float32MultiArray):
        self.latest_metrics = list(msg.data)

    def on_ultrasonic(self, msg: Float32MultiArray):
        vals = list(msg.data)
        if len(vals) >= 8:
            self.latest_us = vals[:8]

    def on_flow_distance(self, msg: Float32MultiArray):
        data = [float(x) for x in msg.data]
        self.latest_flow_data = data
        self.last_flow_time   = time.monotonic()
        if len(data) >= 6:
            self.flow_distance_m = float(data[1])
            self.flow_valid      = bool(data[5] > 0.5)
        elif len(data) >= 2:
            self.flow_distance_m = float(data[1])
            self.flow_valid      = True
        else:
            self.flow_valid = False

    def on_imu(self, msg: Imu):
        now = time.monotonic()
        if self.last_imu_time <= 0.0:
            self.last_imu_time = now
            return
        dt = now - self.last_imu_time
        self.last_imu_time = now
        if 0.0 < dt < 0.2:
            wz = float(msg.angular_velocity.z)
            self.imu_gyro_z   = wz
            self.imu_yaw_rad += wz * dt
            self.imu_yaw_deg  = math.degrees(self.imu_yaw_rad)

    def on_slot_info(self, msg: String):
        try:
            obj       = json.loads(msg.data)
            case_name = str(obj.get("case", self.planner_mode)).strip().lower()
            if case_name in ("left_only", "right_only", "both_sides"):
                if case_name != self.latest_case:
                    self.get_logger().info("case updated: " + case_name)
                self.latest_case = case_name
        except Exception as exc:
            self.get_logger().warning("bad slot_info JSON: " + str(exc))

    def on_esp32_steer_ready(self, msg: Bool):
        self.esp32_steer_ready      = bool(msg.data)
        self.esp32_steer_ready_seen = True

    def on_start_switch(self, msg: Bool):
        if not msg.data:
            return
        self.get_logger().info("START SWITCH received")
        with self.lock:
            if self.busy:
                self.get_logger().warning("ignored — already busy")
                return
            self.busy = True
        threading.Thread(target=self.autopark_thread, daemon=True).start()

    # ------------------------------------------------------------------
    # Autopark thread
    # ------------------------------------------------------------------

    def autopark_thread(self):
        try:
            self.start_autopark()
        except Exception as exc:
            self.get_logger().error("autopark_thread: " + str(exc))
            self.publish_stop("autopark_thread_error")
        finally:
            with self.lock:
                self.busy = False

    def start_autopark(self):
        self.get_logger().info("AUTOPARK START")

        if self.send_arm_command_on_start:
            self.publish_cmd({"type": "arm"})
            time.sleep(0.2)

        if not self.disable_ultrasonic_block:
            if self.get_ultrasonic_min_m() < self.min_clearance_m:
                self.publish_stop("blocked_by_ultrasonic_before_start")
                return

        pose = self.get_start_pose_or_default()
        if pose is None:
            self.publish_stop("no_start_pose")
            return

        yaw_deg  = math.degrees(pose.theta)
        case_name = (
            self.latest_case
            if self.latest_case in ("left_only", "right_only", "both_sides")
            else self.planner_mode
        )

        self.get_logger().info(
            "planning x={:.3f} y={:.3f} yaw={:.1f}° case={}".format(
                pose.x, pose.y, yaw_deg, case_name))

        # -- Plan --
        motions: List[Dict[str, Any]] = []
        result:  Dict[str, Any]       = {}
        used_safe_fallback = False

        try:
            planned = plan_from_start(pose.x, pose.y, yaw_deg, case_name)
            result  = result_to_dict(planned)

            if isinstance(result.get("executable_motions"), list) and result["executable_motions"]:
                motions = result["executable_motions"]
            elif isinstance(result.get("motions"), list) and result["motions"]:
                motions = result["motions"]
            elif isinstance(result.get("primitive_seq"), list) and result["primitive_seq"]:
                motions = [
                    {
                        "gear":      1 if p.get("direction", "r") == "f" else -1,
                        "steer_deg": float(p.get("steer_deg", 0.0)),
                        "dist_m":    float(p.get("dist_m", 0.0)),
                    }
                    for p in result["primitive_seq"]
                ]

            self.get_logger().info("planner success={} motions={}".format(
                result.get("success"), len(motions)))

        except Exception as exc:
            self.get_logger().error("planner: " + str(exc))
            result = {"ok": False, "reason": "planner_exception", "motions": []}

        reason = str(result.get("reason", ""))

        if not motions and self.allow_fallback_path:
            motions = self.get_safe_fallback_motions()
            result["reason"] = "safe_fallback"
            used_safe_fallback = True
        elif "fallback" in reason.lower() and self.allow_fallback_path:
            motions = self.get_safe_fallback_motions()
            result["reason"] = "safe_fallback_replaced"
            used_safe_fallback = True
        elif "fallback" in reason.lower() and not self.allow_fallback_path:
            self.plan_pub.publish(String(data=json.dumps(result)))
            self.publish_stop("planner_failed_no_motion")
            return

        if not motions:
            self.plan_pub.publish(String(data=json.dumps(result)))
            self.publish_stop("planner_failed_no_motion")
            return

        expanded = self.expand_motion_chunks(motions)
        if not expanded:
            self.plan_pub.publish(String(data=json.dumps(result)))
            self.publish_stop("planner_failed_empty_after_expand")
            return

        result.update({
            "control_mode":         "imu_arc_flow_straight_ultrasonic_emergency",
            "used_safe_fallback":   used_safe_fallback,
            "original_motion_count": len(motions),
            "expanded_chunk_count": len(expanded),
            "executable_motions":   expanded,
            "imu_arc_stop_enabled": self.imu_arc_stop_enabled,
            "flow_straight_stop_enabled": self.flow_straight_stop_enabled,
            "imu_arc_min_steer_deg": self.imu_arc_min_steer_deg,
            "speed_table_mps": {
                "forward_turn":     self.forward_turn_speed_mps,
                "reverse_turn":     self.reverse_turn_speed_mps,
                "forward_straight": self.forward_straight_speed_mps,
                "reverse_straight": self.reverse_straight_speed_mps,
            },
        })

        self.plan_pub.publish(String(data=json.dumps(result)))
        self.get_logger().info(
            "plan published case={} original={} chunks={}".format(
                case_name, len(motions), len(expanded)))

        self.execute_motions(expanded)

    # ------------------------------------------------------------------
    # Chunk expansion
    # ------------------------------------------------------------------

    def expand_motion_chunks(self, motions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        expanded: List[Dict[str, Any]] = []
        max_chunk = max(float(self.min_chunk_dist_m), float(self.max_segment_dist_m))
        min_chunk = max(0.01, float(self.min_chunk_dist_m))

        for parent_idx, motion in enumerate(motions):
            if not isinstance(motion, dict):
                continue
            gear      = int(motion.get("gear", -1))
            steer_deg = float(motion.get("steer_deg", 0.0))
            dist_m    = 0.0
            for key in ("dist_m", "target_dist_m", "distance_m", "dist"):
                if key in motion:
                    try:
                        dist_m = abs(float(motion[key]))
                    except Exception:
                        dist_m = 0.0
                    break
            if dist_m <= 0.0:
                continue

            remaining   = dist_m
            chunk_index = 0
            while remaining > 1e-6:
                chunk_dist = min(max_chunk, remaining)
                if 0.0 < (remaining - chunk_dist) < min_chunk:
                    chunk_dist = remaining
                chunk_index += 1
                chunk = dict(motion)
                chunk.update({
                    "gear":                gear,
                    "steer_deg":           steer_deg,
                    "dist_m":              chunk_dist,
                    "target_dist_m":       chunk_dist,
                    "parent_motion_index": parent_idx + 1,
                    "parent_dist_m":       dist_m,
                    "chunk_index":         chunk_index,
                    "is_chunked":          dist_m > max_chunk,
                })
                expanded.append(chunk)
                remaining -= chunk_dist
        return expanded

    # ------------------------------------------------------------------
    # Motion execution
    # ------------------------------------------------------------------

    def execute_motions(self, motions: List[Dict[str, Any]]):
        self.get_logger().info("EXECUTE {} chunks".format(len(motions)))

        for i, motion in enumerate(motions):
            seg  = i + 1
            cmd  = self.motion_to_cmd(motion)
            target_dist_m   = float(cmd["target_dist_m"])
            drive_time_s    = float(cmd["duration"])
            steer_deg       = float(cmd["steer_deg"])
            is_turning      = abs(steer_deg) > self.imu_arc_min_steer_deg

            # Which primary stop method will be used
            stop_method = "imu_arc" if (is_turning and self.imu_arc_stop_enabled) \
                else ("flow" if self.flow_straight_stop_enabled else "time")

            self.get_logger().info(
                "CHUNK {}/{} parent={} chunk={}  "
                "gear={} steer={:.1f}° dist={:.3f}m time={:.2f}s  "
                "stop_method={}".format(
                    seg, len(motions),
                    motion.get("parent_motion_index", seg),
                    motion.get("chunk_index", 1),
                    cmd["gear"], steer_deg, target_dist_m, drive_time_s,
                    stop_method))

            # ── STEP 1: Steer settle ──────────────────────────────────
            steer_gear = int(cmd["gear"]) or 1
            self.publish_cmd({
                "type":      "drive",
                "gear":      steer_gear,
                "speed_mps": 0.0,
                "steer_deg": steer_deg,
                "duration":  self.steer_ready_timeout_s + 1.0,
            })
            self.get_logger().info("STEER → {:.1f}° waiting steer_ready…".format(steer_deg))
            steer_ok = self.wait_for_steer_ready()
            if not steer_ok:
                self.get_logger().warning("steer_ready timeout — ESP32 gate still active")
            if self.steer_settle_pause_s > 0:
                self.sleep_while_ok(self.steer_settle_pause_s)

            # ── STEP 2: Baselines before drive ───────────────────────
            imu_start_deg   = self.imu_yaw_deg
            flow_start_m    = self.flow_distance_m
            us_start        = self.get_ultrasonic_min_m()
            gyro_window_abs = []  # running gyro readings for stuck check

            # ── STEP 3: Drive ─────────────────────────────────────────
            self.publish_cmd(cmd)

            stop_reason, elapsed, retries = self.wait_drive_sensor_stop(
                drive_time_s   = drive_time_s,
                target_dist_m  = target_dist_m,
                steer_deg      = steer_deg,
                is_turning     = is_turning,
                segment_index  = seg,
                imu_start_deg  = imu_start_deg,
                flow_start_m   = flow_start_m,
                cmd            = cmd,
            )

            # ── STEP 4: Log ──────────────────────────────────────────
            imu_delta  = self.angle_delta_deg(imu_start_deg, self.imu_yaw_deg)
            flow_delta = self.safe_delta(flow_start_m, self.flow_distance_m)
            arc_actual = arc_dist_from_yaw_delta(abs(imu_delta or 0.0), steer_deg)

            self.get_logger().info(
                "CHUNK LOG {}: stop={} elapsed={:.2f}s  "
                "imu_delta={:.2f}°  arc_actual={:.3f}m  target={:.3f}m  "
                "flow_delta={}  us_end={:.3f}m  retries={}".format(
                    seg, stop_reason, elapsed,
                    imu_delta or 0.0, arc_actual, target_dist_m,
                    self.fmt_optional(flow_delta, 3),
                    self.get_ultrasonic_min_m(), retries))

            self.publish_stop(stop_reason)

            if stop_reason.startswith("ultrasonic_emergency"):
                self.get_logger().warning("ABORT — ultrasonic emergency")
                return
            if stop_reason.startswith("stuck_failed"):
                self.get_logger().warning("ABORT — car did not move")
                return

            self.sleep_while_ok(self.pause_between_commands)

        self.publish_stop("parking_sequence_done")
        self.get_logger().info("PARKING SEQUENCE DONE")

    # ------------------------------------------------------------------
    # Sensor-based drive stop
    # ------------------------------------------------------------------

    def wait_drive_sensor_stop(
        self,
        drive_time_s:   float,
        target_dist_m:  float,
        steer_deg:      float,
        is_turning:     bool,
        segment_index:  int,
        imu_start_deg:  float,
        flow_start_m:   Optional[float],
        cmd:            Dict[str, Any],
    ) -> Tuple[str, float, int]:
        """
        Monitor drive and stop based on:
          Turning  → IMU arc distance reaches target × arc_stop_factor
          Straight → optical flow delta reaches target × flow_stop_factor
          Both     → time fallback (drive_time_s)
          Both     → ultrasonic emergency < ultrasonic_stop_m
        """
        start_t        = time.monotonic()
        last_log_t     = 0.0
        stuck_retries  = 0

        # Targets for sensor-based stops
        arc_target_deg: Optional[float] = None
        if is_turning and self.imu_arc_stop_enabled:
            steer_rad = math.radians(abs(steer_deg))
            R = WHEELBASE_M / math.tan(steer_rad)
            arc_target_deg = math.degrees(
                (target_dist_m * self.imu_arc_stop_factor) / R)
            self.get_logger().info(
                "  ARC STOP: target={:.3f}m factor={:.2f} → "
                "trigger_yaw_delta={:.2f}° (R={:.3f}m)".format(
                    target_dist_m, self.imu_arc_stop_factor,
                    arc_target_deg, R))

        flow_target_m: Optional[float] = None
        if (not is_turning) and self.flow_straight_stop_enabled and flow_start_m is not None:
            flow_target_m = target_dist_m * self.flow_straight_stop_factor
            self.get_logger().info(
                "  FLOW STOP: target={:.3f}m factor={:.2f} → "
                "trigger_flow_delta={:.3f}m".format(
                    target_dist_m, self.flow_straight_stop_factor, flow_target_m))

        # Stuck monitoring window
        stuck_window_start_t  = start_t
        stuck_imu_start_deg   = imu_start_deg
        stuck_flow_start_m    = flow_start_m
        stuck_checked         = False

        while rclpy.ok():
            now           = time.monotonic()
            elapsed       = now - start_t
            elapsed_window = now - stuck_window_start_t

            # ── Ultrasonic emergency ──────────────────────────────────
            if self.drive_ultrasonic_safety and self.ultrasonic_blocked_now():
                return ("ultrasonic_emergency_seg_{}".format(segment_index),
                        elapsed, stuck_retries)

            # ── IMU arc stop (turning) ────────────────────────────────
            if (arc_target_deg is not None
                    and elapsed >= self.imu_arc_wait_before_check_s):
                delta_yaw = abs(self.angle_delta_deg(imu_start_deg, self.imu_yaw_deg) or 0.0)
                if delta_yaw >= arc_target_deg:
                    return ("imu_arc_stop_seg_{}".format(segment_index),
                            elapsed, stuck_retries)

            # ── Flow straight stop ────────────────────────────────────
            if (flow_target_m is not None
                    and self.flow_valid
                    and self.flow_distance_m is not None
                    and elapsed >= self.flow_straight_wait_before_check_s):
                flow_delta = abs(
                    (self.flow_distance_m or 0.0) - (flow_start_m or 0.0))
                if flow_delta >= flow_target_m:
                    return ("flow_stop_seg_{}".format(segment_index),
                            elapsed, stuck_retries)

            # ── Stuck detection ───────────────────────────────────────
            if (self.enable_stuck_detection
                    and not stuck_checked
                    and elapsed_window >= self.stuck_check_after_s):
                stuck_checked = True
                stuck = False

                if is_turning:
                    # Expect rotation; check gyro_z magnitude
                    gyro_abs = abs(self.imu_gyro_z)
                    stuck = gyro_abs < self.stuck_gyro_min_rads
                    self.get_logger().info(
                        "  STUCK CHECK turn seg={} gyro_z={:.4f} min={:.4f} stuck={}".format(
                            segment_index, gyro_abs, self.stuck_gyro_min_rads, stuck))
                else:
                    # Expect linear motion; check flow delta over window
                    if self.flow_valid and self.flow_distance_m is not None and stuck_flow_start_m is not None:
                        flow_window_delta = abs(
                            (self.flow_distance_m or 0.0) - (stuck_flow_start_m or 0.0))
                        stuck = flow_window_delta < self.stuck_flow_delta_min_m
                        self.get_logger().info(
                            "  STUCK CHECK straight seg={} flow_delta={:.4f} min={:.4f} stuck={}".format(
                                segment_index, flow_window_delta,
                                self.stuck_flow_delta_min_m, stuck))
                    else:
                        # Flow invalid — can't check; assume ok, let time handle it
                        pass

                if stuck:
                    if stuck_retries >= self.stuck_max_retries:
                        return ("stuck_failed_seg_{}".format(segment_index),
                                elapsed, stuck_retries)
                    stuck_retries += 1
                    self.get_logger().warning(
                        "  STUCK RETRY {} seg={}".format(stuck_retries, segment_index))

                    self.publish_stop("stuck_retry_{}".format(stuck_retries))
                    self.sleep_while_ok(self.stuck_retry_pause_s)

                    kick = dict(cmd)
                    kick["speed_mps"] = min(abs(self.stuck_kick_speed_mps), abs(self.speed_scale))
                    kick["duration"]  = self.stuck_kick_duration_s
                    self.publish_cmd(kick)
                    self.sleep_while_ok(self.stuck_kick_duration_s)

                    # Resume with remaining time
                    remaining = max(0.5, drive_time_s - elapsed)
                    resume = dict(cmd)
                    resume["duration"] = remaining
                    self.publish_cmd(resume)

                    # Reset stuck window
                    stuck_window_start_t = time.monotonic()
                    stuck_imu_start_deg  = self.imu_yaw_deg
                    stuck_flow_start_m   = self.flow_distance_m
                    stuck_checked        = False

            # ── Progress log ─────────────────────────────────────────
            if now - last_log_t >= 0.5:
                delta_yaw = abs(self.angle_delta_deg(imu_start_deg, self.imu_yaw_deg) or 0.0)
                arc_now   = arc_dist_from_yaw_delta(delta_yaw, steer_deg)
                flow_now  = self.safe_delta(flow_start_m, self.flow_distance_m)
                self.get_logger().info(
                    "  DRIVE seg={} {:.2f}/{:.2f}s  "
                    "imu_delta={:.2f}° arc={:.3f}m  "
                    "flow_delta={}  us={:.3f}m  gyro_z={:.4f}".format(
                        segment_index, elapsed, drive_time_s,
                        delta_yaw, arc_now,
                        self.fmt_optional(flow_now, 3),
                        self.get_ultrasonic_min_m(),
                        self.imu_gyro_z))
                last_log_t = now

            # ── Time fallback ─────────────────────────────────────────
            if elapsed >= drive_time_s:
                return ("time_stop_seg_{}".format(segment_index),
                        elapsed, stuck_retries)

            time.sleep(0.04)

        return "ros_shutdown", time.monotonic() - start_t, stuck_retries

    # ------------------------------------------------------------------
    # Steer-ready wait
    # ------------------------------------------------------------------

    def wait_for_steer_ready(self) -> bool:
        deadline = time.monotonic() + self.steer_ready_timeout_s
        self.esp32_steer_ready = False
        while rclpy.ok() and time.monotonic() < deadline:
            if self.esp32_steer_ready:
                return True
            if (not self.esp32_steer_ready_seen
                    and time.monotonic() > (deadline - self.steer_ready_timeout_s
                                            + self.steer_wait_fallback_s)):
                self.get_logger().warning(
                    "esp32_steer_ready topic missing — timed fallback done")
                return False
            time.sleep(0.04)
        return False

    # ------------------------------------------------------------------
    # Command conversion
    # ------------------------------------------------------------------

    def motion_to_cmd(self, motion: Dict[str, Any]) -> Dict[str, Any]:
        gear      = int(motion.get("gear", -1))
        steer_deg = float(motion.get("steer_deg", 0.0))
        dist_m    = 0.0
        for key in ("dist_m", "target_dist_m", "distance_m", "dist"):
            if key in motion:
                try:
                    dist_m = abs(float(motion[key]))
                except Exception:
                    dist_m = 0.0
                break
        gear = 1 if gear > 0 else (-1 if gear < 0 else 0)
        steer_deg = self.clamp(steer_deg,
                               -abs(self.max_command_steer_deg),
                               abs(self.max_command_steer_deg))
        calibrated = self.select_calibrated_speed(gear, steer_deg)
        turning    = abs(steer_deg) > self.straight_steer_threshold_deg

        if gear != 0:
            cmd_speed = min(0.09 if turning else 0.08, abs(self.speed_scale))
        else:
            cmd_speed = 0.0

        if gear == 0 or dist_m <= 0.0:
            cmd_speed    = 0.0
            drive_time_s = self.drive_time_min_s
        else:
            drive_time_s = dist_m / max(abs(calibrated), 0.001)
            drive_time_s = self.clamp(drive_time_s,
                                      self.drive_time_min_s,
                                      self.drive_time_max_s)

        return {
            "type":          "drive",
            "gear":          gear,
            "speed_mps":     cmd_speed,
            "steer_deg":     steer_deg,
            "target_dist_m": dist_m,
            "duration":      drive_time_s,
        }

    def select_calibrated_speed(self, gear: int, steer_deg: float) -> float:
        turning = abs(steer_deg) > self.straight_steer_threshold_deg
        if gear >= 0:
            return self.forward_turn_speed_mps if turning else self.forward_straight_speed_mps
        return self.reverse_turn_speed_mps if turning else self.reverse_straight_speed_mps

    # ------------------------------------------------------------------
    # Sensors / helpers
    # ------------------------------------------------------------------

    def get_ultrasonic_min_m(self) -> float:
        vals = []
        for v in self.latest_us:
            try:
                x = float(v)
            except Exception:
                continue
            if x <= 0.0:
                continue
            if x > 3.0:
                x /= 100.0
            if x > 5.0:
                continue
            vals.append(x)
        return min(vals) if vals else 9.9

    def ultrasonic_blocked_now(self) -> bool:
        return self.get_ultrasonic_min_m() < self.ultrasonic_stop_m

    def get_start_pose_or_default(self) -> Optional[Pose2D]:
        if self.latest_pose is not None:
            return self.latest_pose
        if not self.use_default_pose_when_missing:
            return None
        p = Pose2D()
        p.x     = self.default_start_x
        p.y     = self.default_start_y
        p.theta = math.radians(self.default_start_yaw_deg)
        self.get_logger().warning("using default pose ({},{},{})".format(
            p.x, p.y, math.degrees(p.theta)))
        return p

    def get_safe_fallback_motions(self) -> List[Dict[str, Any]]:
        return [
            {"gear": -1, "steer_deg": 10.0, "dist_m": 0.30},
            {"gear": -1, "steer_deg":  5.0, "dist_m": 0.15},
        ]

    def publish_cmd(self, obj: Dict[str, Any]):
        self.cmd_pub.publish(String(data=json.dumps(obj)))

    def publish_stop(self, reason: str):
        self.publish_cmd({"type": "stop", "reason": reason})
        self.get_logger().warning("STOP: " + reason)

    def sleep_while_ok(self, seconds: float):
        end = time.monotonic() + max(0.0, float(seconds))
        while rclpy.ok() and time.monotonic() < end:
            time.sleep(0.04)

    @staticmethod
    def safe_delta(a, b):
        if a is None or b is None:
            return None
        return float(b) - float(a)

    @staticmethod
    def angle_delta_deg(a, b):
        if a is None or b is None:
            return None
        d = float(b) - float(a)
        while d >  180.0: d -= 360.0
        while d < -180.0: d += 360.0
        return d

    @staticmethod
    def fmt_optional(v, digits=3):
        return "None" if v is None else str(round(float(v), digits))

    @staticmethod
    def clamp(value, lo, hi):
        return max(lo, min(hi, float(value)))


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
