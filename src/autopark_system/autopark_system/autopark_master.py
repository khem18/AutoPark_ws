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


class AutoparkMaster(Node):
    """
    Real-car autopark master.

    Main control:
      planner dist_m -> split into safe chunks -> calibrated speed -> drive duration

    Important:
      - Planner full distance is NOT cut anymore.
      - Each planner motion is split into chunks <= max_segment_dist_m.
      - Optical flow is NOT used as exact distance stop.
      - IMU orientation is NOT used.
      - IMU gyro-z is integrated only for logging.
      - Ultrasonic is emergency stop while driving.
      - Stuck retry is available, but should stay disabled for now.
    """

    def __init__(self):
        super().__init__("autopark_master")

        for name, default in [
            ("start_switch_topic", "/autopark/start_switch"),
            ("pose_topic", "/autopark/start_pose"),
            ("parking_metrics_topic", "/parking_metrics"),
            ("ultrasonic_topic", "/autopark/ultrasonic"),
            ("command_topic", "/autopark/cmd_json"),
            ("plan_topic", "/autopark/plan_result"),
            ("slot_topic", "/autopark/slot_info"),
            ("flow_distance_topic", "/autopark/flow_distance"),
            ("imu_topic", "/imu/data_raw"),

            # Before-start ultrasonic block.
            ("disable_ultrasonic_block", True),
            ("min_clearance_m", 0.12),

            # Planner / pose.
            ("planner_mode", "both_sides"),
            ("use_default_pose_when_missing", True),
            ("default_start_x", 0.0),
            ("default_start_y", 0.70),
            ("default_start_yaw_deg", 180.0),

            # Real mode should be false.
            ("allow_fallback_path", False),

            # Command speed cap sent to ESP32.
            ("speed_scale", 0.10),

            # Keep within current ESP32 steering limit.
            ("max_command_steer_deg", 10.0),

            # Steering settle time before each drive chunk.
            ("steer_wait_seconds", 10.0),
            ("pause_between_commands", 1.0),

            # Calibrated real-car speeds for timed distance.
            ("forward_turn_speed_mps", 0.055),
            ("reverse_turn_speed_mps", 0.076),
            ("forward_straight_speed_mps", 0.0675),
            ("reverse_straight_speed_mps", 0.0775),
            ("straight_steer_threshold_deg", 3.0),

            # Safe chunk distance.
            # Planner distance is preserved, but split into chunks <= this value.
            ("max_segment_dist_m", 0.50),
            ("min_chunk_dist_m", 0.05),

            # Time clamp.
            ("drive_time_min_s", 2.0),
            ("drive_time_max_s", 60.0),

            # Ultrasonic safety during drive only.
            ("drive_ultrasonic_safety", True),
            ("ultrasonic_stop_m", 0.025),

            # Kept for compatibility.
            ("flow_valid_required", False),

            # Usually false because physical start switch already arms ESP32.
            ("send_arm_command_on_start", False),

            # Stuck / no-motion detection.
            # Keep false for now because it false-triggered during slip.
            ("enable_stuck_retry", False),
            ("stuck_check_after_s", 2.0),
            ("stuck_flow_delta_min", 0.02),
            ("stuck_yaw_delta_min_deg", 0.1),
            ("stuck_metrics_delta_min", 3.0),
            ("stuck_kick_speed_mps", 0.10),
            ("stuck_kick_duration_s", 0.4),
            ("stuck_retry_pause_s", 0.2),
            ("stuck_max_retries", 1),
        ]:
            self.declare_parameter(name, default)

        self.disable_ultrasonic_block = bool(self.get_parameter("disable_ultrasonic_block").value)
        self.min_clearance_m = float(self.get_parameter("min_clearance_m").value)

        self.planner_mode = str(self.get_parameter("planner_mode").value)
        self.use_default_pose_when_missing = bool(self.get_parameter("use_default_pose_when_missing").value)
        self.default_start_x = float(self.get_parameter("default_start_x").value)
        self.default_start_y = float(self.get_parameter("default_start_y").value)
        self.default_start_yaw_deg = float(self.get_parameter("default_start_yaw_deg").value)
        self.allow_fallback_path = bool(self.get_parameter("allow_fallback_path").value)

        self.speed_scale = float(self.get_parameter("speed_scale").value)
        self.max_command_steer_deg = float(self.get_parameter("max_command_steer_deg").value)

        self.steer_wait_seconds = float(self.get_parameter("steer_wait_seconds").value)
        self.pause_between_commands = float(self.get_parameter("pause_between_commands").value)

        self.forward_turn_speed_mps = float(self.get_parameter("forward_turn_speed_mps").value)
        self.reverse_turn_speed_mps = float(self.get_parameter("reverse_turn_speed_mps").value)
        self.forward_straight_speed_mps = float(self.get_parameter("forward_straight_speed_mps").value)
        self.reverse_straight_speed_mps = float(self.get_parameter("reverse_straight_speed_mps").value)
        self.straight_steer_threshold_deg = float(self.get_parameter("straight_steer_threshold_deg").value)

        self.max_segment_dist_m = float(self.get_parameter("max_segment_dist_m").value)
        self.min_chunk_dist_m = float(self.get_parameter("min_chunk_dist_m").value)

        self.drive_time_min_s = float(self.get_parameter("drive_time_min_s").value)
        self.drive_time_max_s = float(self.get_parameter("drive_time_max_s").value)

        self.drive_ultrasonic_safety = bool(self.get_parameter("drive_ultrasonic_safety").value)
        self.ultrasonic_stop_m = float(self.get_parameter("ultrasonic_stop_m").value)

        self.flow_valid_required = bool(self.get_parameter("flow_valid_required").value)
        self.send_arm_command_on_start = bool(self.get_parameter("send_arm_command_on_start").value)

        self.enable_stuck_retry = bool(self.get_parameter("enable_stuck_retry").value)
        self.stuck_check_after_s = float(self.get_parameter("stuck_check_after_s").value)
        self.stuck_flow_delta_min = float(self.get_parameter("stuck_flow_delta_min").value)
        self.stuck_yaw_delta_min_deg = float(self.get_parameter("stuck_yaw_delta_min_deg").value)
        self.stuck_metrics_delta_min = float(self.get_parameter("stuck_metrics_delta_min").value)
        self.stuck_kick_speed_mps = float(self.get_parameter("stuck_kick_speed_mps").value)
        self.stuck_kick_duration_s = float(self.get_parameter("stuck_kick_duration_s").value)
        self.stuck_retry_pause_s = float(self.get_parameter("stuck_retry_pause_s").value)
        self.stuck_max_retries = int(self.get_parameter("stuck_max_retries").value)

        self.latest_pose: Optional[Pose2D] = None
        self.latest_us = [9.9] * 8
        self.latest_metrics: List[float] = []
        self.latest_case = self.planner_mode

        # Optical-flow monitor values.
        self.latest_flow_data: List[float] = []
        self.flow_vx_mps = 0.0
        self.flow_distance_m: Optional[float] = None
        self.flow_yaw_rate = 0.0
        self.flow_valid = False
        self.last_flow_time = 0.0

        # IMU yaw from gyro-z integration.
        self.imu_yaw_rad = 0.0
        self.imu_yaw_deg = 0.0
        self.last_imu_time = 0.0
        self.imu_gyro_z = 0.0

        self.busy = False
        self.lock = threading.Lock()

        self.create_subscription(
            Bool,
            self.get_parameter("start_switch_topic").value,
            self.on_start_switch,
            10,
        )
        self.create_subscription(
            Pose2D,
            self.get_parameter("pose_topic").value,
            self.on_pose,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            self.get_parameter("parking_metrics_topic").value,
            self.on_metrics,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            self.get_parameter("ultrasonic_topic").value,
            self.on_ultrasonic,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter("slot_topic").value,
            self.on_slot_info,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            self.get_parameter("flow_distance_topic").value,
            self.on_flow_distance,
            20,
        )
        self.create_subscription(
            Imu,
            self.get_parameter("imu_topic").value,
            self.on_imu,
            50,
        )

        self.cmd_pub = self.create_publisher(
            String,
            self.get_parameter("command_topic").value,
            10,
        )
        self.plan_pub = self.create_publisher(
            String,
            self.get_parameter("plan_topic").value,
            10,
        )

        self.get_logger().info(
            "autopark_master ready: planner full distance chunked into safe timed moves"
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
        data = [float(x) for x in list(msg.data)]
        self.latest_flow_data = data
        self.last_flow_time = time.monotonic()

        if len(data) >= 6:
            self.flow_vx_mps = float(data[0])
            self.flow_distance_m = float(data[1])
            self.flow_yaw_rate = float(data[2])
            self.flow_valid = bool(data[5] > 0.5)
        elif len(data) >= 2:
            self.flow_distance_m = float(data[1])
            self.flow_valid = True
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
            self.imu_gyro_z = wz
            self.imu_yaw_rad += wz * dt
            self.imu_yaw_deg = math.degrees(self.imu_yaw_rad)

    def on_slot_info(self, msg: String):
        try:
            obj = json.loads(msg.data)
            case_name = str(obj.get("case", self.planner_mode)).strip().lower()
            if case_name in ("left_only", "right_only", "both_sides"):
                if case_name != self.latest_case:
                    self.get_logger().info("parking case updated from slot_info: " + case_name)
                self.latest_case = case_name
        except Exception as exc:
            self.get_logger().warning("bad slot_info JSON: " + str(exc))

    def on_start_switch(self, msg: Bool):
        self.get_logger().info("START SWITCH CALLBACK: " + str(msg.data))

        if not msg.data:
            return

        with self.lock:
            if self.busy:
                self.get_logger().warning("ignored start switch because busy")
                return
            self.busy = True

        th = threading.Thread(target=self.autopark_thread, daemon=True)
        th.start()

    # ------------------------------------------------------------------
    # Autopark sequence
    # ------------------------------------------------------------------
    def autopark_thread(self):
        try:
            self.start_autopark()
        except Exception as exc:
            self.get_logger().error("autopark_thread error: " + str(exc))
            self.publish_stop("autopark_thread_error")
        finally:
            with self.lock:
                self.busy = False

    def start_autopark(self):
        self.get_logger().info("START_AUTOPARK ENTERED")

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

        yaw_deg = math.degrees(pose.theta)
        self.get_logger().info(
            "planning from pose x="
            + str(round(pose.x, 4))
            + " y="
            + str(round(pose.y, 4))
            + " theta_deg="
            + str(round(yaw_deg, 2))
        )

        case_name = (
            self.latest_case
            if self.latest_case in ("left_only", "right_only", "both_sides")
            else self.planner_mode
        )

        result: Dict[str, Any] = {}
        motions: List[Dict[str, Any]] = []
        used_safe_fallback = False

        try:
            planned = plan_from_start(pose.x, pose.y, yaw_deg, case_name)
            result = result_to_dict(planned)
            motions = result.get("motions", [])
        except Exception as exc:
            self.get_logger().error("planner error: " + str(exc))
            result = {
                "ok": False,
                "reason": "planner_exception",
                "exception": str(exc),
                "motions": [],
            }
            motions = []

        reason = str(result.get("reason", ""))

        self.get_logger().info(
            "PLANNER FINISHED, motions="
            + str(len(motions))
            + " reason="
            + reason
        )

        if not motions and self.allow_fallback_path:
            motions = self.get_safe_fallback_motions()
            result["reason"] = "safe_internal_fallback_no_planner_motion"
            result["motions"] = motions
            used_safe_fallback = True

        if ("fallback" in reason.lower()) and self.allow_fallback_path:
            motions = self.get_safe_fallback_motions()
            result["reason"] = "safe_internal_fallback_replace_adapter_fallback"
            result["motions"] = motions
            used_safe_fallback = True

        if ("fallback" in reason.lower()) and not self.allow_fallback_path:
            result["control_mode"] = "planner_failed_no_motion"
            result["rejected_fallback_path"] = True
            self.plan_pub.publish(String(data=json.dumps(result)))
            self.publish_stop("planner_failed_no_motion")
            return

        if not motions:
            self.plan_pub.publish(String(data=json.dumps(result)))
            self.publish_stop("planner_failed_no_motion")
            return

        expanded_motions = self.expand_motion_chunks(motions)

        if not expanded_motions:
            result["control_mode"] = "planner_failed_empty_after_chunk_expand"
            self.plan_pub.publish(String(data=json.dumps(result)))
            self.publish_stop("planner_failed_empty_after_chunk_expand")
            return

        if self.latest_metrics:
            result["parking_metrics"] = self.latest_metrics

        result["control_mode"] = "planner_full_distance_chunked_timed_drive_ultrasonic_safety"
        result["used_safe_fallback"] = used_safe_fallback
        result["original_motion_count"] = len(motions)
        result["expanded_chunk_count"] = len(expanded_motions)
        result["max_command_steer_deg"] = self.max_command_steer_deg
        result["max_segment_dist_m"] = self.max_segment_dist_m
        result["min_chunk_dist_m"] = self.min_chunk_dist_m
        result["speed_table_mps"] = {
            "forward_turn": self.forward_turn_speed_mps,
            "reverse_turn": self.reverse_turn_speed_mps,
            "forward_straight": self.forward_straight_speed_mps,
            "reverse_straight": self.reverse_straight_speed_mps,
        }
        result["drive_time_min_s"] = self.drive_time_min_s
        result["drive_time_max_s"] = self.drive_time_max_s
        result["ultrasonic_stop_m"] = self.ultrasonic_stop_m
        result["flow_used_for_exact_stop"] = False
        result["imu_orientation_used"] = False
        result["imu_gyro_z_integrated"] = True
        result["enable_stuck_retry"] = self.enable_stuck_retry

        # Replace motions in published plan result with chunked executable motions.
        result["executable_motions"] = expanded_motions

        self.plan_pub.publish(String(data=json.dumps(result)))

        self.get_logger().info(
            "plan published: case="
            + str(case_name)
            + " original_motions="
            + str(len(motions))
            + " expanded_chunks="
            + str(len(expanded_motions))
            + " safe_fallback="
            + str(used_safe_fallback)
        )

        self.execute_motions(expanded_motions)

    def expand_motion_chunks(self, motions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preserve planner full distance, but split each motion into safe chunks.

        Example:
          planner motion: reverse 1.50 m
          chunks: 0.50 + 0.50 + 0.50

        This replaces the old clamp behavior:
          target_dist_m = min(target_dist_m, 0.50)
        which caused the car to stop shallow.
        """
        expanded: List[Dict[str, Any]] = []

        max_chunk = max(float(self.min_chunk_dist_m), float(self.max_segment_dist_m))
        min_chunk = max(0.01, float(self.min_chunk_dist_m))

        for parent_index, motion in enumerate(motions):
            if not isinstance(motion, dict):
                continue

            gear = int(motion.get("gear", -1))
            steer_deg = float(motion.get("steer_deg", 0.0))

            dist_m = 0.0
            for key in ("dist_m", "target_dist_m", "distance_m", "dist"):
                if key in motion:
                    try:
                        dist_m = abs(float(motion[key]))
                    except Exception:
                        dist_m = 0.0
                    break

            if dist_m <= 0.0:
                continue

            remaining = dist_m
            chunk_index = 0

            while remaining > 1e-6:
                if remaining <= max_chunk:
                    chunk_dist = remaining
                else:
                    chunk_dist = max_chunk

                # Avoid a tiny final chunk, merge it into previous chunk if possible.
                if remaining - chunk_dist > 0.0 and remaining - chunk_dist < min_chunk:
                    chunk_dist = remaining

                chunk_index += 1

                chunk = dict(motion)
                chunk["gear"] = gear
                chunk["steer_deg"] = steer_deg
                chunk["dist_m"] = chunk_dist
                chunk["target_dist_m"] = chunk_dist
                chunk["parent_motion_index"] = parent_index + 1
                chunk["parent_dist_m"] = dist_m
                chunk["chunk_index"] = chunk_index
                chunk["is_chunked"] = dist_m > max_chunk

                expanded.append(chunk)

                remaining -= chunk_dist

        return expanded

    def get_safe_fallback_motions(self) -> List[Dict[str, Any]]:
        return [
            {
                "gear": -1,
                "steer_deg": 10.0,
                "dist_m": 0.30,
            },
            {
                "gear": -1,
                "steer_deg": 5.0,
                "dist_m": 0.15,
            },
        ]

    def get_start_pose_or_default(self) -> Optional[Pose2D]:
        pose = self.latest_pose
        if pose is not None:
            return pose

        if not self.use_default_pose_when_missing:
            return None

        pose = Pose2D()
        pose.x = self.default_start_x
        pose.y = self.default_start_y
        pose.theta = math.radians(self.default_start_yaw_deg)

        self.get_logger().warning(
            "using default pose: " + str((pose.x, pose.y, pose.theta))
        )
        return pose

    def execute_motions(self, motions: List[Dict[str, Any]]):
        self.get_logger().info(
            "EXECUTING MOTION CHUNKS: full planner distance preserved, each chunk timed"
        )

        for i, motion in enumerate(motions):
            segment_index = i + 1
            cmd = self.motion_to_cmd(motion)
            target_dist_m = float(cmd["target_dist_m"])
            drive_speed_mps = float(cmd["speed_mps"])
            drive_time_s = float(cmd["duration"])

            parent_index = motion.get("parent_motion_index", segment_index)
            chunk_index = motion.get("chunk_index", 1)
            parent_dist_m = motion.get("parent_dist_m", target_dist_m)

            self.get_logger().info(
                "CHUNK "
                + str(segment_index)
                + "/"
                + str(len(motions))
                + " parent="
                + str(parent_index)
                + " chunk="
                + str(chunk_index)
                + " parent_dist="
                + str(round(float(parent_dist_m), 3))
                + " gear="
                + str(cmd["gear"])
                + " steer="
                + str(round(cmd["steer_deg"], 2))
                + " dist_m="
                + str(round(target_dist_m, 3))
                + " speed_mps="
                + str(round(drive_speed_mps, 3))
                + " drive_time_s="
                + str(round(drive_time_s, 2))
            )

            # 1) Send steer command first.
            steer_gear = int(cmd["gear"])
            if steer_gear == 0:
                steer_gear = 1

            steer_cmd = {
                "type": "drive",
                "gear": steer_gear,
                "speed_mps": 0.0,
                "steer_deg": cmd["steer_deg"],
                "duration": self.steer_wait_seconds,
            }

            self.publish_cmd(steer_cmd)
            self.get_logger().info(
                "STEER PREP: gear="
                + str(steer_gear)
                + " target="
                + str(round(cmd["steer_deg"], 2))
                + " wait_s="
                + str(round(self.steer_wait_seconds, 2))
            )
            self.sleep_while_ok(self.steer_wait_seconds)

            # 2) Monitoring baselines.
            flow_start_abs = self.flow_distance_m
            imu_start = self.imu_yaw_deg
            metrics_start = list(self.latest_metrics)
            us_start = self.get_ultrasonic_min_m()

            # 3) Send timed drive chunk.
            self.publish_cmd(cmd)

            stop_reason, elapsed, retry_count = self.wait_drive_timed(
                drive_time_s=drive_time_s,
                segment_index=segment_index,
                cmd=cmd,
            )

            flow_end_abs = self.flow_distance_m
            imu_end = self.imu_yaw_deg
            metrics_end = list(self.latest_metrics)
            us_end = self.get_ultrasonic_min_m()

            flow_abs_delta = self.safe_delta(flow_start_abs, flow_end_abs)
            imu_delta = self.angle_delta_deg(imu_start, imu_end)
            metrics_delta = self.metrics_change_score(metrics_start, metrics_end)

            self.get_logger().info(
                "CHUNK LOG "
                + str(segment_index)
                + ": elapsed_s="
                + str(round(elapsed, 2))
                + " target_dist_m="
                + str(round(target_dist_m, 3))
                + " flow_abs_delta="
                + self.fmt_optional(flow_abs_delta, 3)
                + " flow_valid="
                + str(self.flow_valid)
                + " imu_yaw_start_deg="
                + self.fmt_optional(imu_start, 2)
                + " imu_yaw_end_deg="
                + self.fmt_optional(imu_end, 2)
                + " imu_delta_deg="
                + self.fmt_optional(imu_delta, 2)
                + " metrics_change="
                + str(round(metrics_delta, 3))
                + " ultrasonic_start_m="
                + str(round(us_start, 3))
                + " ultrasonic_end_m="
                + str(round(us_end, 3))
                + " stuck_retries="
                + str(retry_count)
                + " stop_reason="
                + stop_reason
            )

            self.publish_stop(stop_reason)

            if stop_reason.startswith("ultrasonic_safety_stop"):
                self.get_logger().warning("parking aborted by ultrasonic emergency stop")
                return

            if stop_reason.startswith("stuck_failed"):
                self.get_logger().warning("parking aborted because car did not move")
                return

            self.sleep_while_ok(self.pause_between_commands)

        self.publish_stop("parking_sequence_done")
        self.get_logger().info("PARKING SEQUENCE DONE")

    def wait_drive_timed(
        self,
        drive_time_s: float,
        segment_index: int,
        cmd: Dict[str, Any],
    ) -> Tuple[str, float, int]:
        start_t = time.monotonic()
        last_log_t = 0.0

        stuck_retries = 0
        stuck_checked = False
        stuck_window_start_t = start_t

        flow_window_start = list(self.latest_flow_data)
        yaw_window_start = float(self.imu_yaw_deg)
        metrics_window_start = list(self.latest_metrics)

        while rclpy.ok():
            now = time.monotonic()
            elapsed_total = now - start_t
            elapsed_window = now - stuck_window_start_t

            if self.drive_ultrasonic_safety and self.ultrasonic_blocked_now():
                dmin = self.get_ultrasonic_min_m()
                self.get_logger().warning(
                    "ultrasonic emergency stop seg="
                    + str(segment_index)
                    + " min_m="
                    + str(round(dmin, 3))
                    + " limit_m="
                    + str(round(self.ultrasonic_stop_m, 3))
                )
                return "ultrasonic_safety_stop_segment_" + str(segment_index), elapsed_total, stuck_retries

            if (
                self.enable_stuck_retry
                and not stuck_checked
                and elapsed_window >= self.stuck_check_after_s
            ):
                stuck_checked = True

                flow_now = list(self.latest_flow_data)
                yaw_now = float(self.imu_yaw_deg)
                metrics_now = list(self.latest_metrics)

                flow_delta = self.flow_change_score(flow_window_start, flow_now)
                yaw_delta = abs(self.angle_delta_deg(yaw_window_start, yaw_now) or 0.0)
                metrics_delta = self.metrics_change_score(metrics_window_start, metrics_now)

                turning = abs(float(cmd.get("steer_deg", 0.0))) > self.straight_steer_threshold_deg
                metrics_moved = metrics_delta > self.stuck_metrics_delta_min

                if turning:
                    stuck = (
                        flow_delta < self.stuck_flow_delta_min
                        and yaw_delta < self.stuck_yaw_delta_min_deg
                        and not metrics_moved
                    )
                else:
                    stuck = (
                        flow_delta < self.stuck_flow_delta_min
                        and not metrics_moved
                    )

                self.get_logger().info(
                    "STUCK CHECK seg="
                    + str(segment_index)
                    + " flow_delta="
                    + str(round(flow_delta, 4))
                    + " yaw_delta_deg="
                    + str(round(yaw_delta, 3))
                    + " metrics_delta="
                    + str(round(metrics_delta, 3))
                    + " metrics_moved="
                    + str(metrics_moved)
                    + " turning="
                    + str(turning)
                    + " stuck="
                    + str(stuck)
                )

                if stuck:
                    if stuck_retries >= self.stuck_max_retries:
                        return (
                            "stuck_failed_segment_" + str(segment_index),
                            elapsed_total,
                            stuck_retries,
                        )

                    stuck_retries += 1

                    self.get_logger().warning(
                        "STUCK RETRY "
                        + str(stuck_retries)
                        + " seg="
                        + str(segment_index)
                    )

                    self.publish_stop("stuck_retry_" + str(stuck_retries))
                    self.sleep_while_ok(self.stuck_retry_pause_s)

                    kick_cmd = dict(cmd)
                    kick_cmd["speed_mps"] = min(
                        abs(self.stuck_kick_speed_mps),
                        abs(self.speed_scale),
                    )
                    kick_cmd["duration"] = self.stuck_kick_duration_s

                    self.publish_cmd(kick_cmd)
                    self.get_logger().warning(
                        "KICK CMD seg="
                        + str(segment_index)
                        + " speed="
                        + str(kick_cmd["speed_mps"])
                        + " duration="
                        + str(kick_cmd["duration"])
                    )
                    self.sleep_while_ok(self.stuck_kick_duration_s)

                    remaining = max(0.2, drive_time_s - elapsed_total)
                    resume_cmd = dict(cmd)
                    resume_cmd["duration"] = remaining
                    self.publish_cmd(resume_cmd)

                    stuck_window_start_t = time.monotonic()
                    flow_window_start = list(self.latest_flow_data)
                    yaw_window_start = float(self.imu_yaw_deg)
                    metrics_window_start = list(self.latest_metrics)
                    stuck_checked = False

            if now - last_log_t > 0.5:
                self.get_logger().info(
                    "TIMED DRIVE chunk="
                    + str(segment_index)
                    + " elapsed="
                    + str(round(elapsed_total, 2))
                    + "/"
                    + str(round(drive_time_s, 2))
                    + " ultrasonic_min_m="
                    + str(round(self.get_ultrasonic_min_m(), 3))
                    + " flow_dist_m="
                    + self.fmt_optional(self.flow_distance_m, 3)
                    + " flow_valid="
                    + str(self.flow_valid)
                    + " imu_yaw_deg="
                    + self.fmt_optional(self.imu_yaw_deg, 2)
                    + " gyro_z="
                    + str(round(float(self.imu_gyro_z), 4))
                    + " stuck_retries="
                    + str(stuck_retries)
                )
                last_log_t = now

            if elapsed_total >= drive_time_s:
                return "segment_timed_distance_complete", elapsed_total, stuck_retries

            time.sleep(0.05)

        return "ros_shutdown", time.monotonic() - start_t, stuck_retries

    # ------------------------------------------------------------------
    # Command conversion
    # ------------------------------------------------------------------
    def motion_to_cmd(self, motion: Dict[str, Any]) -> Dict[str, Any]:
        gear = -1
        steer_deg = 0.0
        target_dist_m = 0.0

        if isinstance(motion, dict):
            gear = int(motion.get("gear", gear))
            steer_deg = float(motion.get("steer_deg", steer_deg))

            for key in ("dist_m", "target_dist_m", "distance_m", "dist"):
                if key in motion:
                    try:
                        target_dist_m = abs(float(motion[key]))
                    except Exception:
                        target_dist_m = 0.0
                    break

            # Do NOT clamp here.
            # Distance is already safely chunked in expand_motion_chunks().
            # Clamping here would cut planner distance and stop shallow.

        if gear > 0:
            gear = 1
        elif gear < 0:
            gear = -1
        else:
            gear = 0

        steer_deg = self.clamp(
            steer_deg,
            -abs(self.max_command_steer_deg),
            abs(self.max_command_steer_deg),
        )

        turning = abs(float(steer_deg)) > self.straight_steer_threshold_deg
        calibrated_speed_mps = self.select_calibrated_speed(gear, steer_deg)

        # Command speed sent to ESP32.
        # Must overcome motor deadband.
        if gear > 0:
            command_speed_mps = 0.09 if turning else 0.08
        elif gear < 0:
            command_speed_mps = 0.09 if turning else 0.08
        else:
            command_speed_mps = 0.0

        command_speed_mps = min(abs(command_speed_mps), abs(self.speed_scale))

        if gear == 0 or target_dist_m <= 0.0:
            command_speed_mps = 0.0
            drive_time_s = self.drive_time_min_s
        else:
            drive_time_s = target_dist_m / max(abs(calibrated_speed_mps), 0.001)
            drive_time_s = max(
                self.drive_time_min_s,
                min(self.drive_time_max_s, drive_time_s),
            )

        return {
            "type": "drive",
            "gear": gear,
            "speed_mps": command_speed_mps,
            "steer_deg": steer_deg,
            "target_dist_m": target_dist_m,
            "duration": drive_time_s,
        }

    def select_calibrated_speed(self, gear: int, steer_deg: float) -> float:
        turning = abs(float(steer_deg)) > self.straight_steer_threshold_deg

        if gear >= 0:
            return self.forward_turn_speed_mps if turning else self.forward_straight_speed_mps

        return self.reverse_turn_speed_mps if turning else self.reverse_straight_speed_mps

    # ------------------------------------------------------------------
    # Sensors / safety
    # ------------------------------------------------------------------
    def get_ultrasonic_min_m(self) -> float:
        vals_m = []

        for v in self.latest_us:
            try:
                x = float(v)
            except Exception:
                continue

            if x <= 0.0:
                continue

            # Support both m and cm.
            if x > 3.0:
                x = x / 100.0

            if x > 5.0:
                continue

            vals_m.append(x)

        if not vals_m:
            return 9.9

        return min(vals_m)

    def ultrasonic_blocked_now(self) -> bool:
        return self.get_ultrasonic_min_m() < self.ultrasonic_stop_m

    # ------------------------------------------------------------------
    # Feedback scoring
    # ------------------------------------------------------------------
    def flow_change_score(self, before: List[float], after: List[float]) -> float:
        if not before or not after:
            return 0.0

        n = min(len(before), len(after))

        # In your flow array, the last value is often valid flag 0/1.
        if n >= 2:
            n -= 1

        if n <= 0:
            return 0.0

        diffs = []
        for i in range(n):
            try:
                diffs.append(abs(float(after[i]) - float(before[i])))
            except Exception:
                pass

        if not diffs:
            return 0.0

        return max(diffs)

    def metrics_change_score(self, before: List[float], after: List[float]) -> float:
        if not before or not after:
            return 0.0

        n = min(len(before), len(after))
        if n <= 0:
            return 0.0

        diffs = []
        for i in range(n):
            try:
                diffs.append(abs(float(after[i]) - float(before[i])))
            except Exception:
                pass

        if not diffs:
            return 0.0

        return max(diffs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def publish_cmd(self, obj: Dict[str, Any]):
        self.cmd_pub.publish(String(data=json.dumps(obj)))

    def publish_stop(self, reason: str):
        self.publish_cmd({"type": "stop", "reason": reason})
        self.get_logger().warning("STOP: " + reason)

    def sleep_while_ok(self, seconds: float):
        end_t = time.monotonic() + max(0.0, float(seconds))

        while rclpy.ok() and time.monotonic() < end_t:
            time.sleep(0.05)

    @staticmethod
    def safe_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None

        return float(b) - float(a)

    @staticmethod
    def angle_delta_deg(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None

        d = float(b) - float(a)

        while d > 180.0:
            d -= 360.0

        while d < -180.0:
            d += 360.0

        return d

    @staticmethod
    def fmt_optional(value: Optional[float], digits: int = 3) -> str:
        if value is None:
            return "None"

        return str(round(float(value), digits))

    @staticmethod
    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))


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
