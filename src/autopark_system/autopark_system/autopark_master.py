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
    Real-car autopark master, timed-distance version.

    Control rule:
      planner dist_m -> calibrated speed -> drive duration

    Optical flow and IMU are NOT used as stop conditions.
    They are used only for logging.

    Ultrasonic is used only as emergency stop while driving.
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
            # Keep disabled for tight parking test.
            ("disable_ultrasonic_block", True),
            ("min_clearance_m", 0.12),

            # Planner / pose.
            ("planner_mode", "both_sides"),
            ("use_default_pose_when_missing", True),
            ("default_start_x", 0.0),
            ("default_start_y", 0.70),
            ("default_start_yaw_deg", 180.0),

            # For hardware test only.
            # Real mode should be false.
            ("allow_fallback_path", True),

            # Command speed cap sent to ESP32.
            ("speed_scale", 0.07),

            # IMPORTANT:
            # Your ESP32 current WHEEL_MAX_STEER_DEG is 12 deg.
            # So keep command below that.
            ("max_command_steer_deg", 10.0),

            # Steering settle time before each drive segment.
            # ESP32 steerReady also gates the drive motor.
            ("steer_wait_seconds", 10.0),
            ("pause_between_commands", 1.0),

            # Calibrated real-car speeds.
            # Updated from your test:
            # forward 0.07 for 4 s = 0.27 m -> 0.0675 m/s
            # reverse 0.07 for 4 s = 0.31 m -> 0.0775 m/s
            ("forward_turn_speed_mps", 0.040),
            ("reverse_turn_speed_mps", 0.050),
            ("forward_straight_speed_mps", 0.0675),
            ("reverse_straight_speed_mps", 0.0775),
            ("straight_steer_threshold_deg", 3.0),

            # Time clamp.
            ("drive_time_min_s", 1.0),
            ("drive_time_max_s", 60.0),

            # Ultrasonic safety during drive only.
            # For first fallback test, keep false.
            # For real parking, change to true.
            ("drive_ultrasonic_safety", False),
            ("ultrasonic_stop_m", 0.025),

            # Kept for compatibility, but not used as stop logic.
            ("flow_valid_required", False),

            # Usually false because the physical start switch already arms ESP32.
            ("send_arm_command_on_start", False),
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

        self.drive_time_min_s = float(self.get_parameter("drive_time_min_s").value)
        self.drive_time_max_s = float(self.get_parameter("drive_time_max_s").value)

        self.drive_ultrasonic_safety = bool(self.get_parameter("drive_ultrasonic_safety").value)
        self.ultrasonic_stop_m = float(self.get_parameter("ultrasonic_stop_m").value)

        self.flow_valid_required = bool(self.get_parameter("flow_valid_required").value)
        self.send_arm_command_on_start = bool(self.get_parameter("send_arm_command_on_start").value)

        self.latest_pose: Optional[Pose2D] = None
        self.latest_us = [9.9] * 8
        self.latest_metrics: List[float] = []
        self.latest_case = self.planner_mode

        # Optical-flow monitor values only.
        self.flow_vx_mps = 0.0
        self.flow_distance_m: Optional[float] = None
        self.flow_yaw_rate = 0.0
        self.flow_valid = False
        self.last_flow_time = 0.0

        # IMU monitor values only.
        self.imu_yaw_rad: Optional[float] = None
        self.imu_yaw_deg: Optional[float] = None
        self.last_imu_time = 0.0

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
            20,
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
            "autopark_master ready: timed-distance drive, "
            "safe fallback steer, ultrasonic emergency stop, flow/IMU log only"
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
        data = list(msg.data)
        if len(data) >= 6:
            self.flow_vx_mps = float(data[0])
            self.flow_distance_m = float(data[1])
            self.flow_yaw_rate = float(data[2])
            self.flow_valid = bool(data[5] > 0.5)
            self.last_flow_time = time.monotonic()

    def on_imu(self, msg: Imu):
        q = msg.orientation
        yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.imu_yaw_rad = yaw
        self.imu_yaw_deg = math.degrees(yaw)
        self.last_imu_time = time.monotonic()

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

        # If planner failed/no output and fallback is allowed, use safe fallback.
        if not motions and self.allow_fallback_path:
            motions = self.get_safe_fallback_motions()
            result["reason"] = "safe_internal_fallback_no_planner_motion"
            result["motions"] = motions
            used_safe_fallback = True

        # If planner adapter returned fallback but steering is too aggressive,
        # replace it with our safe fallback for current ±12 deg ESP32 limit.
        if ("fallback" in reason.lower()) and self.allow_fallback_path:
            motions = self.get_safe_fallback_motions()
            result["reason"] = "safe_internal_fallback_replace_adapter_fallback"
            result["motions"] = motions
            used_safe_fallback = True

        # In real mode, reject fallback.
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

        if self.latest_metrics:
            result["parking_metrics"] = self.latest_metrics

        result["control_mode"] = "planner_distance_timed_drive_ultrasonic_safety_flow_imu_log_only"
        result["used_safe_fallback"] = used_safe_fallback
        result["max_command_steer_deg"] = self.max_command_steer_deg
        result["speed_table_mps"] = {
            "forward_turn": self.forward_turn_speed_mps,
            "reverse_turn": self.reverse_turn_speed_mps,
            "forward_straight": self.forward_straight_speed_mps,
            "reverse_straight": self.reverse_straight_speed_mps,
        }
        result["drive_time_min_s"] = self.drive_time_min_s
        result["drive_time_max_s"] = self.drive_time_max_s
        result["ultrasonic_stop_m"] = self.ultrasonic_stop_m
        result["flow_used_for_stop"] = False
        result["imu_used_for_stop"] = False

        self.plan_pub.publish(String(data=json.dumps(result)))

        self.get_logger().info(
            "plan published: case="
            + str(case_name)
            + " motions="
            + str(len(motions))
            + " safe_fallback="
            + str(used_safe_fallback)
        )

        self.execute_motions(motions)

    def get_safe_fallback_motions(self) -> List[Dict[str, Any]]:
        """
        Reverse-only safe fallback for real-car test.

        Reason:
        Forward turning is currently inconsistent on your car.
        Reverse turning already moves more reliably.
        """
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
            "EXECUTING MOTIONS: planner dist_m + calibrated time, ultrasonic emergency stop"
        )

        for i, motion in enumerate(motions):
            segment_index = i + 1
            cmd = self.motion_to_cmd(motion)
            target_dist_m = float(cmd["target_dist_m"])
            drive_speed_mps = float(cmd["speed_mps"])
            drive_time_s = float(cmd["duration"])

            self.get_logger().info(
                "SEGMENT "
                + str(segment_index)
                + "/"
                + str(len(motions))
                + ": gear="
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
            # IMPORTANT:
            # Do NOT use gear 0 here, because some ESP32 AUTO logic ignores steering
            # in neutral. Use the same gear as the coming drive command but speed 0.
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

            # 2) Monitoring baselines only.
            flow_start = self.flow_distance_m
            imu_start = self.imu_yaw_deg
            us_start = self.get_ultrasonic_min_m()

            # 3) Send timed drive command.
            # ESP32 still blocks drive until steerReady() is true.
            self.publish_cmd(cmd)

            stop_reason, elapsed = self.wait_drive_timed(
                drive_time_s=drive_time_s,
                segment_index=segment_index,
            )

            flow_end = self.flow_distance_m
            imu_end = self.imu_yaw_deg
            flow_delta = self.safe_delta(flow_start, flow_end)
            imu_delta = self.angle_delta_deg(imu_start, imu_end)
            us_end = self.get_ultrasonic_min_m()

            self.get_logger().info(
                "SEGMENT LOG "
                + str(segment_index)
                + ": elapsed_s="
                + str(round(elapsed, 2))
                + " target_dist_m="
                + str(round(target_dist_m, 3))
                + " flow_delta_m="
                + self.fmt_optional(flow_delta, 3)
                + " flow_valid="
                + str(self.flow_valid)
                + " imu_yaw_start_deg="
                + self.fmt_optional(imu_start, 2)
                + " imu_yaw_end_deg="
                + self.fmt_optional(imu_end, 2)
                + " imu_delta_deg="
                + self.fmt_optional(imu_delta, 2)
                + " ultrasonic_start_m="
                + str(round(us_start, 3))
                + " ultrasonic_end_m="
                + str(round(us_end, 3))
                + " stop_reason="
                + stop_reason
            )

            self.publish_stop(stop_reason)

            if stop_reason.startswith("ultrasonic_safety_stop"):
                self.get_logger().warning("parking aborted by ultrasonic emergency stop")
                return

            self.sleep_while_ok(self.pause_between_commands)

        self.publish_stop("parking_sequence_done")
        self.get_logger().info("PARKING SEQUENCE DONE")

    def wait_drive_timed(self, drive_time_s: float, segment_index: int) -> Tuple[str, float]:
        start_t = time.monotonic()
        last_log_t = 0.0

        while rclpy.ok():
            now = time.monotonic()
            elapsed = now - start_t

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
                return "ultrasonic_safety_stop_segment_" + str(segment_index), elapsed

            if now - last_log_t > 0.5:
                self.get_logger().info(
                    "TIMED DRIVE seg="
                    + str(segment_index)
                    + " elapsed="
                    + str(round(elapsed, 2))
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
                )
                last_log_t = now

            if elapsed >= drive_time_s:
                return "segment_timed_distance_complete", elapsed

            time.sleep(0.05)

        return "ros_shutdown", time.monotonic() - start_t

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
            # TEMPORARY REAL-CAR SAFETY LIMIT
            # Do not allow one planner segment to move more than 30 cm during testing.
            target_dist_m = min(target_dist_m, 0.50)

        if gear > 0:
            gear = 1
        elif gear < 0:
            gear = -1
        else:
            gear = 0

        # Clamp steering command for current ESP32 limit.
        steer_deg = self.clamp(
            steer_deg,
            -abs(self.max_command_steer_deg),
            abs(self.max_command_steer_deg),
        )

        turning = abs(float(steer_deg)) > self.straight_steer_threshold_deg

        # This speed is used ONLY for drive time calculation.
        calibrated_speed_mps = self.select_calibrated_speed(gear, steer_deg)

        # This speed is sent to ESP32.
        # It must be strong enough to overcome motor deadband.
        if gear > 0:
            if turning:
                command_speed_mps = 0.09
            else:
                command_speed_mps = 0.08
        elif gear < 0:
            if turning:
                command_speed_mps = 0.09
            else:
                command_speed_mps = 0.08
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
    def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

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
