import json
import math
import time
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, Float32MultiArray
from geometry_msgs.msg import Pose2D

from .planner_adapter import plan_from_start, result_to_dict


class AutoparkMaster(Node):
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

            # Before-start ultrasonic block.
            # Keep true for now because tight slot may make pre-start ultrasonic too sensitive.
            ("disable_ultrasonic_block", True),
            ("min_clearance_m", 0.12),

            ("planner_mode", "both_sides"),
            ("use_default_pose_when_missing", True),
            ("default_start_x", 0.0),
            ("default_start_y", 0.70),
            ("default_start_yaw_deg", 180.0),

            # Your calibrated real-car speed command
            ("speed_scale", 0.07),

            # Fallback only if flow is missing
            ("default_motion_seconds", 4.0),

            # Steering still uses wait time.
            # ESP32 steerReady should also block drive if steering is not ready.
            ("steer_wait_seconds", 45.0),
            ("pause_between_commands", 1.0),

            # Optical-flow closed-loop drive
            ("use_flow_distance_control", True),
            ("default_drive_distance_m", 0.35),
            ("flow_valid_required", False),
            ("flow_distance_tolerance_m", 0.01),

            # This is NOT normal stop control.
            # It only keeps ESP32 command active for long time.
            ("esp32_drive_hold_seconds", 300.0),

            # Ultrasonic safety during drive
            ("drive_ultrasonic_safety", True),

            # Safety stop distance.
            # 0.04 m = 4 cm. Your slot is tight, so do not set this too high.
            ("ultrasonic_stop_m", 0.04),
        ]:
            self.declare_parameter(name, default)

        self.disable_ultrasonic_block = bool(
            self.get_parameter("disable_ultrasonic_block").value
        )
        self.min_clearance_m = float(self.get_parameter("min_clearance_m").value)

        self.planner_mode = str(self.get_parameter("planner_mode").value)
        self.use_default_pose_when_missing = bool(
            self.get_parameter("use_default_pose_when_missing").value
        )

        self.default_start_x = float(self.get_parameter("default_start_x").value)
        self.default_start_y = float(self.get_parameter("default_start_y").value)
        self.default_start_yaw_deg = float(
            self.get_parameter("default_start_yaw_deg").value
        )

        self.speed_scale = float(self.get_parameter("speed_scale").value)
        self.default_motion_seconds = float(
            self.get_parameter("default_motion_seconds").value
        )
        self.steer_wait_seconds = float(self.get_parameter("steer_wait_seconds").value)
        self.pause_between_commands = float(
            self.get_parameter("pause_between_commands").value
        )

        self.use_flow_distance_control = bool(
            self.get_parameter("use_flow_distance_control").value
        )
        self.default_drive_distance_m = float(
            self.get_parameter("default_drive_distance_m").value
        )
        self.flow_valid_required = bool(
            self.get_parameter("flow_valid_required").value
        )
        self.flow_distance_tolerance_m = float(
            self.get_parameter("flow_distance_tolerance_m").value
        )

        self.esp32_drive_hold_seconds = float(
            self.get_parameter("esp32_drive_hold_seconds").value
        )

        self.drive_ultrasonic_safety = bool(
            self.get_parameter("drive_ultrasonic_safety").value
        )
        self.ultrasonic_stop_m = float(
            self.get_parameter("ultrasonic_stop_m").value
        )

        self.latest_pose: Optional[Pose2D] = None
        self.latest_us = [9.9] * 8
        self.latest_metrics = []
        self.latest_case = self.planner_mode

        self.flow_vx_mps = 0.0
        self.flow_distance_m = None
        self.flow_yaw_rate = 0.0
        self.flow_valid = False
        self.last_flow_time = 0.0

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

        self.get_logger().info("autopark_master ready: flow distance + ultrasonic safety")

    def on_pose(self, msg):
        self.latest_pose = msg

    def on_metrics(self, msg):
        self.latest_metrics = list(msg.data)

    def on_ultrasonic(self, msg):
        vals = list(msg.data)
        if len(vals) >= 8:
            self.latest_us = vals[:8]

    def on_flow_distance(self, msg):
        data = list(msg.data)
        if len(data) >= 6:
            self.flow_vx_mps = float(data[0])
            self.flow_distance_m = float(data[1])
            self.flow_yaw_rate = float(data[2])
            self.flow_valid = bool(data[5] > 0.5)
            self.last_flow_time = time.monotonic()

    def on_slot_info(self, msg):
        try:
            obj = json.loads(msg.data)
            case_name = str(obj.get("case", self.planner_mode)).strip().lower()

            if case_name in ("left_only", "right_only", "both_sides"):
                if case_name != self.latest_case:
                    self.get_logger().info(
                        "parking case updated from slot_info: " + case_name
                    )
                self.latest_case = case_name

        except Exception as exc:
            self.get_logger().warning("bad slot_info JSON: " + str(exc))

    def on_start_switch(self, msg):
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

        if not self.disable_ultrasonic_block:
            if self.get_ultrasonic_min_m() < self.min_clearance_m:
                self.publish_stop("blocked_by_ultrasonic_before_start")
                return

        pose = self.latest_pose

        if pose is None:
            if not self.use_default_pose_when_missing:
                self.publish_stop("no_start_pose")
                return

            pose = Pose2D()
            pose.x = self.default_start_x
            pose.y = self.default_start_y
            pose.theta = math.radians(self.default_start_yaw_deg)

            self.get_logger().warning(
                "using default pose: " + str((pose.x, pose.y, pose.theta))
            )

        yaw_deg = math.degrees(pose.theta)

        self.get_logger().info(
            "planning from pose x="
            + str(pose.x)
            + " y="
            + str(pose.y)
            + " theta_rad="
            + str(pose.theta)
            + " theta_deg="
            + str(yaw_deg)
        )

        case_name = (
            self.latest_case
            if self.latest_case in ("left_only", "right_only", "both_sides")
            else self.planner_mode
        )

        planned = plan_from_start(pose.x, pose.y, yaw_deg, case_name)
        result = result_to_dict(planned)
        motions = result.get("motions", [])

        self.get_logger().info("PLANNER FINISHED")
        self.get_logger().info("MOTIONS LEN: " + str(len(motions)))

        if not motions:
            self.plan_pub.publish(String(data=json.dumps(result)))
            self.publish_stop("planner_returned_empty_motion_sequence")
            return

        if self.latest_metrics:
            result["parking_metrics"] = self.latest_metrics

        result["control_mode"] = "flow_distance_closed_loop_ultrasonic_safety"
        result["default_drive_distance_m"] = self.default_drive_distance_m
        result["ultrasonic_stop_m"] = self.ultrasonic_stop_m

        self.plan_pub.publish(String(data=json.dumps(result)))

        self.get_logger().info(
            "plan published: case="
            + str(case_name)
            + " motions="
            + str(len(motions))
        )

        self.execute_motions(motions)

    def execute_motions(self, motions):
        self.get_logger().info("EXECUTING MOTIONS WITH FLOW DISTANCE + ULTRASONIC SAFETY")

        for i, motion in enumerate(motions):
            cmd = self.motion_to_cmd(motion)

            self.get_logger().info(
                "SEGMENT "
                + str(i + 1)
                + "/"
                + str(len(motions))
                + ": "
                + json.dumps(cmd)
            )

            # 1) Steering command first, no drive.
            steer_cmd = dict(cmd)
            steer_cmd["type"] = "drive"
            steer_cmd["gear"] = 0
            steer_cmd["speed_mps"] = 0.0
            steer_cmd["duration"] = self.steer_wait_seconds

            self.cmd_pub.publish(String(data=json.dumps(steer_cmd)))

            self.get_logger().info(
                "STEER WAIT: steer_deg="
                + str(steer_cmd["steer_deg"])
                + " wait="
                + str(self.steer_wait_seconds)
                + " sec"
            )

            time.sleep(self.steer_wait_seconds)

            # 2) Drive command.
            # Duration is long hold only. Normal stop is flow target or ultrasonic safety.
            target_dist_m = float(cmd.get("target_dist_m", self.default_drive_distance_m))

            drive_cmd = dict(cmd)
            drive_cmd["duration"] = self.esp32_drive_hold_seconds

            baseline_dist = self.flow_distance_m

            self.get_logger().info(
                "DRIVE START: gear="
                + str(drive_cmd["gear"])
                + " speed="
                + str(drive_cmd["speed_mps"])
                + " steer="
                + str(drive_cmd["steer_deg"])
                + " target_dist_m="
                + str(target_dist_m)
                + " esp32_hold_s="
                + str(self.esp32_drive_hold_seconds)
                + " baseline_flow="
                + str(baseline_dist)
                + " ultrasonic_min_m="
                + str(round(self.get_ultrasonic_min_m(), 3))
            )

            self.cmd_pub.publish(String(data=json.dumps(drive_cmd)))

            # Wait a little after drive command.
            # This lets ESP32 finish steerReady gating and avoids counting steering vibration.
            time.sleep(1.0)

            # Reset baseline AFTER drive command is active.
            baseline_dist = self.flow_distance_m

            if self.use_flow_distance_control and baseline_dist is not None:
                stop_reason = self.wait_until_stop_condition(
                    baseline_dist=baseline_dist,
                    target_dist_m=target_dist_m,
                )
                self.publish_stop(stop_reason)
            else:
                self.get_logger().warning(
                    "flow distance unavailable, fallback to default_motion_seconds"
                )
                time.sleep(self.default_motion_seconds)
                self.publish_stop("segment_pause_fallback_time")

            time.sleep(self.pause_between_commands)

        self.publish_stop("parking_sequence_done")
        self.get_logger().info("PARKING SEQUENCE DONE")

    def wait_until_stop_condition(self, baseline_dist, target_dist_m):
        last_log_t = 0.0

        while rclpy.ok():
            now = time.monotonic()

            # Ultrasonic safety stop.
            # This replaces short duration timeout.
            if self.drive_ultrasonic_safety and self.ultrasonic_blocked_now():
                dmin = self.get_ultrasonic_min_m()
                self.get_logger().warning(
                    "ultrasonic safety stop: min_clearance_m="
                    + str(round(dmin, 3))
                    + " limit="
                    + str(round(self.ultrasonic_stop_m, 3))
                )
                return "ultrasonic_safety_stop"

            if self.flow_distance_m is None:
                time.sleep(0.05)
                continue

            flow_age = now - self.last_flow_time
            moved = max(0.0, float(self.flow_distance_m) - float(baseline_dist))

            if self.flow_valid_required:
                if (not self.flow_valid) or flow_age > 1.0:
                    if now - last_log_t > 1.0:
                        self.get_logger().warning(
                            "waiting for valid flow: valid="
                            + str(self.flow_valid)
                            + " age="
                            + str(round(flow_age, 2))
                        )
                        last_log_t = now
                    time.sleep(0.05)
                    continue

            if now - last_log_t > 0.5:
                self.get_logger().info(
                    "FLOW DRIVE: moved="
                    + str(round(moved, 3))
                    + " / target="
                    + str(round(target_dist_m, 3))
                    + " vx="
                    + str(round(self.flow_vx_mps, 3))
                    + " valid="
                    + str(self.flow_valid)
                    + " ultrasonic_min_m="
                    + str(round(self.get_ultrasonic_min_m(), 3))
                )
                last_log_t = now

            if moved + self.flow_distance_tolerance_m >= target_dist_m:
                self.get_logger().info(
                    "target distance reached: moved="
                    + str(round(moved, 3))
                    + " target="
                    + str(round(target_dist_m, 3))
                )
                return "target_distance_reached"

            time.sleep(0.05)

        return "ros_shutdown"

    def get_ultrasonic_min_m(self):
        vals_m = []

        for v in self.latest_us:
            try:
                x = float(v)
            except Exception:
                continue

            # Ignore invalid readings.
            if x <= 0.0:
                continue

            # If value looks like cm, convert to m.
            # Example: 35 means 35 cm = 0.35 m.
            # If value is already m, keep it.
            if x > 3.0:
                x = x / 100.0

            # Ignore impossible very large fallback values.
            if x > 5.0:
                continue

            vals_m.append(x)

        if not vals_m:
            return 9.9

        return min(vals_m)

    def ultrasonic_blocked_now(self):
        dmin = self.get_ultrasonic_min_m()
        return dmin < self.ultrasonic_stop_m

    def motion_to_cmd(self, motion):
        gear = -1
        steer_deg = 0.0
        speed_mps = self.speed_scale
        target_dist_m = self.default_drive_distance_m

        if isinstance(motion, dict):
            if "gear" in motion:
                gear = int(motion["gear"])

            if "steer_deg" in motion:
                steer_deg = float(motion["steer_deg"])

            if "speed_mps" in motion:
                speed_mps = abs(float(motion["speed_mps"]))

            for key in ("target_dist_m", "dist_m", "distance_m", "dist"):
                if key in motion:
                    try:
                        target_dist_m = abs(float(motion[key]))
                    except Exception:
                        pass
                    break

        # Real-car minimum distance.
        # Planner segment distances can be too small for the physical car.

        speed_mps = min(abs(speed_mps), self.speed_scale)

        if gear == 0:
            speed_mps = 0.0

        return {
            "type": "drive",
            "gear": gear,
            "speed_mps": speed_mps,
            "steer_deg": steer_deg,
            "target_dist_m": target_dist_m,

            # Long hold only. Not used as normal stop.
            # Normal stop = flow target reached or ultrasonic safety stop.
            "duration": self.esp32_drive_hold_seconds,
        }

    def publish_stop(self, reason):
        cmd = {
            "type": "stop",
            "reason": reason,
        }
        self.cmd_pub.publish(String(data=json.dumps(cmd)))
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
