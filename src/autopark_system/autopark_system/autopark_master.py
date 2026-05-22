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

            ("min_clearance_m", 0.12),
            ("disable_ultrasonic_block", True),

            ("planner_mode", "both_sides"),
            ("use_default_pose_when_missing", True),
            ("default_start_x", 0.0),
            ("default_start_y", 0.70),
            ("default_start_yaw_deg", 180.0),

            ("speed_scale", 0.07),

            # fallback only
            ("default_motion_seconds", 4.0),

            # steering is still time-based for now
            ("steer_wait_seconds", 30.0),
            ("pause_between_commands", 1.0),

            # closed-loop drive distance
            ("use_flow_distance_control", True),
            ("default_drive_distance_m", 0.20),
            ("drive_timeout_seconds", 12.0),
            ("flow_valid_required", True),
            ("flow_distance_tolerance_m", 0.01),
        ]:
            self.declare_parameter(name, default)

        self.min_clearance_m = float(self.get_parameter("min_clearance_m").value)
        self.disable_ultrasonic_block = bool(self.get_parameter("disable_ultrasonic_block").value)

        self.planner_mode = str(self.get_parameter("planner_mode").value)
        self.use_default_pose_when_missing = bool(self.get_parameter("use_default_pose_when_missing").value)

        self.default_start_x = float(self.get_parameter("default_start_x").value)
        self.default_start_y = float(self.get_parameter("default_start_y").value)
        self.default_start_yaw_deg = float(self.get_parameter("default_start_yaw_deg").value)

        self.speed_scale = float(self.get_parameter("speed_scale").value)
        self.default_motion_seconds = float(self.get_parameter("default_motion_seconds").value)
        self.steer_wait_seconds = float(self.get_parameter("steer_wait_seconds").value)
        self.pause_between_commands = float(self.get_parameter("pause_between_commands").value)

        self.use_flow_distance_control = bool(self.get_parameter("use_flow_distance_control").value)
        self.default_drive_distance_m = float(self.get_parameter("default_drive_distance_m").value)
        self.drive_timeout_seconds = float(self.get_parameter("drive_timeout_seconds").value)
        self.flow_valid_required = bool(self.get_parameter("flow_valid_required").value)
        self.flow_distance_tolerance_m = float(self.get_parameter("flow_distance_tolerance_m").value)

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

        self.create_subscription(Bool, self.get_parameter("start_switch_topic").value, self.on_start_switch, 10)
        self.create_subscription(Pose2D, self.get_parameter("pose_topic").value, self.on_pose, 10)
        self.create_subscription(Float32MultiArray, self.get_parameter("parking_metrics_topic").value, self.on_metrics, 10)
        self.create_subscription(Float32MultiArray, self.get_parameter("ultrasonic_topic").value, self.on_ultrasonic, 10)
        self.create_subscription(String, self.get_parameter("slot_topic").value, self.on_slot_info, 10)
        self.create_subscription(Float32MultiArray, self.get_parameter("flow_distance_topic").value, self.on_flow_distance, 20)

        self.cmd_pub = self.create_publisher(String, self.get_parameter("command_topic").value, 10)
        self.plan_pub = self.create_publisher(String, self.get_parameter("plan_topic").value, 10)

        self.get_logger().info("autopark_master ready with flow-distance control")

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
                    self.get_logger().info("parking case updated from slot_info: " + case_name)
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
            if self.latest_us and min(self.latest_us) < self.min_clearance_m:
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

        result["control_mode"] = "flow_distance_closed_loop"
        result["default_drive_distance_m"] = self.default_drive_distance_m

        self.plan_pub.publish(String(data=json.dumps(result)))

        self.get_logger().info(
            "plan published: case="
            + str(case_name)
            + " motions="
            + str(len(motions))
        )

        self.execute_motions(motions)

    def execute_motions(self, motions):
        self.get_logger().info("EXECUTING MOTIONS WITH FLOW DISTANCE")

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

            target_dist_m = float(cmd.get("target_dist_m", self.default_drive_distance_m))
            timeout_s = float(cmd.get("timeout", self.drive_timeout_seconds))

            drive_cmd = dict(cmd)
            drive_cmd["duration"] = timeout_s + 1.0

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
                + " timeout_s="
                + str(timeout_s)
                + " baseline_flow="
                + str(baseline_dist)
            )

            self.cmd_pub.publish(String(data=json.dumps(drive_cmd)))

            if self.use_flow_distance_control and baseline_dist is not None:
                reached = self.wait_until_distance_reached(
                    baseline_dist=baseline_dist,
                    target_dist_m=target_dist_m,
                    timeout_s=timeout_s,
                )

                if reached:
                    self.publish_stop("target_distance_reached")
                else:
                    self.publish_stop("drive_timeout_or_flow_invalid")
            else:
                self.get_logger().warning(
                    "flow distance unavailable, fallback to default_motion_seconds"
                )
                time.sleep(self.default_motion_seconds)
                self.publish_stop("segment_pause_fallback_time")

            time.sleep(self.pause_between_commands)

        self.publish_stop("parking_sequence_done")
        self.get_logger().info("PARKING SEQUENCE DONE")

    def wait_until_distance_reached(self, baseline_dist, target_dist_m, timeout_s):
        start_t = time.monotonic()
        last_log_t = 0.0

        while rclpy.ok():
            now = time.monotonic()
            elapsed = now - start_t

            if elapsed >= timeout_s:
                self.get_logger().warning(
                    "drive timeout: elapsed="
                    + str(round(elapsed, 2))
                    + " target="
                    + str(target_dist_m)
                )
                return False

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
                )
                last_log_t = now

            if moved + self.flow_distance_tolerance_m >= target_dist_m:
                self.get_logger().info(
                    "target distance reached: moved="
                    + str(round(moved, 3))
                    + " target="
                    + str(round(target_dist_m, 3))
                )
                return True

            time.sleep(0.05)

        return False

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
            # Planner segment distances can be too small for the physical car,
            # so force a safe minimum drive distance for testing.
            if gear != 0:
                target_dist_m = max(target_dist_m, self.default_drive_distance_m)

        speed_mps = min(abs(speed_mps), self.speed_scale)

        if gear == 0:
            speed_mps = 0.0

        return {
            "type": "drive",
            "gear": gear,
            "speed_mps": speed_mps,
            "steer_deg": steer_deg,
            "target_dist_m": target_dist_m,
            "duration": self.drive_timeout_seconds + 1.0,
            "timeout": self.drive_timeout_seconds,
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
