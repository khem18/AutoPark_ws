import json
import math
import threading
import time
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
            ("start_switch_topic",           "/autopark/start_switch"),
            ("pose_topic",                   "/autopark/start_pose"),
            ("parking_metrics_topic",        "/parking_metrics"),
            ("ultrasonic_topic",             "/autopark/ultrasonic"),
            ("command_topic",                "/autopark/cmd_json"),
            ("plan_topic",                   "/autopark/plan_result"),

            ("min_clearance_m",              0.12),
            ("disable_ultrasonic_block",     True),

            ("planner_mode",                 "both_sides"),
            ("use_default_pose_when_missing", True),
            ("default_start_x",              0.0),
            ("default_start_y",              0.70),
            ("default_start_yaw_deg",        180.0),

            ("speed_scale",                  0.12),
            ("default_motion_seconds",       1.0),
            ("pause_between_commands",       0.30),
            ("steer_wait_seconds",           4.0),

            # ── Ultrasonic centering ──────────────────────────────────
            ("us_center_enable",             True),

            # Lateral (left / right wall) sensors
            ("us_left_idx",                  1),
            ("us_right_idx",                 2),
            ("us_deadband_m",                0.060),
            ("us_steer_gain",                150.0),
            ("us_max_steer_deg",             15.0),
            ("us_correction_dist_m",         0.12),
            ("us_correction_speed_mps",      0.06),
            ("us_max_attempts",              3),

            # Rear wall (depth) sensor
            ("us_rear_idx",                  6),
            ("us_rear_target_m",             0.20),   # target 20 cm from wall
            ("us_rear_tolerance_m",          0.06),   # ±6 cm  →  14–26 cm is OK
            ("us_depth_max_step_m",          0.04),   # max reverse per depth step
            ("us_depth_max_attempts",        4),

            # Stop timing
            # Send stop this many seconds BEFORE the calculated end of motion
            # to compensate for ROS→serial→ESP32 latency (~100 ms) + decel.
            ("us_stop_buffer_s",             0.20),
            # After stop, wait this long before reading sensors or sending next cmd.
            ("us_stop_settle_s",             0.60),
            # Steer settle wait used only during correction moves (faster than main).
            ("us_steer_wait_s",              2.0),
        ]:
            self.declare_parameter(name, default)

        # ── existing params ───────────────────────────────────────────
        self.min_clearance_m           = float(self.get_parameter("min_clearance_m").value)
        self.disable_ultrasonic_block  = bool(self.get_parameter("disable_ultrasonic_block").value)
        self.planner_mode              = str(self.get_parameter("planner_mode").value)
        self.use_default_pose_when_missing = bool(
            self.get_parameter("use_default_pose_when_missing").value)
        self.default_start_x           = float(self.get_parameter("default_start_x").value)
        self.default_start_y           = float(self.get_parameter("default_start_y").value)
        self.default_start_yaw_deg     = float(self.get_parameter("default_start_yaw_deg").value)
        self.speed_scale               = float(self.get_parameter("speed_scale").value)
        self.default_motion_seconds    = float(self.get_parameter("default_motion_seconds").value)
        self.pause_between_commands    = float(self.get_parameter("pause_between_commands").value)
        self.steer_wait_seconds        = float(self.get_parameter("steer_wait_seconds").value)

        # ── ultrasonic params ─────────────────────────────────────────
        self.us_center_enable          = bool(self.get_parameter("us_center_enable").value)
        self.us_left_idx               = int(self.get_parameter("us_left_idx").value)
        self.us_right_idx              = int(self.get_parameter("us_right_idx").value)
        self.us_deadband_m             = float(self.get_parameter("us_deadband_m").value)
        self.us_steer_gain             = float(self.get_parameter("us_steer_gain").value)
        self.us_max_steer_deg          = float(self.get_parameter("us_max_steer_deg").value)
        self.us_correction_dist_m      = float(self.get_parameter("us_correction_dist_m").value)
        self.us_correction_speed_mps   = float(self.get_parameter("us_correction_speed_mps").value)
        self.us_max_attempts           = int(self.get_parameter("us_max_attempts").value)

        self.us_rear_idx               = int(self.get_parameter("us_rear_idx").value)
        self.us_rear_target_m          = float(self.get_parameter("us_rear_target_m").value)
        self.us_rear_tolerance_m       = float(self.get_parameter("us_rear_tolerance_m").value)
        self.us_depth_max_step_m       = float(self.get_parameter("us_depth_max_step_m").value)
        self.us_depth_max_attempts     = int(self.get_parameter("us_depth_max_attempts").value)

        self.us_stop_buffer_s          = float(self.get_parameter("us_stop_buffer_s").value)
        self.us_stop_settle_s          = float(self.get_parameter("us_stop_settle_s").value)
        self.us_steer_wait_s           = float(self.get_parameter("us_steer_wait_s").value)

        # ── state ─────────────────────────────────────────────────────
        self.latest_pose: Optional[Pose2D] = None
        self.latest_us   = [9.9] * 8
        self.latest_metrics = []
        self.busy = False

        self.create_subscription(
            Bool, self.get_parameter("start_switch_topic").value, self.on_start_switch, 10)
        self.create_subscription(
            Pose2D, self.get_parameter("pose_topic").value, self.on_pose, 10)
        self.create_subscription(
            Float32MultiArray, self.get_parameter("parking_metrics_topic").value,
            self.on_metrics, 10)
        self.create_subscription(
            Float32MultiArray, self.get_parameter("ultrasonic_topic").value,
            self.on_ultrasonic, 10)

        self.cmd_pub  = self.create_publisher(
            String, self.get_parameter("command_topic").value, 10)
        self.plan_pub = self.create_publisher(
            String, self.get_parameter("plan_topic").value, 10)

        self.get_logger().info("autopark_master ready")

    # ── Subscribers ───────────────────────────────────────────────────

    def on_pose(self, msg):
        self.latest_pose = msg

    def on_metrics(self, msg):
        self.latest_metrics = list(msg.data)

    def on_ultrasonic(self, msg):
        vals = list(msg.data)
        if len(vals) >= 8:
            self.latest_us = vals[:8]

    def on_start_switch(self, msg):
        self.get_logger().info("START SWITCH CALLBACK: " + str(msg.data))
        if not msg.data:
            return
        if self.busy:
            self.get_logger().warning("ignored start switch because busy")
            return
        self.busy = True
        # Launch in a daemon thread so the ROS executor keeps spinning.
        # Without this, on_ultrasonic() never fires during parking because
        # rclpy.spin() is blocked waiting for this callback to return.
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

    # ── Main parking sequence ─────────────────────────────────────────

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
            "planning from pose x=" + str(pose.x)
            + " y=" + str(pose.y)
            + " theta_deg=" + str(yaw_deg))

        planned = plan_from_start(pose.x, pose.y, yaw_deg, self.planner_mode)
        result  = result_to_dict(planned)
        motions = result.get("motions", [])

        self.get_logger().info("PLANNER FINISHED — motions=" + str(len(motions)))

        if not motions:
            self.plan_pub.publish(String(data=json.dumps(result)))
            self._led("red")
            self.publish_stop("planner_returned_empty_motion_sequence")
            return

        if self.latest_metrics:
            result["parking_metrics"] = self.latest_metrics

        self.plan_pub.publish(String(data=json.dumps(result)))

        # ── Step 1: main planner motion sequence ──────────────────────
        self.execute_motions(motions)

        if self.us_center_enable:
            # ── Step 2: lateral centering (left / right walls) ────────
            self._run_lateral_centering()

            # ── Step 3: depth correction (rear wall 20 ± 6 cm) ───────
            self._run_depth_correction()

        self._led("green")
        self.publish_stop("parking_complete")

    # ── Planner motion execution (unchanged) ──────────────────────────

    def execute_motions(self, motions):
        self.get_logger().info("EXECUTING MOTIONS")

        for i, motion in enumerate(motions):
            cmd = self.motion_to_cmd(motion)

            self.get_logger().info(
                "CMD " + str(i + 1) + "/" + str(len(motions))
                + ": " + json.dumps(cmd))

            # Pre-steer with wheels stopped
            steer_cmd = dict(cmd)
            steer_cmd["speed_mps"] = 0.0
            steer_cmd["gear"]      = 0
            steer_cmd["type"]      = "drive"
            self.cmd_pub.publish(String(data=json.dumps(steer_cmd)))

            self.get_logger().info(
                "WAIT STEER: steer_deg=" + str(steer_cmd["steer_deg"])
                + " wait=" + str(self.steer_wait_seconds) + "s")
            time.sleep(self.steer_wait_seconds)

            # Drive segment
            self.cmd_pub.publish(String(data=json.dumps(cmd)))
            duration = float(cmd.get("duration", self.default_motion_seconds))
            time.sleep(max(0.10, duration))

            self.publish_stop("segment_pause")
            time.sleep(self.pause_between_commands)

        self.publish_stop("parking_sequence_done")
        self.get_logger().info("PARKING SEQUENCE DONE")

    # ── Step 2: lateral centering ─────────────────────────────────────

    def _run_lateral_centering(self):
        """
        Equalise left and right wall distances.

        error = (left_dist - right_dist) / 2
          positive → car biased right → need to shift LEFT
          negative → car biased left  → need to shift RIGHT

        When reversing with +steer (left turn command) the rear swings
        right, shifting the body left. Sign of correction_steer = sign
        of error.
        """
        self.get_logger().info("LATERAL CENTERING START")
        time.sleep(0.40)  # let car settle after main sequence

        for attempt in range(self.us_max_attempts):
            left_dist  = self.latest_us[self.us_left_idx]
            right_dist = self.latest_us[self.us_right_idx]

            self.get_logger().info(
                f"[LATERAL {attempt + 1}/{self.us_max_attempts}] "
                f"left={left_dist:.3f}m  right={right_dist:.3f}m")

            if left_dist > 2.0 or right_dist > 2.0:
                self.get_logger().warning(
                    "lateral sensor no-reading — skipping centering")
                break

            error = (left_dist - right_dist) / 2.0

            if abs(error) < self.us_deadband_m:
                self.get_logger().info(
                    f"[LATERAL] error={error*100:.1f} cm < deadband "
                    f"{self.us_deadband_m*100:.0f} cm → OK")
                break

            raw_steer = error * self.us_steer_gain
            correction_steer = max(-self.us_max_steer_deg,
                                   min(self.us_max_steer_deg, raw_steer))

            self.get_logger().info(
                f"[LATERAL] error={error*100:.1f} cm → "
                f"steer={correction_steer:.1f} deg")

            # Phase 1: forward arc (opposite steer to reverse phase)
            # Using -correction_steer on exit and +correction_steer on
            # re-entry creates an S-maneuver: both arcs shift the car
            # laterally in the same direction, doubling the correction
            # compared to a straight exit while the heading changes from
            # each arc partially cancel each other.
            self.get_logger().info(
                f"[LATERAL] phase 1: forward arc steer={-correction_steer:.1f}")
            self._send_correction_move(
                gear=1, steer_deg=-correction_steer,
                dist_m=self.us_correction_dist_m)

            # Phase 2: reverse arc (same steer sign as before)
            self.get_logger().info(
                f"[LATERAL] phase 2: reverse arc steer={correction_steer:.1f}")
            self._send_correction_move(
                gear=-1, steer_deg=correction_steer,
                dist_m=self.us_correction_dist_m)

            time.sleep(0.30)

        self.get_logger().info("LATERAL CENTERING DONE")

    # ── Step 3: depth correction (rear wall) ─────────────────────────

    def _run_depth_correction(self):
        """
        Adjust depth so the rear wall distance is us_rear_target_m ± us_rear_tolerance_m
        (default 20 ± 6 cm → acceptable window 14 cm … 26 cm).

        Uses straight-only moves (steer=0) and caps each reverse step to
        us_depth_max_step_m so the car cannot overshoot into the wall even
        if the stop buffer is slightly off.

        Stop-timing safety
        ──────────────────
        _send_correction_move sends the stop command us_stop_buffer_s
        seconds BEFORE the calculated end of travel.  At 0.06 m/s that
        means the car stops ~1.2 cm short of the intended position.
        For the last reverse step this undershoot is intentional — it
        provides a physical safety margin against the wall.
        """
        self.get_logger().info("DEPTH CORRECTION START")
        time.sleep(0.40)

        target = self.us_rear_target_m
        tol    = self.us_rear_tolerance_m
        lo     = target - tol   # 0.14 m
        hi     = target + tol   # 0.26 m

        for attempt in range(self.us_depth_max_attempts):
            rear_dist = self.latest_us[self.us_rear_idx]

            self.get_logger().info(
                f"[DEPTH {attempt + 1}/{self.us_depth_max_attempts}] "
                f"rear={rear_dist:.3f}m  target={target:.2f}m  "
                f"window=[{lo:.2f},{hi:.2f}]m")

            if rear_dist > 2.0:
                self.get_logger().warning(
                    "rear sensor no-reading — skipping depth correction")
                break

            if lo <= rear_dist <= hi:
                self.get_logger().info(
                    f"[DEPTH] {rear_dist*100:.1f} cm within "
                    f"{target*100:.0f}±{tol*100:.0f} cm → OK")
                break

            if rear_dist < lo:
                # Too close — drive forward to create space
                move_dist = min(lo - rear_dist + 0.02, self.us_correction_dist_m)
                self.get_logger().info(
                    f"[DEPTH] too close ({rear_dist*100:.1f} cm < {lo*100:.0f} cm) "
                    f"→ forward {move_dist*100:.1f} cm")
                self._send_correction_move(
                    gear=1, steer_deg=0.0, dist_m=move_dist)

            else:
                # Too far — reverse toward wall in a small capped step.
                # Each step is at most us_depth_max_step_m to prevent
                # wall collision even if the stop command is slightly late.
                move_dist = min(rear_dist - target, self.us_depth_max_step_m)
                self.get_logger().info(
                    f"[DEPTH] too far ({rear_dist*100:.1f} cm > {hi*100:.0f} cm) "
                    f"→ reverse {move_dist*100:.1f} cm (capped at "
                    f"{self.us_depth_max_step_m*100:.0f} cm)")
                self._send_correction_move(
                    gear=-1, steer_deg=0.0, dist_m=move_dist)

            time.sleep(0.30)

        final = self.latest_us[self.us_rear_idx]
        self.get_logger().info(
            f"DEPTH CORRECTION DONE — final rear dist={final*100:.1f} cm")

    # ── Low-level correction move ─────────────────────────────────────

    def _send_correction_move(self, gear: int, steer_deg: float, dist_m: float):
        """
        Execute one correction arc/straight segment.

        Stop-buffer fix
        ───────────────
        The stop command is sent us_stop_buffer_s seconds BEFORE the
        calculated end of motion.  This compensates for the chain:
            Python sleep ends
            → ROS publish
            → serial_bridge.on_cmd()
            → serial.write()
            → ESP32 reads on next 2 ms loop tick
            → stopDriveMotor() called
        Total latency is typically 50–150 ms.  At us_correction_speed_mps
        = 0.06 m/s the car travels 0.9–1.5 cm during that window.
        Stopping 0.20 s early keeps overshoot well under 1 cm.

        After stop we wait us_stop_settle_s (default 0.60 s) before
        returning so the car is fully stationary when the caller reads
        sensors or sends the next pre-steer command.
        """
        speed    = self.us_correction_speed_mps
        duration = dist_m / max(speed, 0.01)

        # ── Pre-steer (wheels move, body stationary) ──────────────────
        steer_cmd = {
            "type":      "drive",
            "gear":      0,
            "speed_mps": 0.0,
            "steer_deg": steer_deg,
        }
        self.cmd_pub.publish(String(data=json.dumps(steer_cmd)))
        time.sleep(self.us_steer_wait_s)

        # ── Drive ─────────────────────────────────────────────────────
        drive_cmd = {
            "type":      "drive",
            "gear":      gear,
            "speed_mps": speed,
            "steer_deg": steer_deg,
        }
        self.cmd_pub.publish(String(data=json.dumps(drive_cmd)))

        # Stop early to compensate for serial + ESP32 latency
        effective_sleep = max(0.05, duration - self.us_stop_buffer_s)
        time.sleep(effective_sleep)

        # ── Stop + settle ─────────────────────────────────────────────
        self.publish_stop("correction_move_done")
        time.sleep(self.us_stop_settle_s)  # wait for car to be stationary

    # ── Helpers ───────────────────────────────────────────────────────

    def motion_to_cmd(self, motion):
        gear      = -1
        steer_deg = 0.0
        speed_mps = self.speed_scale
        duration  = self.default_motion_seconds

        if isinstance(motion, dict):
            if "gear"      in motion: gear      = int(motion["gear"])
            if "steer_deg" in motion: steer_deg = float(motion["steer_deg"])
            if "speed_mps" in motion: speed_mps = abs(float(motion["speed_mps"]))
            if "duration"  in motion:
                duration = float(motion["duration"])
            elif "dist_m"  in motion:
                dist = abs(float(motion["dist_m"]))
                duration = max(0.4, dist / max(speed_mps, 0.05))
            elif "dist"    in motion:
                dist = abs(float(motion["dist"]))
                duration = max(0.4, dist / max(speed_mps, 0.05))

        speed_mps = min(abs(speed_mps), self.speed_scale)
        # Old clamp max(1.2, min(duration, 2.0)) truncated the arc to only
        # 0.24 m travel. Arc needs ~15 s at speed_scale=0.12 m/s. Fixed.
        duration  = max(0.5, min(duration, 60.0))

        return {
            "type":      "drive",
            "gear":      gear,
            "speed_mps": speed_mps,
            "steer_deg": steer_deg,
            "duration":  duration,
        }

    def _led(self, color: str):
        """Send LED colour command to ESP32 via serial_bridge."""
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
