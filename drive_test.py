#!/usr/bin/env python3
"""
drive_test.py — Manual ESP32 drive tester
==========================================
Tests forward, reverse, and steer independently to identify
whether the issue is hardware (forward pins dead) or software.

HOW TO RUN:
  Terminal 1 (start serial bridge only):
    ros2 run autopark_system serial_bridge

  Terminal 2 (run this test):
    python3 drive_test.py

  OR run both together:
    ros2 run autopark_system serial_bridge &
    sleep 3 && python3 drive_test.py

WHAT IT TESTS:
  Step 1 — ARM
  Step 2 — STEER to 0° (straight), steer_active_hold=True
  Step 3 — FORWARD gear=1, speed=0.08, 3s        ← watch if car moves
  Step 4 — STOP
  Step 5 — STEER to 30°
  Step 6 — REVERSE gear=-1, speed=0.08, 3s       ← watch if car moves
  Step 7 — STOP

If step 3 doesn't move but step 6 does → forward PWM pins (32,25) are hardware problem.
If neither moves → serial/arm issue.
If both move → software bug (steer tolerance or steer_active_hold).
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import json
import time


class DriveTest(Node):
    def __init__(self):
        super().__init__('drive_test')
        self.pub = self.create_publisher(String, '/autopark/cmd_json', 10)
        self.sub = self.create_subscription(
            String, '/autopark/esp32_status', self.on_status, 10)
        self.step = 0
        self.step_start = time.monotonic()
        self.done = False
        self.timer = self.create_timer(0.05, self.tick)
        self.get_logger().info('=' * 50)
        self.get_logger().info('Drive Test Starting...')
        self.get_logger().info('Watch the car at each step!')
        self.get_logger().info('=' * 50)

    def on_status(self, msg):
        try:
            d = json.loads(msg.data)
            steer = d.get('steer_deg', '?')
            drive = d.get('drive_pwm', '?')
            gear  = d.get('auto_gear', '?')
            active = d.get('auto_command_active', '?')
            self.get_logger().info(
                f'  ESP32 → steer={steer:.1f}°  drive_pwm={drive}  gear={gear}  active={active}',
                throttle_duration_sec=0.5)
        except Exception:
            pass

    def send(self, cmd: dict, label: str):
        self.get_logger().info(f'  >> SEND [{label}]: {json.dumps(cmd)}')
        self.pub.publish(String(data=json.dumps(cmd)))

    def tick(self):
        if self.done:
            return

        elapsed = time.monotonic() - self.step_start

        # ── Step 0: ARM ──────────────────────────────────────────────────
        if self.step == 0:
            self.get_logger().info('\n[STEP 1] Sending ARM...')
            self.send({"type": "arm"}, "ARM")
            self.step = 1
            self.step_start = time.monotonic()

        # ── Step 1: wait 0.5s, then steer to 0° ─────────────────────────
        elif self.step == 1 and elapsed >= 0.5:
            self.get_logger().info('\n[STEP 2] Steering to 0° (straight)...')
            self.send({
                "type": "drive", "gear": 1, "speed_mps": 0.0,
                "steer_deg": 0.0, "duration": 8.0, "steer_active_hold": True
            }, "STEER→0°")
            self.step = 2
            self.step_start = time.monotonic()

        # ── Step 2: wait 3s for steer to settle, then drive FORWARD ──────
        elif self.step == 2 and elapsed >= 3.0:
            self.get_logger().info('\n[STEP 3] ★ FORWARD drive — WATCH IF CAR MOVES ★')
            self.get_logger().info('  gear=1  speed=0.08  steer=0°  duration=3s')
            self.send({
                "type": "drive", "gear": 1, "speed_mps": 0.08,
                "steer_deg": 0.0, "duration": 5.0, "steer_active_hold": True
            }, "DRIVE FORWARD")
            self.step = 3
            self.step_start = time.monotonic()

        # ── Step 3: wait 3s then STOP ────────────────────────────────────
        elif self.step == 3 and elapsed >= 3.0:
            self.get_logger().info('\n[STEP 4] STOP')
            self.send({"type": "stop", "reason": "test_pause"}, "STOP")
            self.step = 4
            self.step_start = time.monotonic()

        # ── Step 4: wait 1s, then steer to 30° ──────────────────────────
        elif self.step == 4 and elapsed >= 1.0:
            self.get_logger().info('\n[STEP 5] Steering to 30°...')
            self.send({
                "type": "drive", "gear": -1, "speed_mps": 0.0,
                "steer_deg": 30.0, "duration": 8.0, "steer_active_hold": False
            }, "STEER→30°")
            self.step = 5
            self.step_start = time.monotonic()

        # ── Step 5: wait 3s then drive REVERSE ───────────────────────────
        elif self.step == 5 and elapsed >= 3.0:
            self.get_logger().info('\n[STEP 6] ★ REVERSE drive — WATCH IF CAR MOVES ★')
            self.get_logger().info('  gear=-1  speed=0.08  steer=30°  duration=3s')
            self.send({
                "type": "drive", "gear": -1, "speed_mps": 0.08,
                "steer_deg": 30.0, "duration": 5.0, "steer_active_hold": False
            }, "DRIVE REVERSE")
            self.step = 6
            self.step_start = time.monotonic()

        # ── Step 6: wait 3s then STOP ALL ────────────────────────────────
        elif self.step == 6 and elapsed >= 3.0:
            self.get_logger().info('\n[STEP 7] STOP ALL — test complete')
            self.send({"type": "stop", "reason": "clear"}, "STOP ALL")
            self.get_logger().info('\n' + '=' * 50)
            self.get_logger().info('RESULTS:')
            self.get_logger().info('  If FORWARD moved  → software bug (already being fixed)')
            self.get_logger().info('  If FORWARD silent + REVERSE moved → HARDWARE: pins 32/25 dead')
            self.get_logger().info('  If neither moved  → serial_bridge not running or not connected')
            self.get_logger().info('=' * 50)
            self.done = True


def main():
    rclpy.init()
    node = DriveTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
