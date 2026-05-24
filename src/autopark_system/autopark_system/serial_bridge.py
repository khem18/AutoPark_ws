"""
serial_bridge.py  — v_next
Adds two new published topics from the ESP32 drive status JSON:
  /autopark/esp32_steer_ready   Bool   (steer_ready field)
  /autopark/esp32_status        String (full JSON, for debug/logging)
"""

import json
import time
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray, Bool

try:
    import serial
except ImportError:
    serial = None


class SerialBridge(Node):
    def __init__(self):
        super().__init__('serial_bridge')

        for name, default in [
            ('command_topic',            '/autopark/cmd_json'),
            ('ultrasonic_topic',         '/autopark/ultrasonic'),
            ('start_switch_topic',       '/autopark/start_switch'),
            # NEW
            ('esp32_steer_ready_topic',  '/autopark/esp32_steer_ready'),
            ('esp32_status_topic',       '/autopark/esp32_status'),

            ('drive_port',    '/dev/ttyUSB0'),
            ('us_front_port', ''),
            ('us_rear_port',  ''),
            ('us_all_port',   '/dev/ttyUSB1'),
            ('baud',          115200),
            ('debug_serial',  False),
        ]:
            self.declare_parameter(name, default)

        def gp(n):
            return self.get_parameter(n).value

        self.command_topic           = str(gp('command_topic'))
        self.ultrasonic_topic        = str(gp('ultrasonic_topic'))
        self.start_switch_topic      = str(gp('start_switch_topic'))
        self.esp32_steer_ready_topic = str(gp('esp32_steer_ready_topic'))
        self.esp32_status_topic      = str(gp('esp32_status_topic'))
        self.drive_port   = str(gp('drive_port'))
        self.us_front_port= str(gp('us_front_port'))
        self.us_rear_port = str(gp('us_rear_port'))
        self.us_all_port  = str(gp('us_all_port'))
        self.baud         = int(gp('baud'))
        self.debug_serial = bool(gp('debug_serial'))

        self.drive    = self._open(self.drive_port,    self.baud)
        self.us_front = self._open(self.us_front_port, self.baud)
        self.us_rear  = self._open(self.us_rear_port,  self.baud)
        self.us_all   = self._open(self.us_all_port,   self.baud)

        self.drive_buf     = ''
        self.us_front_buf  = ''
        self.us_rear_buf   = ''
        self.us_all_buf    = ''

        self.last_start_switch = None
        self.last_ultra        = None

        self.create_subscription(String, self.command_topic, self.on_cmd, 10)

        self.us_pub           = self.create_publisher(Float32MultiArray, self.ultrasonic_topic,        10)
        self.start_pub        = self.create_publisher(Bool,              self.start_switch_topic,      10)
        # NEW publishers
        self.steer_ready_pub  = self.create_publisher(Bool,   self.esp32_steer_ready_topic, 20)
        self.esp32_status_pub = self.create_publisher(String, self.esp32_status_topic,      10)

        self.timer = self.create_timer(0.05, self.poll_serial)

    # ------------------------------------------------------------------

    def _open(self, port: str, baud: int):
        if serial is None:
            self.get_logger().error('pyserial not installed')
            return None
        if not port:
            return None
        try:
            ser = serial.Serial(port=port, baudrate=baud, timeout=0.05, write_timeout=0.5)
            time.sleep(2.0)
            self.get_logger().info('OPENED serial {} @ {}'.format(port, baud))
            return ser
        except Exception as exc:
            self.get_logger().warning('cannot open {}: {}'.format(port, exc))
            return None

    def on_cmd(self, msg: String):
        raw = msg.data.strip()
        if self.drive is None:
            self.get_logger().error('drive serial NOT open — command not sent')
            return
        try:
            line = raw + '\n'
            n = self.drive.write(line.encode('utf-8'))
            self.drive.flush()
            self.get_logger().info('DRIVE TX {} bytes -> {}'.format(n, raw))
        except Exception as exc:
            self.get_logger().warning('drive write failed: {}'.format(exc))
            try:
                self.drive.close()
            except Exception:
                pass
            self.drive = self._open(self.drive_port, self.baud)

    def poll_serial(self):
        self._poll_drive_state()
        values = self._poll_ultrasonic_values()
        if len(values) == 8:
            m = Float32MultiArray()
            m.data = values
            self.us_pub.publish(m)
            self.last_ultra = values

    def _poll_drive_state(self):
        objs = self._read_json_objects(self.drive, 'drive_buf')

        for obj in objs:
            if self.debug_serial:
                self.get_logger().info('drive_obj: {}'.format(obj))

            # ── start_switch ──
            if 'start_switch' in obj:
                value = bool(obj['start_switch'])
                self.start_pub.publish(Bool(data=value))
                self.last_start_switch = value

            # ── NEW: steer_ready ──
            if 'steer_ready' in obj:
                steer_ready_val = bool(obj['steer_ready'])
                self.steer_ready_pub.publish(Bool(data=steer_ready_val))

            # ── NEW: full status JSON → /autopark/esp32_status ──
            # Publish every status frame so other nodes can monitor wheel angle,
            # drive pwm, etc. for logging / debugging.
            if 'mode' in obj or 'steer_deg' in obj:
                try:
                    self.esp32_status_pub.publish(String(data=json.dumps(obj)))
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Ultrasonic (unchanged from original)
    # ------------------------------------------------------------------

    def _poll_ultrasonic_values(self) -> List[float]:
        if self.us_all is not None:
            objs = self._read_json_objects(self.us_all, 'us_all_buf')
            vals = self._latest_ultra_from_objs(objs, expected_count=8)
            if len(vals) == 8:
                return vals
        vals_front = []
        vals_rear  = []
        if self.us_front is not None:
            objs_front = self._read_json_objects(self.us_front, 'us_front_buf')
            vals_front = self._latest_ultra_from_objs(objs_front, expected_count=4)
        if self.us_rear is not None:
            objs_rear = self._read_json_objects(self.us_rear, 'us_rear_buf')
            vals_rear = self._latest_ultra_from_objs(objs_rear, expected_count=4)
        if len(vals_front) == 4 and len(vals_rear) == 4:
            return vals_front + vals_rear
        return []

    def _latest_ultra_from_objs(self, objs, expected_count):
        latest = []
        for obj in objs:
            vals = self._parse_ultra_obj(obj, expected_count)
            if len(vals) == expected_count:
                latest = vals
                if self.debug_serial:
                    self.get_logger().info('ultra_m: {}'.format(
                        [round(v, 4) for v in vals]))
        return latest

    def _parse_ultra_obj(self, obj, expected_count):
        arr = obj.get('distances_m', obj.get('distances_cm', []))
        try:
            vals = [float(x) for x in arr]
        except Exception:
            return []
        if str(obj.get('unit', 'm')).lower() == 'cm':
            vals = [v / 100.0 for v in vals]
        return vals if len(vals) == expected_count else []

    # ------------------------------------------------------------------
    # JSON parsing (unchanged)
    # ------------------------------------------------------------------

    def _read_json_objects(self, ser, buf_name: str) -> List[dict]:
        if ser is None:
            return []
        try:
            waiting = ser.in_waiting
        except Exception as exc:
            self.get_logger().warning('{} in_waiting failed: {}'.format(buf_name, exc))
            return []
        if waiting <= 0:
            return []
        try:
            chunk = ser.read(waiting).decode('utf-8', errors='ignore')
        except Exception as exc:
            self.get_logger().warning('{} read failed: {}'.format(buf_name, exc))
            return []
        if not chunk:
            return []
        old_buf  = getattr(self, buf_name)
        new_buf  = old_buf + chunk
        json_texts, remainder = self._extract_complete_jsons(new_buf)
        setattr(self, buf_name, remainder)
        objs = []
        for text in json_texts:
            try:
                objs.append(json.loads(text))
            except Exception:
                if self.debug_serial:
                    self.get_logger().warning('bad json skipped: {}'.format(text))
        return objs

    def _extract_complete_jsons(self, s: str) -> Tuple[List[str], str]:
        objs = []
        start = None
        depth = 0
        in_string = False
        escape = False
        last_consumed = 0
        for i, ch in enumerate(s):
            if in_string:
                if escape:       escape = False
                elif ch == '\\': escape = True
                elif ch == '"':  in_string = False
                continue
            if ch == '"':
                in_string = True; continue
            if ch == '{':
                if depth == 0: start = i
                depth += 1; continue
            if ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        objs.append(s[start:i + 1])
                        last_consumed = i + 1
                        start = None
        remainder = s[last_consumed:]
        if len(remainder) > 4096:
            remainder = remainder[-1024:]
        return objs, remainder


def main(args=None):
    rclpy.init(args=args)
    node = SerialBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
