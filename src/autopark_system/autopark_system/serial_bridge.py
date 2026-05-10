import json
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
            ('command_topic', '/autopark/cmd_json'),
            ('ultrasonic_topic', '/autopark/ultrasonic'),
            ('start_switch_topic', '/autopark/start_switch'),
            ('drive_port', '/dev/ttyUSB0'),
            ('us_front_port', ''),
            ('us_rear_port', ''),
            ('us_all_port', '/dev/ttyUSB1'),
            ('baud', 115200),
            ('debug_serial', False),
        ]:
            self.declare_parameter(name, default)

        self.command_topic = str(self.get_parameter('command_topic').value)
        self.ultrasonic_topic = str(self.get_parameter('ultrasonic_topic').value)
        self.start_switch_topic = str(self.get_parameter('start_switch_topic').value)
        self.drive_port = str(self.get_parameter('drive_port').value)
        self.us_front_port = str(self.get_parameter('us_front_port').value)
        self.us_rear_port = str(self.get_parameter('us_rear_port').value)
        self.us_all_port = str(self.get_parameter('us_all_port').value)
        self.baud = int(self.get_parameter('baud').value)
        self.debug_serial = bool(self.get_parameter('debug_serial').value)

        self.drive = self._open(self.drive_port, self.baud)
        self.us_front = self._open(self.us_front_port, self.baud)
        self.us_rear = self._open(self.us_rear_port, self.baud)
        self.us_all = self._open(self.us_all_port, self.baud)

        self.drive_buf = ''
        self.us_front_buf = ''
        self.us_rear_buf = ''
        self.us_all_buf = ''

        self.last_start_switch = None
        self.last_ultra = None

        self.create_subscription(String, self.command_topic, self.on_cmd, 10)
        self.us_pub = self.create_publisher(Float32MultiArray, self.ultrasonic_topic, 10)
        self.start_pub = self.create_publisher(Bool, self.start_switch_topic, 10)

        self.timer = self.create_timer(0.05, self.poll_serial)

    def _open(self, port: str, baud: int):
        if serial is None or not port:
            return None
        try:
            return serial.Serial(
                port=port,
                baudrate=baud,
                timeout=0.05,
                write_timeout=0.05
            )
        except Exception as exc:
            self.get_logger().warning(f'cannot open {port}: {exc}')
            return None

    def on_cmd(self, msg: String):
        if self.drive is None:
            return
        try:
            self.drive.write((msg.data.strip() + '\n').encode('utf-8'))
        except Exception as exc:
            self.get_logger().warning(f'drive write failed: {exc}')

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
                self.get_logger().info(f'drive_obj: {obj}')

            if 'start_switch' in obj:
                value = bool(obj['start_switch'])
                self.start_pub.publish(Bool(data=value))
                self.last_start_switch = value

    def _poll_ultrasonic_values(self) -> List[float]:
        if self.us_all is not None:
            objs = self._read_json_objects(self.us_all, 'us_all_buf')
            vals = self._latest_ultra_from_objs(objs, expected_count=8)
            if len(vals) == 8:
                return vals

        vals_front = []
        vals_rear = []

        if self.us_front is not None:
            objs_front = self._read_json_objects(self.us_front, 'us_front_buf')
            vals_front = self._latest_ultra_from_objs(objs_front, expected_count=4)

        if self.us_rear is not None:
            objs_rear = self._read_json_objects(self.us_rear, 'us_rear_buf')
            vals_rear = self._latest_ultra_from_objs(objs_rear, expected_count=4)

        if len(vals_front) == 4 and len(vals_rear) == 4:
            return vals_front + vals_rear

        return []

    def _latest_ultra_from_objs(self, objs: List[dict], expected_count: int) -> List[float]:
        latest = []
        for obj in objs:
            vals = self._parse_ultra_obj(obj, expected_count)
            if len(vals) == expected_count:
                latest = vals
                if self.debug_serial:
                    self.get_logger().info(f'ultra_vals_m: {[round(v, 4) for v in vals]}')
        return latest

    def _parse_ultra_obj(self, obj: dict, expected_count: int) -> List[float]:
        arr = obj.get('distances_m', obj.get('distances_cm', []))
        try:
            vals = [float(x) for x in arr]
        except Exception:
            return []

        unit = str(obj.get('unit', 'm')).lower()
        if unit == 'cm':
            vals = [v / 100.0 for v in vals]

        return vals if len(vals) == expected_count else []

    def _read_json_objects(self, ser, buf_name: str) -> List[dict]:
        if ser is None:
            return []

        try:
            waiting = ser.in_waiting
        except Exception as exc:
            self.get_logger().warning(f'{buf_name} in_waiting failed: {exc}')
            return []

        if waiting <= 0:
            return []

        try:
            chunk = ser.read(waiting).decode('utf-8', errors='ignore')
        except Exception as exc:
            self.get_logger().warning(f'{buf_name} read failed: {exc}')
            return []

        if not chunk:
            return []

        old_buf = getattr(self, buf_name)
        new_buf = old_buf + chunk

        json_texts, remainder = self._extract_complete_jsons(new_buf)
        setattr(self, buf_name, remainder)

        objs = []
        for text in json_texts:
            try:
                obj = json.loads(text)
                objs.append(obj)
            except Exception:
                if self.debug_serial:
                    self.get_logger().warning(f'bad json skipped: {text}')
                continue

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
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
                continue

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
