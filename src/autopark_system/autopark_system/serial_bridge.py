"""
serial_bridge.py  —  v_next5
Fixes:
  1. Ultrasonic parsing: handles BOTH JSON and pipe-text format
       "x1: 47.85 cm | x2: 23.14 cm | ... | x8: 219.93 cm |"
  2. btn_state transition → cam_check_request (unchanged)
"""
import json, re, time
from typing import List, Tuple
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray, Bool


class SerialBridge(Node):
    def __init__(self):
        super().__init__('serial_bridge')
        for name, default in [
            ('command_topic',            '/autopark/cmd_json'),
            ('ultrasonic_topic',         '/autopark/ultrasonic'),
            ('start_switch_topic',       '/autopark/start_switch'),
            ('esp32_steer_ready_topic',  '/autopark/esp32_steer_ready'),
            ('esp32_status_topic',       '/autopark/esp32_status'),
            ('cam_check_request_topic',  '/autopark/cam_check_request'),
            ('drive_port',    '/dev/ttyUSB0'),
            ('us_all_port',   '/dev/ttyUSB1'),
            ('us_front_port', ''),
            ('us_rear_port',  ''),
            ('baud',          115200),
            ('debug_serial',  False),
        ]:
            self.declare_parameter(name, default)

        def gp(n): return self.get_parameter(n).value

        self.drive_port    = str(gp('drive_port'))
        self.us_all_port   = str(gp('us_all_port'))
        self.us_front_port = str(gp('us_front_port'))
        self.us_rear_port  = str(gp('us_rear_port'))
        self.baud          = int(gp('baud'))
        self.debug_serial  = bool(gp('debug_serial'))

        try:
            import serial as _serial
            self._serial = _serial
        except ImportError:
            self._serial = None
            self.get_logger().error('pyserial not installed')

        self.drive    = self._open(self.drive_port,    self.baud)
        self.us_all   = self._open(self.us_all_port,   self.baud)
        self.us_front = self._open(self.us_front_port, self.baud)
        self.us_rear  = self._open(self.us_rear_port,  self.baud)

        self.drive_buf = self.us_all_buf = self.us_front_buf = self.us_rear_buf = ''
        self.last_btn_state = 'idle'
        # Line buffers for non-JSON formats
        self.us_all_line_buf = self.us_front_line_buf = self.us_rear_line_buf = ''

        self.create_subscription(String, gp('command_topic'), self.on_cmd, 10)

        self.us_pub           = self.create_publisher(Float32MultiArray, gp('ultrasonic_topic'),        10)
        self.start_pub        = self.create_publisher(Bool,   gp('start_switch_topic'),      10)
        self.steer_ready_pub  = self.create_publisher(Bool,   gp('esp32_steer_ready_topic'), 20)
        self.esp32_status_pub = self.create_publisher(String, gp('esp32_status_topic'),      10)
        self.cam_req_pub      = self.create_publisher(Bool,   gp('cam_check_request_topic'), 10)

        self.timer = self.create_timer(0.05, self.poll_serial)
        self.get_logger().info('serial_bridge v_next5 started')

    def _open(self, port, baud):
        if not self._serial or not port: return None
        try:
            s = self._serial.Serial(port=port, baudrate=baud, timeout=0.05, write_timeout=0.5)
            time.sleep(2.0)
            self.get_logger().info(f'OPENED {port} @ {baud}')
            return s
        except Exception as e:
            self.get_logger().warning(f'Cannot open {port}: {e}')
            return None

    def on_cmd(self, msg: String):
        if self.drive is None:
            self.get_logger().error('drive serial NOT open')
            return
        try:
            self.drive.write((msg.data.strip() + '\n').encode('utf-8'))
            self.drive.flush()
            if self.debug_serial:
                self.get_logger().info(f'TX: {msg.data.strip()}')
        except Exception as e:
            self.get_logger().warning(f'drive write: {e}')
            self.drive = self._open(self.drive_port, self.baud)

    def poll_serial(self):
        self._poll_drive()
        vals = self._poll_us()
        if len(vals) == 8:
            m = Float32MultiArray()
            m.data = [float(v) for v in vals]
            self.us_pub.publish(m)
            if self.debug_serial:
                self.get_logger().info(f'US: {[f"{v*1000:.0f}mm" for v in vals]}')

    def _poll_drive(self):
        for obj in self._read_json(self.drive, 'drive_buf'):
            if self.debug_serial:
                self.get_logger().info(f'RX: {obj}')

            if 'start_switch' in obj:
                self.start_pub.publish(Bool(data=bool(obj['start_switch'])))

            if 'steer_ready' in obj:
                self.steer_ready_pub.publish(Bool(data=bool(obj['steer_ready'])))

            if 'btn_state' in obj:
                new_state = str(obj['btn_state'])
                if new_state != self.last_btn_state:
                    self.get_logger().info(f'btn_state: {self.last_btn_state} → {new_state}')
                    if new_state == 'waiting_cam':
                        self.cam_req_pub.publish(Bool(data=True))
                    elif new_state == 'idle' and self.last_btn_state != 'idle':
                        self.cam_req_pub.publish(Bool(data=False))
                    self.last_btn_state = new_state

            if 'mode' in obj or 'steer_deg' in obj:
                try:
                    self.esp32_status_pub.publish(String(data=json.dumps(obj)))
                except Exception:
                    pass

    def _poll_us(self) -> List[float]:
        if self.us_all is not None:
            vals = self._read_us_sensor(self.us_all, 'us_all_buf', 'us_all_line_buf', 8)
            if len(vals) == 8:
                return vals

        front, rear = [], []
        if self.us_front:
            vals = self._read_us_sensor(self.us_front, 'us_front_buf', 'us_front_line_buf', 4)
            if len(vals) == 4: front = vals
        if self.us_rear:
            vals = self._read_us_sensor(self.us_rear, 'us_rear_buf', 'us_rear_line_buf', 4)
            if len(vals) == 4: rear = vals
        if len(front) == 4 and len(rear) == 4:
            return front + rear
        return []

    def _read_us_sensor(self, ser, json_buf_attr: str, line_buf_attr: str,
                        expected_n: int) -> List[float]:
        """
        Read ultrasonic sensor. Handles two formats:
          A) JSON:  {"distances_cm": [47.85, 23.14, ...]}
                    {"distances_m":  [0.4785, 0.2314, ...]}
          B) Text:  "x1: 47.85 cm | x2: 23.14 cm | ... | x8: 219.93 cm |"
        Returns list of floats in METRES, or [] if nothing valid yet.
        """
        if ser is None:
            return []
        try:
            waiting = ser.in_waiting
        except Exception as e:
            self.get_logger().warning(f'{json_buf_attr} in_waiting: {e}')
            return []
        if waiting <= 0:
            return []
        try:
            chunk = ser.read(waiting).decode('utf-8', errors='ignore')
        except Exception as e:
            self.get_logger().warning(f'{json_buf_attr} read: {e}')
            return []

        result = []

        # ── Try JSON format first ──────────────────────────────────────────
        json_buf = getattr(self, json_buf_attr) + chunk
        json_texts, json_remainder = self._extract_json(json_buf)
        setattr(self, json_buf_attr, json_remainder)
        for t in json_texts:
            try:
                obj = json.loads(t)
                vals = self._parse_us_json(obj, expected_n)
                if len(vals) == expected_n:
                    result = vals
            except Exception:
                pass
        if result:
            return result

        # ── Try pipe-text format ───────────────────────────────────────────
        # Format: "x1: 47.85 cm | x2: 23.14 cm | ... | x8: 219.93 cm |"
        line_buf = getattr(self, line_buf_attr) + chunk
        lines = line_buf.split('\n')
        setattr(self, line_buf_attr, lines[-1])  # keep incomplete last line
        for line in lines[:-1]:
            vals = self._parse_us_text(line.strip(), expected_n)
            if len(vals) == expected_n:
                result = vals
        return result

    def _parse_us_json(self, obj: dict, n: int) -> List[float]:
        arr = obj.get('distances_m', obj.get('distances_cm', []))
        try:
            vals = [float(x) for x in arr]
        except Exception:
            return []
        unit = str(obj.get('unit', 'm')).lower()
        if 'cm' in unit or obj.get('distances_cm') is not None:
            vals = [v / 100.0 for v in vals]
        if len(vals) != n:
            return []
        # Clamp to reasonable range (0.01m – 5.0m)
        return [max(0.01, min(5.0, v)) for v in vals]

    def _parse_us_text(self, line: str, n: int) -> List[float]:
        """
        Parse 'x1: 47.85 cm | x2: 23.14 cm | ... | x8: 219.93 cm |'
        Returns values in METRES.
        """
        # Match all "xN: VALUE cm" patterns
        matches = re.findall(r'x\d+:\s*([\d.]+)\s*cm', line)
        if len(matches) != n:
            return []
        try:
            vals_m = [float(v) / 100.0 for v in matches]  # cm → m
        except Exception:
            return []
        # Clamp to 0.01 – 5.0 m
        return [max(0.01, min(5.0, v)) for v in vals_m]

    def _extract_json(self, s: str) -> Tuple[List[str], str]:
        objs, start, depth = [], None, 0
        in_str = esc = False
        last = 0
        for i, c in enumerate(s):
            if in_str:
                esc = not esc and c == '\\'
                if not esc and c == '"': in_str = False
                continue
            if c == '"': in_str = True; continue
            if c == '{':
                if depth == 0: start = i
                depth += 1
            elif c == '}' and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(s[start:i+1]); last = i+1; start = None
        rem = s[last:]
        if len(rem) > 4096: rem = rem[-1024:]
        return objs, rem


def main(args=None):
    rclpy.init(args=args)
    node = SerialBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
