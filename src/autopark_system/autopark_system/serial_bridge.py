"""
serial_bridge.py  —  v_next4
New: watches btn_state transitions from ESP32 and publishes /autopark/cam_check_request.
"""
import json, time
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
            ('cam_check_request_topic',  '/autopark/cam_check_request'),  # NEW
            ('drive_port',    '/dev/ttyUSB2'),
            ('us_all_port',   '/dev/ttyUSB1'),
            ('us_front_port', ''),
            ('us_rear_port',  ''),
            ('baud',          115200),
            ('debug_serial',  False),
        ]:
            self.declare_parameter(name, default)

        def gp(n): return self.get_parameter(n).value

        self.drive_port   = str(gp('drive_port'))
        self.us_all_port  = str(gp('us_all_port'))
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
        self.last_btn_state = 'idle'    # track transitions

        self.create_subscription(String, gp('command_topic'), self.on_cmd, 10)

        self.us_pub           = self.create_publisher(Float32MultiArray, gp('ultrasonic_topic'),        10)
        self.start_pub        = self.create_publisher(Bool,   gp('start_switch_topic'),      10)
        self.steer_ready_pub  = self.create_publisher(Bool,   gp('esp32_steer_ready_topic'), 20)
        self.esp32_status_pub = self.create_publisher(String, gp('esp32_status_topic'),      10)
        self.cam_req_pub      = self.create_publisher(Bool,   gp('cam_check_request_topic'), 10)  # NEW

        self.timer = self.create_timer(0.05, self.poll_serial)

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
            m = Float32MultiArray(); m.data = vals
            self.us_pub.publish(m)

    def _poll_drive(self):
        for obj in self._read_json(self.drive, 'drive_buf'):
            if self.debug_serial:
                self.get_logger().info(f'RX: {obj}')

            # start_switch (existing)
            if 'start_switch' in obj:
                self.start_pub.publish(Bool(data=bool(obj['start_switch'])))

            # steer_ready (existing)
            if 'steer_ready' in obj:
                self.steer_ready_pub.publish(Bool(data=bool(obj['steer_ready'])))

            # ── NEW: btn_state transition detection ───────────────────────
            # When ESP32 transitions to "waiting_cam", publish cam_check_request=True.
            # This tells autopark_master to check camera and respond with LED color.
            if 'btn_state' in obj:
                new_state = str(obj['btn_state'])
                if new_state != self.last_btn_state:
                    self.get_logger().info(
                        f'btn_state: {self.last_btn_state} → {new_state}')
                    if new_state == 'waiting_cam':
                        self.cam_req_pub.publish(Bool(data=True))
                    elif new_state == 'idle' and self.last_btn_state != 'idle':
                        self.cam_req_pub.publish(Bool(data=False))
                    self.last_btn_state = new_state

            # Full status JSON (for logging/debug)
            if 'mode' in obj or 'steer_deg' in obj:
                try:
                    self.esp32_status_pub.publish(String(data=json.dumps(obj)))
                except Exception:
                    pass

    def _poll_us(self) -> List[float]:
        if self.us_all is not None:
            for obj in self._read_json(self.us_all, 'us_all_buf'):
                v = self._parse_us(obj, 8)
                if len(v) == 8: return v
        front, rear = [], []
        if self.us_front:
            for obj in self._read_json(self.us_front, 'us_front_buf'):
                v = self._parse_us(obj, 4)
                if len(v) == 4: front = v
        if self.us_rear:
            for obj in self._read_json(self.us_rear, 'us_rear_buf'):
                v = self._parse_us(obj, 4)
                if len(v) == 4: rear = v
        if len(front) == 4 and len(rear) == 4:
            return front + rear
        return []

    def _parse_us(self, obj, n):
        arr = obj.get('distances_m', obj.get('distances_cm', []))
        try: vals = [float(x) for x in arr]
        except: return []
        if str(obj.get('unit','m')).lower() == 'cm':
            vals = [v/100 for v in vals]
        return vals if len(vals) == n else []

    def _read_json(self, ser, buf_attr) -> List[dict]:
        if ser is None: return []
        try:
            waiting = ser.in_waiting
        except Exception as e:
            self.get_logger().warning(f'{buf_attr}: {e}')
            return []
        if waiting <= 0: return []
        try:
            chunk = ser.read(waiting).decode('utf-8', errors='ignore')
        except Exception as e:
            self.get_logger().warning(f'{buf_attr} read: {e}')
            return []
        buf = getattr(self, buf_attr) + chunk
        texts, remainder = self._extract_json(buf)
        setattr(self, buf_attr, remainder)
        objs = []
        for t in texts:
            try: objs.append(json.loads(t))
            except: pass
        return objs

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
    try: rclpy.spin(node)
    finally: node.destroy_node(); rclpy.shutdown()
