"""
serial_bridge.py  —  v_next10
Changes vs v_next8 (this header was stale — said v_next8 while the code
had already moved to v_next10, which is exactly what caused confusion
about whether the latest fixes were actually deployed):
  4. US HEALTH diagnostics: startup log reports whether us_all_port opened
     OK, plus a periodic health check reporting port state / zero-frames /
     stale-data age — makes "ultrasonic never updates" diagnosable from
     logs alone instead of guessing.
  5. Throttled (2s) the per-poll-cycle warning logs for in_waiting/read
     failures — these were firing up to 20x/sec during sustained port
     contention, which was itself contributing to system-wide sluggishness.
  6. _parse_us_text rewritten: parses each "xN: value cm" field
     independently with last-known-good carry-forward, instead of
     requiring an exact 8-field regex match per line (one corrupted field
     used to discard the entire reading).
  7. "No signal" recognized explicitly as a fixed 9.9 OOB sentinel value,
     not silently skipped — one sensor (x3) reports this on every line,
     which previously left that field permanently unset and blocked
     publishing forever.
  8. Publish readiness now gates ONLY on the 3 sensors autopark_master
     actually reads (x4/x5/x7 — us_trusted_idx), not all 8. An explicit
     "US GATE" startup log line states this plainly so it's unambiguous
     from the log alone which version is running.

Original v_next8 changes vs v_next7:
  1. publishStatus() in the ESP32 firmware sends a full JSON every 100 ms
     that includes btn_state.  This is received on the drive port (ttyUSB0)
     and forwarded on /autopark/esp32_status so autopark_master can watch
     for btn_state=="parking" to start the parking thread.
  2. on_cam_check_request topic is still published (True/False) so
     autopark_master.on_cam_check_request() can respond with yellow/red.
  3. start_switch is still published as Bool for any other subscribers
     that need it.
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
            ('enc_handles_straight', True),
            ('enc_straight_thresh_deg', 5.0),
        ]:
            self.declare_parameter(name, default)

        def gp(n): return self.get_parameter(n).value

        self.drive_port    = str(gp('drive_port'))
        self.us_all_port   = str(gp('us_all_port'))
        self.us_front_port = str(gp('us_front_port'))
        self.us_rear_port  = str(gp('us_rear_port'))
        self.baud          = int(gp('baud'))
        self.debug_serial  = bool(gp('debug_serial'))
        self.enc_handles_straight   = bool(gp('enc_handles_straight'))
        self.enc_straight_thresh    = float(gp('enc_straight_thresh_deg'))

        self.enc_busy = False

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
        self.us_all_line_buf = self.us_front_line_buf = self.us_rear_line_buf = ''

        # FIX (per request): only x4 (left, idx 3), x5 (right, idx 4), and
        # x7 (rear, idx 6) are ever actually read by autopark_master —
        # those are the only ones that should gate readiness. Previously
        # ALL 8 had to be seen at least once before anything published,
        # which is exactly what let x3's permanent "No signal" block
        # publication forever even though the 3 sensors that matter were
        # reading fine. Now: all 8 still default to the 9.9 OOB sentinel
        # and get updated opportunistically whenever they parse, but only
        # the 3 trusted indices are required before a frame is considered
        # ready to publish.
        self.us_last_field    = [9.9] * 8
        self.us_trusted_idx   = {3, 4, 6}   # x4, x5, x7 — must match autopark_master's
                                             # us_left_idx / us_right_idx / us_rear_idx
        self.us_trusted_seen  = set()

        # FIX: diagnostics for the "/autopark/ultrasonic never updates"
        # failure mode. autopark_master was found running with self.latest_us
        # stuck at its [9.9]*8 placeholder default the entire round (KINETIC-A
        # logged L=9.900 R=9.900 while the raw sensor showed real values the
        # whole time) — meaning this node's us_pub was never successfully
        # publishing a full 8-value frame. These counters/timer surface
        # exactly where that's failing instead of requiring guesswork next
        # time: port-not-open vs read-failing vs parse-failing.
        self.us_last_publish_time = 0.0
        self.us_publish_count     = 0
        self.us_fail_streak       = 0
        self.create_timer(5.0, self._us_health_check)

        self.create_subscription(String, gp('command_topic'), self.on_cmd, 10)
        self.create_subscription(Bool, '/enc_busy', self.on_enc_busy, 10)

        self.us_pub           = self.create_publisher(Float32MultiArray, gp('ultrasonic_topic'),        10)
        self.start_pub        = self.create_publisher(Bool,   gp('start_switch_topic'),      10)
        self.steer_ready_pub  = self.create_publisher(Bool,   gp('esp32_steer_ready_topic'), 20)
        self.esp32_status_pub = self.create_publisher(String, gp('esp32_status_topic'),      10)
        self.cam_req_pub      = self.create_publisher(Bool,   gp('cam_check_request_topic'), 10)

        self.timer = self.create_timer(0.05, self.poll_serial)
        self.get_logger().info(
            f'serial_bridge v_next10 started  '
            f'enc_handles_straight={self.enc_handles_straight}  '
            f'thresh={self.enc_straight_thresh}°')
        # FIX: explicit marker so it's unambiguous from the log alone
        # whether this trusted-only-gate version is actually deployed,
        # instead of guessing from behavior. If this line is missing,
        # an older serial_bridge.py is what's actually running.
        self.get_logger().info(
            f'serial_bridge US GATE: trusted_idx={sorted(self.us_trusted_idx)} '
            f'(x4/x5/x7) — publishes once these 3 are seen, '
            f'independent of the other 5 fields')
        # FIX: make the us_all port state impossible to miss at startup —
        # this is the single most diagnostic line for the stuck-at-9.9 bug.
        if self.us_all is None:
            self.get_logger().error(
                f'us_all_port ({self.us_all_port}) FAILED TO OPEN — '
                f'/autopark/ultrasonic will NEVER publish, autopark_master '
                f'will be stuck reading its [9.9]*8 placeholder forever.')
        else:
            self.get_logger().info(
                f'us_all_port ({self.us_all_port}) opened OK — '
                f'waiting for first valid 8-value frame...')

    def on_enc_busy(self, msg: Bool):
        self.enc_busy = bool(msg.data)

    def _us_health_check(self):
        # FIX: periodic visibility into ultrasonic publish health.
        age = time.monotonic() - self.us_last_publish_time if self.us_last_publish_time > 0 else -1
        if self.us_all is None:
            self.get_logger().error(
                'US HEALTH: us_all port is None (never opened) — '
                '/autopark/ultrasonic has NEVER published')
        elif self.us_publish_count == 0:
            self.get_logger().error(
                'US HEALTH: port is open but ZERO successful 8-value frames '
                'parsed since startup — check baud rate / sensor firmware '
                'output format / port-sharing with another reader')
        elif age > 2.0:
            self.get_logger().warning(
                f'US HEALTH: last successful publish was {age:.1f}s ago '
                f'(fail_streak={self.us_fail_streak}) — data is stale, '
                f'check for port contention (e.g. a manual miniterm session '
                f'open on the same device)')

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

        # Block REVERSE drive commands only when enc_busy (avoid port conflict).
        # Forward, steer, stop, led, arm etc. are always forwarded.
        if self.enc_busy:
            try:
                obj = json.loads(msg.data)
                if (obj.get('type') == 'drive'
                        and float(obj.get('speed_mps', 0)) > 0
                        and int(obj.get('gear', 1)) < 0):
                    if self.debug_serial:
                        self.get_logger().info(
                            f'SKIP (enc_busy=True, rev): {msg.data.strip()[:80]}')
                    return
            except Exception:
                pass

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
            self.us_last_publish_time = time.monotonic()
            self.us_publish_count    += 1
            self.us_fail_streak       = 0
            if self.debug_serial:
                self.get_logger().info(f'US: {[f"{v*1000:.0f}mm" for v in vals]}')
        else:
            self.us_fail_streak += 1

    def _read_json(self, ser, buf_attr):
        if ser is None:
            return []
        try:
            waiting = ser.in_waiting
        except Exception as e:
            # FIX: throttled. At 20Hz, an unthrottled warning here during
            # sustained port contention (e.g. a manual miniterm session
            # also open on the same device) was firing up to 20x/sec,
            # spamming the terminal hard enough to contribute to general
            # desktop sluggishness — including unrelated cv2 GUI windows
            # (LotDetector) getting flagged "not responding" by the window
            # manager, since X11 event servicing was starved alongside it.
            self.get_logger().warning(
                f'{buf_attr} in_waiting: {e}', throttle_duration_sec=2.0)
            return []
        if waiting <= 0:
            return []
        try:
            chunk = ser.read(waiting).decode('utf-8', errors='ignore')
        except Exception as e:
            self.get_logger().warning(
                f'{buf_attr} read: {e}', throttle_duration_sec=2.0)
            return []

        buf = getattr(self, buf_attr) + chunk
        texts, remainder = self._extract_json(buf)
        setattr(self, buf_attr, remainder)

        result = []
        for t in texts:
            try:
                result.append(json.loads(t))
            except Exception:
                pass
        return result

    def _poll_drive(self):
        for obj in self._read_json(self.drive, 'drive_buf'):
            if self.debug_serial:
                self.get_logger().info(f'RX: {obj}')

            # ── start_switch (raw bool) ───────────────────────────────
            if 'start_switch' in obj:
                self.start_pub.publish(Bool(data=bool(obj['start_switch'])))

            # ── steer_ready ───────────────────────────────────────────
            if 'steer_ready' in obj:
                self.steer_ready_pub.publish(Bool(data=bool(obj['steer_ready'])))

            # ── btn_state state machine ────────────────────────────────
            # ESP32 publishStatus() sends btn_state every 100 ms.
            # On transition to waiting_cam → trigger cam_check_request so
            # autopark_master.on_cam_check_request() can reply with yellow/red.
            # On transition back to idle → send False to cancel.
            if 'btn_state' in obj:
                new_state = str(obj['btn_state'])
                if new_state != self.last_btn_state:
                    self.get_logger().info(f'btn_state: {self.last_btn_state} → {new_state}')
                    if new_state == 'waiting_cam':
                        self.cam_req_pub.publish(Bool(data=True))
                    elif new_state == 'idle' and self.last_btn_state != 'idle':
                        self.cam_req_pub.publish(Bool(data=False))
                    self.last_btn_state = new_state

            # ── full status object → esp32_status ─────────────────────
            # Forward every status packet so autopark_master can watch
            # for btn_state=="parking" to start the parking thread.
            # Also forwards mode, steer_deg, led_color, etc. for logging.
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
        if ser is None:
            return []
        try:
            waiting = ser.in_waiting
        except Exception as e:
            self.get_logger().warning(
                f'{json_buf_attr} in_waiting: {e}', throttle_duration_sec=2.0)
            return []
        if waiting <= 0:
            return []
        try:
            chunk = ser.read(waiting).decode('utf-8', errors='ignore')
        except Exception as e:
            self.get_logger().warning(
                f'{json_buf_attr} read: {e}', throttle_duration_sec=2.0)
            return []

        result = []

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

        line_buf = getattr(self, line_buf_attr) + chunk
        lines = line_buf.split('\n')
        setattr(self, line_buf_attr, lines[-1])
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
        return [max(0.01, min(5.0, v)) for v in vals]

    def _parse_us_text(self, line: str, n: int) -> List[float]:
        # FIX (per request): only x4/x5/x7 (us_trusted_idx) gate readiness
        # now — those are the only sensors autopark_master ever reads.
        # All 8 fields are still parsed and carried forward opportunistically
        # (so the array shape stays 8 for downstream compatibility), but a
        # frame is considered "ready to publish" the moment all 3 trusted
        # indices have been seen at least once, regardless of what the
        # other 5 (including the permanently "No signal" x3) are doing.
        # "No signal" still recognized explicitly (not just numeric values)
        # and treated as a fixed 9.9 OOB sentinel either way.
        matches = re.findall(r'x(\d+):\s*(No\s*signal|[\d.]+)\s*cm', line, re.IGNORECASE)
        if not matches:
            return []
        updated_any = False
        for idx_str, val_str in matches:
            try:
                idx = int(idx_str) - 1   # x1 -> index 0
                if not (0 <= idx < n):
                    continue
                if val_str.strip().lower().replace(' ', '') == 'nosignal':
                    self.us_last_field[idx] = 9.9   # sentinel: no reading here
                else:
                    self.us_last_field[idx] = max(0.01, min(5.0, float(val_str) / 100.0))
                if idx in self.us_trusted_idx:
                    self.us_trusted_seen.add(idx)
                updated_any = True
            except Exception:
                continue
        if not updated_any:
            return []
        if not self.us_trusted_idx <= self.us_trusted_seen:
            return []   # haven't seen all of x4/x5/x7 at least once yet
        return list(self.us_last_field[:n])

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
