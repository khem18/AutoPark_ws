import json
import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Imu


class PerceptionBridge(Node):
    """
    Bridge the old camera/map/IMU stack into the current autopark_system topics.

    FIX: perception_bridge now tracks when the camera last sent valid metrics.
    If no metrics received within `camera_metrics_timeout_s`, the pose is NOT
    published. This means autopark_master sees no fresh pose → camera check
    fails → RED LED when car is in a position the camera cannot detect.

    Without this fix: perception_bridge published the default fixed pose
    unconditionally every 100ms even when lot_detector saw nothing,
    making the camera check always pass → always yellow LED.
    """

    def __init__(self):
        super().__init__('perception_bridge')

        for name, default in [
            ('parking_metrics_topic', '/parking_metrics'),
            ('imu_topic', '/imu/data_raw'),
            ('start_pose_topic', '/autopark/start_pose'),
            ('slot_topic', '/autopark/slot_info'),
            ('publish_start_pose', True),
            ('use_camera_pose', False),
            ('default_start_x', 0.0),
            ('default_start_y', 0.70),
            ('default_start_yaw_deg', 180.0),
            ('camera_x_scale', -0.01),
            ('camera_y_scale', 0.01),
            ('camera_x_offset_m', 0.0),
            ('camera_y_offset_m', 0.0),
            ('yaw_from_camera_tilt', False),
            ('case_detection_mode', 'fixed'),
            ('fixed_case', 'both_sides'),
            # NEW: how long (seconds) after last metrics before we consider camera lost
            ('camera_metrics_timeout_s', 1.0),
        ]:
            self.declare_parameter(name, default)

        self.publish_start_pose = bool(self.get_parameter('publish_start_pose').value)
        self.use_camera_pose = bool(self.get_parameter('use_camera_pose').value)
        self.default_start_x = float(self.get_parameter('default_start_x').value)
        self.default_start_y = float(self.get_parameter('default_start_y').value)
        self.default_start_yaw_deg = float(self.get_parameter('default_start_yaw_deg').value)
        self.camera_x_scale = float(self.get_parameter('camera_x_scale').value)
        self.camera_y_scale = float(self.get_parameter('camera_y_scale').value)
        self.camera_x_offset_m = float(self.get_parameter('camera_x_offset_m').value)
        self.camera_y_offset_m = float(self.get_parameter('camera_y_offset_m').value)
        self.yaw_from_camera_tilt = bool(self.get_parameter('yaw_from_camera_tilt').value)
        self.case_detection_mode = str(self.get_parameter('case_detection_mode').value)
        self.fixed_case = str(self.get_parameter('fixed_case').value)
        self.camera_metrics_timeout_s = float(self.get_parameter('camera_metrics_timeout_s').value)

        self.latest_metrics: Optional[list] = None
        self.latest_metrics_time: float = 0.0   # NEW: timestamp of last valid metrics
        self.latest_imu_z = 0.0
        self.latest_imu_stamp = None

        self.create_subscription(
            Float32MultiArray,
            self.get_parameter('parking_metrics_topic').value,
            self.on_metrics,
            10,
        )
        self.create_subscription(
            Imu,
            self.get_parameter('imu_topic').value,
            self.on_imu,
            10,
        )

        self.pose_pub = self.create_publisher(Pose2D, self.get_parameter('start_pose_topic').value, 10)
        self.slot_pub = self.create_publisher(String, self.get_parameter('slot_topic').value, 10)
        self.timer = self.create_timer(0.10, self.publish_bridge)

        self.get_logger().info(
            'perception_bridge ready (FIXED): '
            '/parking_metrics + /imu/data_raw -> /autopark/start_pose + /autopark/slot_info\n'
            f'  camera_metrics_timeout_s={self.camera_metrics_timeout_s}\n'
            '  Pose is ONLY published when camera recently detected yellow lines.\n'
            '  No detection -> no pose -> autopark_master camera check fails -> RED LED.'
        )

    def on_metrics(self, msg: Float32MultiArray):
        vals = [float(v) for v in msg.data]
        if len(vals) >= 5:
            self.latest_metrics = vals[:5]
            self.latest_metrics_time = time.monotonic()   # record when we got it

    def on_imu(self, msg: Imu):
        self.latest_imu_z = float(msg.angular_velocity.z)
        self.latest_imu_stamp = msg.header.stamp

    def camera_metrics_fresh(self) -> bool:
        """
        Returns True only if camera sent valid metrics recently.
        'Recently' = within camera_metrics_timeout_s seconds.
        """
        if self.latest_metrics is None:
            return False
        age = time.monotonic() - self.latest_metrics_time
        return age <= self.camera_metrics_timeout_s

    def choose_case(self, vals) -> str:
        if self.case_detection_mode == 'simple_from_metrics' and vals is not None:
            target_x_cm = vals[2]
            if target_x_cm > 80.0:
                return 'right_only'
            if target_x_cm < -80.0:
                return 'left_only'
            return 'both_sides'
        if self.fixed_case in ('left_only', 'right_only', 'both_sides'):
            return self.fixed_case
        return 'both_sides'

    def make_pose(self, vals) -> Pose2D:
        pose = Pose2D()

        if self.use_camera_pose and vals is not None:
            # vals[2] = k_x = along-aisle distance in cm (negative = before slot) → pose.x
            # vals[3] = k_y = perpendicular distance in cm (sideways to slot) → pose.y
            fwd_cm  = vals[2]   # forward/along-aisle
            side_cm = vals[3]   # sideways/perpendicular
            tilt_deg = vals[4]
            pose.x = self.camera_x_offset_m + self.camera_x_scale * fwd_cm
            pose.y = self.camera_y_offset_m + self.camera_y_scale * side_cm
            yaw_deg = self.default_start_yaw_deg
            if self.yaw_from_camera_tilt:
                yaw_deg = self.default_start_yaw_deg + tilt_deg
            pose.theta = math.radians(yaw_deg)
        else:
            pose.x = self.default_start_x
            pose.y = self.default_start_y
            pose.theta = math.radians(self.default_start_yaw_deg)

        return pose

    def publish_bridge(self):
        vals = self.latest_metrics
        fresh = self.camera_metrics_fresh()
        case_name = self.choose_case(vals)

        slot_obj = {
            'case': case_name,
            'source': 'perception_bridge',
            'camera_metrics_available': vals is not None,
            'camera_metrics_fresh': fresh,           # NEW field for debugging
            'camera_metrics': vals if vals is not None else [],
            'imu_yaw_rate_rad_s': self.latest_imu_z,
            'use_camera_pose': self.use_camera_pose,
        }
        self.slot_pub.publish(String(data=json.dumps(slot_obj)))

        # KEY FIX: only publish start_pose when camera has fresh detection.
        # If camera sees nothing → no pose published → autopark_master gets
        # no fresh pose → _check_camera_origin() returns False → RED LED.
        if self.publish_start_pose:
            if fresh:
                self.pose_pub.publish(self.make_pose(vals))
                # (optional debug log — comment out after testing)
                # self.get_logger().info(f'Pose published from camera (age OK)')
            else:
                age = time.monotonic() - self.latest_metrics_time if self.latest_metrics_time > 0 else -1
                self.get_logger().warn(
                    f'Camera metrics NOT fresh (age={age:.1f}s) → pose NOT published',
                    throttle_duration_sec=2.0
                )


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
