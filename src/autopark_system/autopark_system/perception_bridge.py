import json
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Imu


class PerceptionBridge(Node):
    """
    Bridge the old camera/map/IMU stack into the current autopark_system topics.

    Inputs from ZIP camera/map code:
      /parking_metrics Float32MultiArray
        [car_start_x, car_start_y, target_x_cm, target_y_cm, tilt_deg]
      /imu/data_raw sensor_msgs/Imu

    Outputs used by current control code:
      /autopark/start_pose Pose2D
      /autopark/slot_info String(JSON)

    Important:
      For safe real-car integration, this node can use a fixed near-slot start pose
      while still publishing camera metrics and IMU yaw-rate. Enable camera pose only
      after checking signs with RViz / rqt_plot.
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
            ('camera_x_scale', -0.01),   # target_x_cm -> planner x meters, sign checked later
            ('camera_y_scale', 0.01),    # target_y_cm -> planner y meters
            ('camera_x_offset_m', 0.0),
            ('camera_y_offset_m', 0.0),
            ('yaw_from_camera_tilt', False),
            ('case_detection_mode', 'fixed'),  # fixed / simple_from_metrics
            ('fixed_case', 'both_sides'),
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

        self.latest_metrics: Optional[list[float]] = None
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

        self.get_logger().info('perception_bridge ready: /parking_metrics + /imu/data_raw -> /autopark/start_pose + /autopark/slot_info')

    def on_metrics(self, msg: Float32MultiArray):
        vals = [float(v) for v in msg.data]
        if len(vals) >= 5:
            self.latest_metrics = vals[:5]

    def on_imu(self, msg: Imu):
        self.latest_imu_z = float(msg.angular_velocity.z)
        self.latest_imu_stamp = msg.header.stamp

    def choose_case(self, vals: Optional[list[float]]) -> str:
        if self.case_detection_mode == 'simple_from_metrics' and vals is not None:
            # Conservative placeholder: target_x_cm far to one side can select side case.
            # Keep both_sides near center because the middle slot is the desired target.
            target_x_cm = vals[2]
            if target_x_cm > 80.0:
                return 'right_only'
            if target_x_cm < -80.0:
                return 'left_only'
            return 'both_sides'
        if self.fixed_case in ('left_only', 'right_only', 'both_sides'):
            return self.fixed_case
        return 'both_sides'

    def make_pose(self, vals: Optional[list[float]]) -> Pose2D:
        pose = Pose2D()

        if self.use_camera_pose and vals is not None:
            target_x_cm = vals[2]
            target_y_cm = vals[3]
            tilt_deg = vals[4]
            pose.x = self.camera_x_offset_m + self.camera_x_scale * target_x_cm
            pose.y = self.camera_y_offset_m + self.camera_y_scale * target_y_cm
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
        case_name = self.choose_case(vals)

        slot_obj = {
            'case': case_name,
            'source': 'perception_bridge',
            'camera_metrics_available': vals is not None,
            'camera_metrics': vals if vals is not None else [],
            'imu_yaw_rate_rad_s': self.latest_imu_z,
            'use_camera_pose': self.use_camera_pose,
        }
        self.slot_pub.publish(String(data=json.dumps(slot_obj)))

        if self.publish_start_pose:
            self.pose_pub.publish(self.make_pose(vals))


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
