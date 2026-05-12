import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction, ExecuteProcess

CALIB    = '/home/ddddd/AutoPark_ws/src/bringup/config/lastest_ost.yaml'
VINS_CFG = '/home/ddddd/AutoPark_ws/src/params/vins_config_rdkx5.yaml'

def generate_launch_description():

    # 1. SIDE CAMERA
    side_cam = Node(
        package='mipi_cam',
        executable='mipi_cam',
        name='side_mipi_cam',
        parameters=[{
            'video_device': 'ov5647',
            'device_mode': 'single',
            'channel': 0,
            'image_width': 1280,
            'image_height': 720,
            'out_format': 'nv12',
            'camera_calibration_file_path': CALIB,
        }],
        remappings=[
            ('/image_raw',   '/side_cam/image_nv12'),
            ('/camera_info', '/side_cam/camera_info'),
        ],
    )

    side_codec = Node(
        package='vision',
        executable='nv12_to_bgr',
        name='side_nv12_to_bgr',
        parameters=[{
            'sub_topic': '/side_cam/image_nv12',
            'pub_topic': '/side_cam/image_raw',
        }],
    )

    # 2. REAR CAMERA (delayed 5 s)
    rear_cam = Node(
        package='mipi_cam',
        executable='mipi_cam',
        name='rear_mipi_cam',
        parameters=[{
            'video_device': 'ov5647',
            'device_mode': 'single',
            'channel': 2,
            'image_width': 1280,
            'image_height': 720,
            'out_format': 'nv12',
            'camera_calibration_file_path': CALIB,
        }],
        remappings=[
            ('/image_raw',   '/rear_cam/image_nv12'),
            ('/camera_info', '/rear_cam/camera_info'),
        ],
    )

    rear_codec = Node(
        package='vision',
        executable='nv12_to_bgr',
        name='rear_nv12_to_bgr',
        parameters=[{
            'sub_topic': '/rear_cam/image_nv12',
            'pub_topic': '/rear_cam/image_raw',
        }],
    )

    delayed_rear = TimerAction(
        period=5.0,
        actions=[rear_cam, rear_codec],
    )

    # 3. GRAYSCALE CONVERTER
    grayscale_node = Node(
        package='vision',
        executable='grayscale_node',
        name='grayscale_converter',
        output='screen',
    )

    # 4. MPU6050
    imu_node = Node(
        package='mpu6050_cpp',
        executable='mpu6050_node',
        name='mpu6050_cpp_node',
        output='screen',
    )

    # 5. VINS ESTIMATOR (ExecuteProcess) - delayed 8 s
    vins_estimator = ExecuteProcess(
        cmd=['/home/ddddd/AutoPark_ws/install/vins/lib/vins/vins_node', VINS_CFG],
        output='screen'
    )

    delayed_vins = TimerAction(
        period=5.0,  # 🚨 ผมลดเวลาลงเหลือ 5 วินาที จะได้ไม่ต้องรอนานครับ!
        actions=[vins_estimator],
    )

    return LaunchDescription([
        side_cam,
        side_codec,
        grayscale_node,
        imu_node,
        delayed_rear,
        delayed_vins,
    ])
