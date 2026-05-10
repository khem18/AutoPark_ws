import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    # 1. Side Camera (Starts T=0)
    front_cam = Node(
        package='mipi_cam',
        executable='mipi_cam',
        name='front_mipi_cam',
        parameters=[{
            'video_device': 'ov5647',
            'device_mode': 'single',
            'channel': 0,
            'image_width': 1280,
            'image_height': 720,
            'out_format': 'mono8',
            'camera_calibration_file_path': '/home/ddddd/AutoPark_ws/src/bringup/config/lastest_ost.yaml'
        }],
        remappings=[('/image_raw', '/front_cam/image_raw'),
					('/camera_info', '/front_cam/camera_info')
        ]
    )

    # 2. Grayscale Converter (Starts T=0 - waits for front_cam topics)
    grayscale_node = Node(
        package='vision',
        executable='grayscale_node',
        name='grayscale_converter',
        output='screen'
    )

    # 3. Rear Camera (Starts T=3s to prevent I2C bus crash)
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
            'out_format': 'mono8',
            'camera_calibration_file_path': '/home/ddddd/AutoPark_ws/src/bringup/config/lastest_ost.yaml'
        }],
        remappings=[('/image_raw', '/rear_cam/image_raw')]
    )

    delayed_rear_cam = TimerAction(
        period=30.0,
        actions=[rear_cam]
    )

    return LaunchDescription([
        front_cam,
        grayscale_node,
        delayed_rear_cam
    ])
