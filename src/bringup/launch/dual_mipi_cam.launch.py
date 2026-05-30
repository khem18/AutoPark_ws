import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    home = os.path.expanduser('~')

    calib_arg = DeclareLaunchArgument(
        'calib_path',
        default_value=os.path.join(home, 'AutoPark_ws/src/bringup/config/lastest_ost.yaml'),
        description='Camera calibration YAML path for the OV5647 cameras.',
    )
    vins_arg = DeclareLaunchArgument(
        'vins_config',
        default_value=os.path.join(home, 'AutoPark_ws/src/params/vins_config_rdkx5.yaml'),
        description='VINS config path. Only used if start_vins:=true.',
    )
    start_vins_arg = DeclareLaunchArgument(
        'start_vins',
        default_value='false',
        description='Set true only after camera + IMU topics are stable.',
    )

    calib = LaunchConfiguration('calib_path')
    vins_cfg = LaunchConfiguration('vins_config')

    # 1) SIDE / FRONT CAMERA used by the parking lot detector.
    side_cam = Node(
        package='mipi_cam',
        executable='mipi_cam',
        name='side_mipi_cam',
        output='screen',
        parameters=[{
            'video_device': 'ov5647',
            'device_mode': 'single',
            'channel': 0,
            'image_width': 1280,
            'image_height': 720,
            'out_format': 'nv12',
            'camera_calibration_file_path': calib,
        }],
        remappings=[
            ('/image_raw', '/side_cam/image_nv12'),
            ('/camera_info', '/side_cam/camera_info'),
        ],
    )

    side_codec = Node(
        package='vision',
        executable='nv12_to_bgr',
        name='side_nv12_to_bgr',
        output='screen',
        parameters=[{
            'sub_topic': '/side_cam/image_nv12',
            'pub_topic': '/side_cam/image_raw',
        }],
    )

    # Optional alias: some older current-code files expect /front_cam/image_raw.
    # The parking detector still uses /side_cam/image_raw.
    side_to_front_alias = Node(
        package='vision',
        executable='nv12_to_bgr',
        name='front_alias_disabled_dummy',
        output='screen',
        parameters=[{
            'sub_topic': '/unused_disable_alias',
            'pub_topic': '/unused_disable_alias_bgr',
        }],
    )

    # 2) REAR CAMERA for rear view / VINS grayscale.
    rear_cam = Node(
        package='mipi_cam',
        executable='mipi_cam',
        name='rear_mipi_cam',
        output='screen',
        parameters=[{
            'video_device': 'ov5647',
            'device_mode': 'single',
            'channel': 2,
            'image_width': 1280,
            'image_height': 720,
            'out_format': 'nv12',
            'camera_calibration_file_path': calib,
        }],
        remappings=[
            ('/image_raw', '/rear_cam/image_nv12'),
            ('/camera_info', '/rear_cam/camera_info'),
        ],
    )

    rear_codec = Node(
        package='vision',
        executable='nv12_to_bgr',
        name='rear_nv12_to_bgr',
        output='screen',
        parameters=[{
            'sub_topic': '/rear_cam/image_nv12',
            'pub_topic': '/rear_cam/image_raw',
        }],
    )

    delayed_rear = TimerAction(period=5.0, actions=[rear_cam, rear_codec])

    # 3) Grayscale images for debugging / VINS.
    grayscale_node = Node(
        package='vision',
        executable='grayscale_node',
        name='grayscale_converter',
        output='screen',
    )

    # 4) MPU6050 IMU -> /imu/data_raw.
    imu_node = Node(
        package='mpu6050_cpp',
        executable='mpu6050_node',
        name='mpu6050_cpp_node',
        output='screen',
    )

    # VINS is intentionally not auto-started here. First confirm:
    #   ros2 topic hz /rear_cam/image_gray
    #   ros2 topic hz /imu/data_raw
    # Then start your VINS command manually using the vins_config path.

    return LaunchDescription([
        calib_arg,
        vins_arg,
        start_vins_arg,
        side_cam,
        side_codec,
        delayed_rear,
        grayscale_node,
        imu_node,
    ])