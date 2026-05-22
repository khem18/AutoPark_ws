from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params = PathJoinSubstitution([
        FindPackageShare('autopark_system'),
        'config',
        'autopark_params.yaml',
    ])

    use_camera_imu = LaunchConfiguration('use_camera_imu')
    start_vins = LaunchConfiguration('start_vins')

    camera_imu_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('bringup'),
                'launch',
                'dual_mipi_cam.launch.py',
            ])
        ),
        launch_arguments={'start_vins': start_vins}.items(),
        condition=IfCondition(use_camera_imu),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_camera_imu',
            default_value='true',
            description='true on RDK real car. false for serial/control-only bench tests.',
        ),
        DeclareLaunchArgument(
            'start_vins',
            default_value='false',
            description='Start VINS only after /rear_cam/image_gray and /imu/data_raw are stable.',
        ),

        camera_imu_launch,

        # Camera -> /parking_metrics
        Node(
            package='autopark_logic',
            executable='lot_detector',
            name='lot_detector',
            output='screen',
            parameters=[params],
            condition=IfCondition(use_camera_imu),
        ),

        # /parking_metrics -> /local_map + /goal_pose
        Node(
            package='autopark_logic',
            executable='local_mapper',
            name='local_mapper',
            output='screen',
            parameters=[params],
            condition=IfCondition(use_camera_imu),
        ),

        # /parking_metrics + /imu/data_raw -> /autopark/start_pose + /autopark/slot_info
        Node(
            package='autopark_system',
            executable='perception_bridge',
            name='perception_bridge',
            output='screen',
            parameters=[params],
        ),

        # Optical flow + IMU simple distance estimator
        Node(
            package='autopark_system',
            executable='flow_distance_node',
            name='flow_distance_node',
            output='screen',
            parameters=[params],
            condition=IfCondition(use_camera_imu),
        ),

        # ESP32 serial bridge
        Node(
            package='autopark_system',
            executable='serial_bridge',
            name='serial_bridge',
            output='screen',
            parameters=[params],
        ),

        # Keep this node present but subscribed to disabled topic in YAML.
        # autopark_master currently executes motions directly after planning.
        Node(
            package='autopark_system',
            executable='motion_executor',
            name='motion_executor',
            output='screen',
            parameters=[params],
        ),

        Node(
            package='autopark_system',
            executable='autopark_master',
            name='autopark_master',
            output='screen',
            parameters=[params],
        ),
    ])
