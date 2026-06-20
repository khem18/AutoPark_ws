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
    start_vins     = LaunchConfiguration('start_vins')

    camera_imu_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('bringup'),
                'launch',
                'dual_mipi_cam.launch.py',
            ])
        ),
        launch_arguments={'start_vins': start_vins}.items(),
        condition=IfCondition(use_camera_imu),   # only starts when use_camera_imu=true
    )

    return LaunchDescription([

        # ── Launch arguments ────────────────────────────────────────────────
        DeclareLaunchArgument(
            'use_camera_imu',
            default_value='false',          # ← CHANGED from 'true' to 'false'
            description='true = RDK real car with cameras. false = serial/control-only tests.',
        ),
        DeclareLaunchArgument(
            'start_vins',
            default_value='false',
            description='Start VINS only after /rear_cam/image_gray and /imu/data_raw are stable.',
        ),

        # ── Camera + IMU launch (only when use_camera_imu=true) ─────────────
        # This includes dual_mipi_cam.launch.py which requires nv12_to_bgr.
        # Skipped when use_camera_imu=false (bench tests / serial-only mode).
        camera_imu_launch,

        # ── Camera → /parking_metrics (only when use_camera_imu=true) ───────
        Node(
            package='autopark_logic',
            executable='lot_detector',
            name='lot_detector',
            output='screen',
            parameters=[params],
            condition=IfCondition(use_camera_imu),
        ),

        # ── /parking_metrics → /local_map + /goal_pose ─────────────────────
        Node(
            package='autopark_logic',
            executable='local_mapper',
            name='local_mapper',
            output='screen',
            parameters=[params],
            condition=IfCondition(use_camera_imu),
        ),

        # ── /parking_metrics + /imu/data_raw → /autopark/start_pose ─────────
        # perception_bridge runs always (provides default pose when camera off)
        Node(
            package='autopark_system',
            executable='perception_bridge',
            name='perception_bridge',
            output='screen',
            parameters=[params],
        ),

        # ── Optical flow + IMU distance estimator ───────────────────────────
        Node(
            package='autopark_system',
            executable='flow_distance_node',
            name='flow_distance_node',
            output='screen',
            parameters=[params],
            condition=IfCondition(use_camera_imu),
        ),

        # ── MPU6050 IMU — runs always (independent of camera) ───────────────
        # Publishes /imu/data_raw for imu_arc_stop in autopark_master.
        # Previously gated by use_camera_imu (wrong — IMU has no camera dependency).
        Node(
            package='mpu6050_cpp',
            executable='mpu6050_node',
            name='mpu6050_cpp_node',
            output='screen',
        ),

        # ── ESP32 serial bridge (always runs) ───────────────────────────────
        Node(
            package='autopark_system',
            executable='serial_bridge',
            name='serial_bridge',
            output='screen',
            parameters=[params],
        ),

        # ── motion_executor (kept but disabled via YAML plan topic) ─────────
        # autopark_master executes motions directly — this node is idle.
        Node(
            package='autopark_system',
            executable='motion_executor',
            name='motion_executor',
            output='screen',
            parameters=[params],
        ),

        # ── Main parking controller ─────────────────────────────────────────
        Node(
            package='autopark_system',
            executable='autopark_master',
            name='autopark_master',
            output='screen',
            parameters=[params],
        ),

        Node(
            package='autopark_system',
            executable='vins_pose_bridge',
            name='vins_pose_bridge',
            output='screen',
            parameters=[params],
        ),

        Node(
            package='encoder_bridge',
            executable='encoder_bridge',
            name='encoder_bridge',
            parameters=[{
                'enc_port':            '/dev/ttyUSB2',
                'drive_port':          '/dev/ttyUSB0',
                'speed_scale':         0.01,
                # Calibrated driving speeds for 98 kg passenger load:
                # Move 1 forward  calibrated at 0.08 m/s -> enc_fwd_speed_mps = 0.08
                # Move 3/4        calibrated at 0.06 m/s -> enc_rev_speed_mps = 0.06
                # (was 0.06 / 0.04 for unloaded car; increased for 98 kg load)
                'enc_fwd_speed_mps':   0.08,
                'enc_rev_speed_mps':   0.06,
                'straight_steer_thresh': 5.0,
                # ── [v5] Passenger / heavy-load stuck detection ──────────────
                # stuck_speed_enabled: set True to activate the speed-boost feature.
                # When enabled and the car stalls under 98 kg passenger load, the
                # encoder detects no movement after stuck_check_s seconds and
                # boosts the session speed by stuck_boost_mps, repeating until
                # stuck_max_speed_mps. Session speed persists for the full round.
                # Tune: if car still stalls -> lower stuck_check_s or raise
                #       stuck_boost_mps. If steering accuracy suffers at
                #       high speed -> lower stuck_max_speed_mps.
                'stuck_speed_enabled':  True,    # ← ON: enable auto-boost for 98 kg load
                'stuck_boost_mps':      0.020,   # +20 mm/s per stuck event
                'stuck_max_speed_mps':  0.150,   # hard cap (150 mm/s)
                'stuck_check_s':        3.0,     # seconds before declaring stuck
                # [v7] Larger boost multiplier when car has moved 0m at first stuck check.
                # 98 kg load stalls from rest; 3x boost gets it moving sooner.
                # Set to 1.0 to disable (same behaviour as v6).
                'stuck_zero_boost_factor': 3.0,  # x3 boost if dist==0 at first stuck check
                # [v7] Min encoder movement per stuck_check_s interval to NOT be stuck.
                # 80 mm in 3 s (26 mm/s) means car is barely moving → still boost.
                # Tune down (e.g. 0.020) if empty-car boost fires too early.
                'stuck_min_move_m':     0.080,
            }]
        ),
    ])
