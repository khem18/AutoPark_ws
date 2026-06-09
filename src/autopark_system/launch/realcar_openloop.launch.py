from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params = PathJoinSubstitution([
        FindPackageShare('autopark_system'), 'config', 'autopark_params.yaml'
    ])

    return LaunchDescription([

        # ── Camera: NV12 → BGR8 ──────────────────────────────────────
        Node(
            package='vision',
            executable='grayscale_node',
            name='grayscale_converter',
            output='screen',
        ),

        # ── Parking lot detector + ego pose estimator ────────────────
        # Publishes: /parking_metrics  /lot_obstacle  /ego_pose
        Node(
            package='autopark_logic',
            executable='lot_detector',
            name='lot_detector',
            output='screen',
        ),

        # ── Ego pose bridge (ONLY if using Option A) ─────────────────
        # Converts /ego_pose (Float32MultiArray) → /autopark/start_pose (Pose2D)
        # Remove this node if you applied the 3-block patch to autopark_master.py
        # directly (Option B) — the master will subscribe to /ego_pose itself.
        Node(
            package='autopark_logic',
            executable='ego_pose_bridge',
            name='ego_pose_bridge',
            output='screen',
        ),

        # ── Hardware bridge ──────────────────────────────────────────
        # /dev/ttyUSB0 = drive ESP32,  /dev/ttyUSB1 = ultrasonic ESP32
        Node(
            package='autopark_system',
            executable='serial_bridge',
            name='serial_bridge',
            output='screen',
            parameters=[params],
        ),

        # ── Open-loop motion executor ────────────────────────────────
        Node(
            package='autopark_system',
            executable='motion_executor',
            name='motion_executor',
            output='screen',
            parameters=[params],
        ),

        # ── Parking master ───────────────────────────────────────────
        # Waits for start switch → reads /ego_pose (or /autopark/start_pose)
        # → calls planner → publishes motion plan
        Node(
            package='autopark_system',
            executable='autopark_master',
            name='autopark_master',
            output='screen',
            parameters=[params],
        ),
    ])