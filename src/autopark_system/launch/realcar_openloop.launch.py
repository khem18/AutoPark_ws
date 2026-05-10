from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    params = PathJoinSubstitution([FindPackageShare('autopark_system'), 'config', 'autopark_params.yaml'])
    return LaunchDescription([
        Node(package='vision', executable='grayscale_node', name='grayscale_converter', output='screen'),
        Node(package='autopark_logic', executable='lot_detector', name='lot_detector', output='screen'),
        Node(package='autopark_system', executable='serial_bridge', name='serial_bridge', output='screen', parameters=[params]),
        Node(package='autopark_system', executable='motion_executor', name='motion_executor', output='screen', parameters=[params]),
        Node(package='autopark_system', executable='autopark_master', name='autopark_master', output='screen', parameters=[params]),
    ])
