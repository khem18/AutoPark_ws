import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    params_file = os.path.expanduser('~/Auto-parking-Kart/src/params/nav2_params.yaml')

    return LaunchDescription([
        # 1. Static TF: หมุดพิกัด
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_pub',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link']
        ),

        # 2. Planner Server: สมองกลคำนวณเส้นทาง
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[params_file],
            remappings=[('/map', '/local_map')]
        ),

        # 3. [เพิ่มใหม่] BT Navigator: โหนดเลขาคอยรับคำสั่งจากปุ่ม RViz
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[params_file]
        ),

        # 4. Lifecycle Manager: สั่งปลุกทั้ง Planner และ Navigator พร้อมกัน
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager',
            output='screen',
            parameters=[{
                'autostart': True,
                'node_names': ['planner_server', 'bt_navigator'] # <--- เพิ่ม bt_navigator ในลิสต์นี้
            }]
        )
    ])
