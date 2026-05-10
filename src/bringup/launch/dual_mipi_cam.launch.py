import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    
    # ==========================================
    # 📷 1. กล้องข้าง (Side Camera) - เริ่มทันทีที่ T=0
    # ==========================================
    side_cam = Node(
        package='mipi_cam',
        executable='mipi_cam',
        name='side_mipi_cam',
        parameters=[{
            'video_device': 'ov5647',
            'device_mode': 'single',
            'channel': 0, # Port 0
            'image_width': 1280,
            'image_height': 720,
            'out_format': 'nv12',
            'camera_calibration_file_path': '/home/ddddd/AutoPark_ws/src/bringup/config/lastest_ost.yaml'
        }],
        remappings=[('/image_raw', '/side_cam/image_nv12'),
                    ('/camera_info', '/side_cam/camera_info')]
    )

    side_codec = Node(
        package='hobot_codec',
        executable='hobot_codec_republish',
        name='side_codec_converter',
        parameters=[{
            'in_mode': 'ros',
            'in_format': 'nv12',
            'out_mode': 'ros',
            'out_format': 'bgr8',
            'sub_topic': '/side_cam/image_nv12',
            'pub_topic': '/side_cam/image_raw'
        }]
    )

    # ==========================================
    # 📷 2. กล้องหลัง (Rear Camera) - ตั้งค่าไว้ก่อน
    # ==========================================
    rear_cam = Node(
        package='mipi_cam',
        executable='mipi_cam',
        name='rear_mipi_cam',
        parameters=[{
            'video_device': 'ov5647',
            'device_mode': 'single',
            'channel': 2, # Port 2
            'image_width': 1280,
            'image_height': 720,
            'out_format': 'nv12',
            'camera_calibration_file_path': '/home/ddddd/AutoPark_ws/src/bringup/config/lastest_ost.yaml'
        }],
        remappings=[('/image_raw', '/rear_cam/image_nv12'),
                    ('/camera_info', '/rear_cam/camera_info')]
    )

    rear_codec = Node(
        package='hobot_codec',
        executable='hobot_codec_republish',
        name='rear_codec_converter',
        parameters=[{
            'in_mode': 'ros',
            'in_format': 'nv12',
            'out_mode': 'ros',
            'out_format': 'bgr8',
            'sub_topic': '/rear_cam/image_nv12',
            'pub_topic': '/rear_cam/image_raw'
        }]
    )

    # ⏳ สร้าง TimerAction หน่วงเวลา 5 วินาทีสำหรับระบบกล้องหลัง
    delayed_rear_system = TimerAction(
        period=5.0,
        actions=[rear_cam, rear_codec]
    )

    # ==========================================
    # 🧠 3. โหนดส่วนกลาง (ทำงานทันที)
    # ==========================================
    grayscale_node = Node(
        package='vision',
        executable='grayscale_node',
        name='grayscale_converter',
        output='screen'
    )

    return LaunchDescription([
        side_cam,
        side_codec,
        grayscale_node,
        delayed_rear_system # ระบบกล้องหลังจะเริ่มหลังจากนี้ 5 วินาที
    ])
