import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class GrayscaleConverter(Node):
    def __init__(self):
        super().__init__('grayscale_converter')

        # Subscriptions for both cameras
        self.side_sub = self.create_subscription(
            Image, '/side_cam/image_raw', self.side_callback, 10)
            
        self.rear_sub = self.create_subscription(
            Image, '/rear_cam/image_raw', self.rear_callback, 10)

        # Publishers for the grayscale versions
        self.side_pub = self.create_publisher(Image, '/side_cam/image_gray', 10)
        self.rear_pub = self.create_publisher(Image, '/rear_cam/image_gray', 10)

        self.bridge = CvBridge()
        self.get_logger().info('Grayscale Converter is back online with Time Sync Fix!')

    def side_callback(self, msg):
        self.process_and_publish(msg, self.side_pub)

    def rear_callback(self, msg):
        self.process_and_publish(msg, self.rear_pub)

    def process_and_publish(self, msg, publisher):
        try:
            # Convert ROS BGR8 to OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Simple grayscale conversion
            gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            # Convert back to ROS Image (mono8) and publish
            gray_msg = self.bridge.cv2_to_imgmsg(gray_img, encoding='mono8')
            
            # ก๊อปปี้ข้อมูล Header เดิมมา (เพื่อให้ Frame ID ยังคงถูกต้อง)
            gray_msg.header = msg.header 
            
            # 🚨 ท่าไม้ตาย: บังคับแสตมป์เวลาใหม่ให้เป็นเวลาปัจจุบันของระบบ ROS 2
            # เพื่อให้เวลาของภาพตรงกับเวลาของ IMU แบบเป๊ะๆ (แก้ปัญหา VINS-Fusion รอภาพ)
            gray_msg.header.stamp = self.get_clock().now().to_msg()
            
            publisher.publish(gray_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = GrayscaleConverter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
