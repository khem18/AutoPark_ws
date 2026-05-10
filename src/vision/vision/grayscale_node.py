import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class GrayscaleConverter(Node):
    def __init__(self):
        super().__init__('grayscale_converter')
        
        # Subscriptions for both cameras
        self.front_sub = self.create_subscription(
            Image, '/front_cam/image_raw', self.front_callback, 10)
        self.rear_sub = self.create_subscription(
            Image, '/rear_cam/image_raw', self.rear_callback, 10)
            
        # Publishers for the grayscale versions
        self.front_pub = self.create_publisher(Image, '/front_cam/image_gray', 10)
        self.rear_pub = self.create_publisher(Image, '/rear_cam/image_gray', 10)
        
        self.bridge = CvBridge()
        self.get_logger().info('Grayscale Converter is back online and running cool!')

    def front_callback(self, msg):
        self.process_and_publish(msg, self.front_pub)

    def rear_callback(self, msg):
        self.process_and_publish(msg, self.rear_pub)

    def process_and_publish(self, msg, publisher):
        # Convert ROS BGR8 to OpenCV
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Simple grayscale conversion
        gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Convert back to ROS Image (mono8) and publish
        gray_msg = self.bridge.cv2_to_imgmsg(gray_img, encoding='mono8')
        gray_msg.header = msg.header # Preserve timestamps for calibration
        publisher.publish(gray_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GrayscaleConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
