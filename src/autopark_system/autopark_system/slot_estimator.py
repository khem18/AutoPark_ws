import json
from typing import Optional
import cv2, numpy as np, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

class SlotEstimator(Node):
    def __init__(self):
        super().__init__('slot_estimator')
        self.declare_parameter('right_image_topic','/front_cam/image_raw')
        self.declare_parameter('rear_image_topic','/rear_cam/image_raw')
        self.declare_parameter('slot_topic','/autopark/slot_info')
        self.declare_parameter('yellow_low_hsv',[15,50,50])
        self.declare_parameter('yellow_high_hsv',[45,255,255])
        self.low=np.array(self.get_parameter('yellow_low_hsv').value,dtype=np.uint8)
        self.high=np.array(self.get_parameter('yellow_high_hsv').value,dtype=np.uint8)
        self.bridge=CvBridge(); self.latest_right=None; self.latest_rear=None
        self.create_subscription(Image,self.get_parameter('right_image_topic').value,self.on_right,10)
        self.create_subscription(Image,self.get_parameter('rear_image_topic').value,self.on_rear,10)
        self.slot_pub=self.create_publisher(String,self.get_parameter('slot_topic').value,10)
        self.timer=self.create_timer(0.2,self.publish_estimate)
    def on_right(self,msg): self.latest_right=self.bridge.imgmsg_to_cv2(msg,'bgr8')
    def on_rear(self,msg): self.latest_rear=self.bridge.imgmsg_to_cv2(msg,'bgr8')
    def _yellow_score(self,frame):
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV); mask=cv2.inRange(hsv,self.low,self.high)
        return float(np.count_nonzero(mask))/float(mask.size)
    def publish_estimate(self):
        if self.latest_right is None or self.latest_rear is None: return
        rs=self._yellow_score(self.latest_right); bs=self._yellow_score(self.latest_rear)
        if rs>0.08 and bs>0.08: case='both_sides'
        elif rs>0.08: case='right_only'
        elif bs>0.08: case='left_only'
        else: case='both_sides'
        self.slot_pub.publish(String(data=json.dumps({'case':case,'right_score':rs,'rear_score':bs,'source':'simple_hsv_placeholder'})))

def main(args=None):
    rclpy.init(args=args); node=SlotEstimator(); rclpy.spin(node); node.destroy_node(); rclpy.shutdown()
