import json, math
from typing import List, Optional
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose2D
from std_msgs.msg import String

class PathFollower(Node):
    def __init__(self):
        super().__init__('path_follower')
        for name, default in [('path_topic','/autopark/path'),('pose_topic','/autopark/start_pose'),('command_topic','/autopark/cmd_json'),('target_speed_mps',0.35),('wheelbase_m',0.739),('max_steer_deg',30.0)]:
            self.declare_parameter(name, default)
        self.target_speed_mps=float(self.get_parameter('target_speed_mps').value)
        self.max_steer_deg=float(self.get_parameter('max_steer_deg').value)
        self.create_subscription(Path,self.get_parameter('path_topic').value,self.on_path,10)
        self.create_subscription(Pose2D,self.get_parameter('pose_topic').value,self.on_pose,10)
        self.cmd_pub=self.create_publisher(String,self.get_parameter('command_topic').value,10)
        self.current_pose=None; self.path_points=[]; self.index=0; self.active=False
        self.timer=self.create_timer(0.05,self.tick)
    def on_pose(self,msg): self.current_pose=msg
    def on_path(self,msg):
        self.path_points=[(p.pose.position.x,p.pose.position.y) for p in msg.poses]
        self.index=0; self.active=len(self.path_points)>0
    def tick(self):
        if not self.active or self.current_pose is None or not self.path_points: return
        if self.index>=len(self.path_points):
            self.cmd_pub.publish(String(data=json.dumps({'type':'stop','reason':'path_complete'}))); self.active=False; return
        tx,ty=self.path_points[self.index]; dx=tx-self.current_pose.x; dy=ty-self.current_pose.y; dist=math.hypot(dx,dy)
        if dist<0.05: self.index+=1; return
        desired=math.atan2(dy,dx); yaw=math.radians(self.current_pose.theta)
        err=math.atan2(math.sin(desired-yaw), math.cos(desired-yaw))
        steer_deg=max(-self.max_steer_deg, min(self.max_steer_deg, math.degrees(err)))
        gear=1
        if abs(err)>math.radians(100): gear=-1; steer_deg=-steer_deg
        self.cmd_pub.publish(String(data=json.dumps({'type':'drive','speed_mps':self.target_speed_mps,'gear':gear,'steer_deg':steer_deg,'target_index':self.index})))

def main(args=None):
    rclpy.init(args=args); node=PathFollower(); rclpy.spin(node); node.destroy_node(); rclpy.shutdown()
