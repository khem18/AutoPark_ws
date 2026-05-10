import json
import time
from typing import List, Dict, Any
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MotionExecutor(Node):
    def __init__(self):
        super().__init__('motion_executor')
        for name, default in [
            ('plan_topic','/autopark/plan_result'),
            ('command_topic','/autopark/cmd_json'),
            ('open_loop_speed_mps',0.25),
            ('segment_pause_s',0.20)]:
            self.declare_parameter(name, default)
        self.speed_mps=float(self.get_parameter('open_loop_speed_mps').value)
        self.segment_pause_s=float(self.get_parameter('segment_pause_s').value)
        self.cmd_pub=self.create_publisher(String, self.get_parameter('command_topic').value, 10)
        self.create_subscription(String, self.get_parameter('plan_topic').value, self.on_plan, 10)
        self.executing=False
        self.get_logger().info('motion_executor ready (open-loop)')

    def on_plan(self, msg: String):
        if self.executing:
            self.get_logger().warning('executor busy, ignoring new plan')
            return
        try:
            obj=json.loads(msg.data)
            motions=obj.get('motions', [])
        except Exception as exc:
            self.get_logger().warning(f'bad plan JSON: {exc}')
            return
        if not motions:
            self.get_logger().warning('plan has no motions')
            return
        self.executing=True
        try:
            self.execute_motions(motions)
        finally:
            self.executing=False

    def execute_motions(self, motions: List[Dict[str, Any]]):
        for idx, motion in enumerate(motions):
            gear=1 if int(motion.get('gear',1)) >= 0 else -1
            steer_deg=float(motion.get('steer_deg',0.0))
            dist_m=abs(float(motion.get('dist_m', motion.get('dist',0.0))))
            if dist_m <= 1e-4:
                continue
            duration=max(0.05, dist_m / max(0.05, self.speed_mps))
            cmd={'type':'drive','speed_mps':self.speed_mps,'gear':gear,'steer_deg':steer_deg,'target_index':idx}
            self.cmd_pub.publish(String(data=json.dumps(cmd)))
            self.get_logger().info(f"segment {idx+1}/{len(motions)}: gear={gear} steer={steer_deg:.1f} dist={dist_m:.3f} duration={duration:.2f}s")
            t_end=time.monotonic()+duration
            while time.monotonic() < t_end and rclpy.ok():
                time.sleep(0.02)
            self.cmd_pub.publish(String(data=json.dumps({'type':'stop','reason':f'segment_{idx}_done'})))
            time.sleep(self.segment_pause_s)
        self.cmd_pub.publish(String(data=json.dumps({'type':'stop','reason':'plan_complete'})))
        self.get_logger().info('motion plan complete')

def main(args=None):
    rclpy.init(args=args)
    node=MotionExecutor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
