import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from nav2_msgs.action import ComputePathToPose

class GoalBridge(Node):
    def __init__(self):
        super().__init__('goal_bridge')
        # ดักฟัง Topic จากปุ่ม 2D Goal Pose ใน RViz
        self.subscription = self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)
        # เตรียมส่งคำสั่งให้ Planner
        self.action_client = ActionClient(self, ComputePathToPose, '/compute_path_to_pose')
        self.get_logger().info('พร้อมแล้ว! จิ้ม 2D Goal Pose ใน RViz ได้เลย...')

    def goal_cb(self, msg):
        self.get_logger().info('รับทราบเป้าหมาย! กำลังส่งให้สมองกลคิดเส้นทาง...')
        self.action_client.wait_for_server()
        goal_msg = ComputePathToPose.Goal()
        goal_msg.goal = msg
        self.action_client.send_goal_async(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GoalBridge()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
