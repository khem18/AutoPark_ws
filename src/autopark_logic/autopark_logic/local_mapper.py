import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
import math
import numpy as np

class LocalMapper(Node):
    def __init__(self):
        super().__init__('local_mapper')
        
        # Listen to the camera's raw data
        self.sub = self.create_subscription(Float32MultiArray, '/parking_metrics', self.metrics_callback, 10)
        
        # Publish the Map and the Goal for Hybrid A*
        self.map_pub = self.create_publisher(OccupancyGrid, '/local_map', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # --- MAP PARAMETERS ---
        self.resolution = 0.05  # 5 cm per pixel (Keeps the map lightweight for fast edge computing)
        self.width = 120        # 6 meters wide total
        self.height = 120       # 6 meters long total
        
        self.get_logger().info("🌍 LOCAL MAPPER ONLINE: Bridging Vision to Hybrid A*")

    def euler_to_quaternion(self, yaw):
        # Converts 2D rotation into 3D quaternion for ROS
        qx = 0.0
        qy = 0.0
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return qx, qy, qz, qw

    def metrics_callback(self, msg):
        # Raw Data from Camera: [car_x, car_y, target_x_cm(Right), target_y_cm(Forward), tilt_deg]
        cam_target_x_cm = msg.data[2]
        cam_target_y_cm = msg.data[3]
        tilt_deg = msg.data[4]

        # ==============================================================
        # 1. THE ROS 2 COORDINATE CONVERSION
        # ==============================================================
        ros_target_x_m = cam_target_y_cm / 100.0    # แกน X คือลึกไปข้างหน้า
        ros_target_y_m = -cam_target_x_cm / 100.0   # แกน Y คือซ้าย/ขวา
        
        # องศาของช่องจอด (เอาไว้หมุนรูปตัว U)
        map_yaw_rad = math.radians(-tilt_deg)

        # 🎯 FIX 1: พลิกลูกศร 180 องศาสำหรับการถอยจอด (Reverse Parking)
        # บวก math.pi (180 องศา) เข้าไป เพื่อบังคับให้หน้ารถหันออกจากซอง
        goal_yaw_rad = map_yaw_rad + math.pi

        # ==============================================================
        # 2. PUBLISH THE GOAL POSE
        # ==============================================================
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map" 
        
        goal.pose.position.x = ros_target_x_m
        goal.pose.position.y = ros_target_y_m
        goal.pose.position.z = 0.0
        
        # ใช้ goal_yaw_rad ที่ถูกพลิกแล้วในการสร้างลูกศร
        qx, qy, qz, qw = self.euler_to_quaternion(goal_yaw_rad)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        
        self.goal_pub.publish(goal)

        # ==============================================================
        # 3. CREATE AND PUBLISH THE OCCUPANCY GRID (THE FAKE MAP)
        # ==============================================================
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = "map"
        
        grid.info.resolution = self.resolution
        grid.info.width = self.width
        grid.info.height = self.height
        
        grid.info.origin.position.x = - (self.width * self.resolution) / 2.0  
        grid.info.origin.position.y = - 1.0  
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0

        map_data = np.zeros((self.height, self.width), dtype=np.int8)

        def world_to_grid(wx, wy):
            gx = int((wx - grid.info.origin.position.x) / self.resolution)
            gy = int((wy - grid.info.origin.position.y) / self.resolution)
            return gx, gy

        cx, cy = world_to_grid(ros_target_x_m, ros_target_y_m)
        
        # FIX 2: วาดตัว U ให้ลึกตามแกน X และกว้างตามแกน Y
        wall_thickness = 2 
        spot_depth_cells = int(2.50 / self.resolution)   # ความลึกของซอง (แกน X)
        spot_width_cells = int(0.18 / self.resolution)   # ความกว้างของซอง (แกน Y)
        
        for x_offset in range(-spot_depth_cells, spot_depth_cells):
            for y_offset in range(-spot_width_cells, spot_width_cells):
                
                # วาดกำแพงหลังสุด (+X), กำแพงซ้าย (+Y), กำแพงขวา (-Y)
                if x_offset > spot_depth_cells - wall_thickness or \
                   y_offset < -spot_width_cells + wall_thickness or \
                   y_offset > spot_width_cells - wall_thickness:
                    
                    # หมุนกำแพงตามความเอียงของกล้อง (ใช้ map_yaw_rad แบบไม่กลับหัว)
                    rot_x = x_offset * math.cos(map_yaw_rad) - y_offset * math.sin(map_yaw_rad)
                    rot_y = x_offset * math.sin(map_yaw_rad) + y_offset * math.cos(map_yaw_rad)
                    
                    gx = cx + int(rot_x)
                    gy = cy + int(rot_y)
                    
                    if 0 <= gx < self.width and 0 <= gy < self.height:
                        map_data[gy, gx] = 100
        # Flatten the 2D array into a 1D list and publish!
        grid.data = map_data.flatten().tolist()
        self.map_pub.publish(grid)
        
        self.get_logger().info(f"✅ Map Published! Goal -> X: {ros_target_x_m:.2f}m, Y: {ros_target_y_m:.2f}m")

def main(args=None):
    rclpy.init(args=args)
    node = LocalMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
