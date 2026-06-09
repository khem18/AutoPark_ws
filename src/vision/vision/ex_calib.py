import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class AutoCalibrator(Node):
    def __init__(self):
        super().__init__('auto_calibrate')
        # Change to your camera topic if needed (e.g., /image_raw)
        self.subscription = self.create_subscription(Image, 'side_cam/image_raw', self.process_frame, 10)
        self.bridge = CvBridge()
        
        # Checkerboard setup
        self.board_size = (8, 8)
        self.square_size_cm = 2.4 
        
        self.get_logger().info("🏁 Auto-Calibrator Started! Hold the 8x6 checkerboard flat on the floor in front of the kart...")
        self.matrix_calculated = False
        self.frame_count = 0  # Our lag-fixing frame counter!

    def process_frame(self, data):
        if self.matrix_calculated:
            return 
            
        # SKIPPING LOGIC: Only do heavy math every 15 frames (twice a second)
        self.frame_count += 1
        if self.frame_count % 15 != 0:
            raw_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            frame = cv2.resize(raw_image, (1920, 1080))
            cv2.imshow("Auto-Calibrator", frame)
            cv2.waitKey(1)
            return
            
        # If it IS the 15th frame, do the heavy processing:
        raw_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        frame = cv2.resize(raw_image, (1920, 1080))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. FIND THE CHECKERBOARD (With FAST_CHECK enabled)
        ret, corners = cv2.findChessboardCorners(
            gray, 
            self.board_size, 
            flags=cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH
        )

        if ret:
            # Refine the corner locations to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw the rainbow lines on the screen so you know it worked
            cv2.drawChessboardCorners(frame, self.board_size, corners, ret)
            
            # 2. GRAB THE 4 OUTERMOST CORNERS (src_pts)
            pt_TL = corners[0][0]
            pt_TR = corners[7][0]
            pt_BL = corners[40][0]
            pt_BR = corners[47][0]
            
            src_pts = np.float32([pt_BL, pt_BR, pt_TL, pt_TR])

            # 3. DEFINE THE PERFECT PHYSICAL RECTANGLE (dst_pts)
            width_cm = 7 * self.square_size_cm   # 17.5 cm
            height_cm = 5 * self.square_size_cm  # 12.5 cm
            
            margin_x = 220 # Push it to the middle of the screen
            margin_y = 250
            
            dst_pts = np.float32([
                [margin_x, margin_y + height_cm],             # Bottom-Left
                [margin_x + width_cm, margin_y + height_cm],  # Bottom-Right
                [margin_x, margin_y],                         # Top-Left
                [margin_x + width_cm, margin_y]               # Top-Right
            ])

            # 4. CALCULATE AND PRINT THE RESULTS
            self.get_logger().info("\n\n✅ CHECKERBOARD DETECTED! COPY THESE ARRAYS INTO YOUR LOT DETECTOR:\n")
            print(f"src_pts = np.float32([\n    [{pt_BL[0]:.1f}, {pt_BL[1]:.1f}], \n    [{pt_BR[0]:.1f}, {pt_BR[1]:.1f}], \n    [{pt_TL[0]:.1f}, {pt_TL[1]:.1f}], \n    [{pt_TR[0]:.1f}, {pt_TR[1]:.1f}]\n])")
            print(f"\ndst_pts = np.float32([\n    [{dst_pts[0][0]}, {dst_pts[0][1]}], \n    [{dst_pts[1][0]}, {dst_pts[1][1]}], \n    [{dst_pts[2][0]}, {dst_pts[2][1]}], \n    [{dst_pts[3][0]}, {dst_pts[3][1]}]\n])")
            print("\nNOTE: Because dst_pts is measured in exact cm, your 'cm_per_pixel' multiplier in the lot detector is now exactly 1.0!")
            
            self.matrix_calculated = True

        cv2.imshow("Auto-Calibrator", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = AutoCalibrator()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
