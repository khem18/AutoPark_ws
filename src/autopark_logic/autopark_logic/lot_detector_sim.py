import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import numpy as np
import math

class LotDetectorSim(Node):
    def __init__(self):
        super().__init__('lot_detector_sim')
        
        # --- GAZEBO & REAL WORLD PARAMETER ---
        # This allows us to change the camera topic from the terminal without editing the code!
        self.declare_parameter('camera_topic', '/front_cam/image_raw')
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        
        # --- THE ANTI-LAG QoS PROFILE ---
        anti_lag_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1  
        )
        
        self.subscription = self.create_subscription(
            Image, 
            camera_topic,  # <-- Now uses the dynamic parameter!
            self.listener_callback, 
            anti_lag_qos
        )
        self.bridge = CvBridge()
        
        # --- CREATE THE PARKING METRICS PUBLISHER ---
        # NOTE: Topic matches the Open-Loop Parker dashboard!
        self.metrics_pub = self.create_publisher(Float32MultiArray, '/parking_metrics', 10)
        
        # --- AWB MEMORY (Required for smooth color transitions!) ---
        self.gain_b = 1.0
        self.gain_g = 1.0
        self.gain_r = 1.0
        
        self.get_logger().info("Lot Detector Started! Safety Bypass ACTIVE. Publishing to /parking_metrics")

    def listener_callback(self, data):
        # 1. READ AND RESIZE CAMERA FEED
        raw_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        frame = cv2.resize(raw_image, (640, 480))

        # ==========================================
        # THE RDK X5 CAMERA FLIP HACK
        # ==========================================
        frame = cv2.flip(frame, 1) 

        # --- HIGH-SPEED DYNAMIC AWB (SMOOTHED) ---
        tiny_frame = cv2.resize(frame, (64, 48))
        b_tiny, g_tiny, r_tiny = cv2.split(tiny_frame)
        
        avg_b = np.mean(b_tiny)
        avg_g = np.mean(g_tiny)
        avg_r = np.mean(r_tiny)
        avg_all = (avg_b + avg_g + avg_r) / 3.0
        
        target_b = avg_all / avg_b if avg_b > 0 else 1.0
        target_g = avg_all / avg_g if avg_g > 0 else 1.0
        target_r = avg_all / avg_r if avg_r > 0 else 1.0
        
        self.gain_b = (0.5 * self.gain_b) + (0.5 * target_b)
        self.gain_g = (0.5 * self.gain_g) + (0.5 * target_g)
        self.gain_r = (0.5 * self.gain_r) + (0.5 * target_r)
        
        b, g, r = cv2.split(frame)
        b = cv2.convertScaleAbs(b, alpha=self.gain_b)
        g = cv2.convertScaleAbs(g, alpha=self.gain_g)
        r = cv2.convertScaleAbs(r, alpha=self.gain_r)
        
        frame = cv2.merge([b, g, r])

        # ==============================================================
        # 2. BIRD'S EYE VIEW (IPM) TRANSFORM
        # ==============================================================
        
        # SOURCE POINTS: Tracing the yellow lines
        src_pts = np.float32([
            [20.0, 380.0],   # Bottom-Left
            [620.0, 380.0],  # Bottom-Right
            [210.0, 210.0],  # Top-Left 
            [430.0, 210.0]   # Top-Right 
        ])

        # DESTINATION POINTS: Force lines to be perfectly straight vertical
        dst_pts = np.float32([
            [200.0, 480.0],  
            [440.0, 480.0],  
            [200.0, 0.0],    
            [440.0, 0.0]     
        ])
        
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        bev_frame = cv2.warpPerspective(frame, matrix, (640, 480))

        # ==============================================================
        # ---> CREATE THE "REAL FLOOR" COOKIE CUTTER MASK
        # ==============================================================
        white_canvas = np.ones((480, 640), dtype=np.uint8) * 255
        floor_mask = cv2.warpPerspective(white_canvas, matrix, (640, 480))
        kernel_shrink = np.ones((10, 10), np.uint8)
        floor_mask = cv2.erode(floor_mask, kernel_shrink)

        # 3. DETECT YELLOW LINES (ANTI-SKIN & FILL-FACTOR GEOMETRY)
        blurred_bev = cv2.medianBlur(bev_frame, 21)
        hsv = cv2.cvtColor(blurred_bev, cv2.COLOR_BGR2HSV)
        
        # --- NIGHT VISION (CLAHE) ---
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        hsv_enhanced = cv2.merge([h, s, v])
        
        # ANTI-SKIN TONE & STATIC
        lower_yellow = np.array([18, 40, 30])  
        upper_yellow = np.array([45, 255, 255])
        yellow_mask = cv2.inRange(hsv_enhanced, lower_yellow, upper_yellow)
        
        # ERASER & BAND-AID
        kernel_eraser = np.ones((7, 7), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel_eraser)
        
        kernel_vertical = np.ones((40, 5), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel_vertical)
        
        # SLICE OFF THE OVERHANG
        yellow_mask = cv2.bitwise_and(yellow_mask, floor_mask)
        
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_lines = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800:
                continue
                
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (width, height), angle = rect
            
            length = max(width, height)
            thickness = min(width, height)
            
            if thickness == 0:
                continue
                
            aspect_ratio = length / thickness
            box_area = length * thickness
            fill_factor = area / box_area if box_area > 0 else 0

            if length > 80 and thickness < 70 and aspect_ratio > 3.0 and fill_factor > 0.65:
                valid_lines.append({
                    'cx': cx, 'cy': cy, 
                    'thickness': thickness, 'rect': rect
                })
                
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(bev_frame, [box], 0, (0, 0, 255), 2)

        valid_lines = sorted(valid_lines, key=lambda r: r['cx'])
        overlay = bev_frame.copy()

        line1 = None
        line2 = None

        # 4. MEASURE THE TRUE PERPENDICULAR INNER GAP
        if len(valid_lines) >= 2:
            line1 = valid_lines[0]
            
            for l in valid_lines[1:]:
                if l['cx'] > (line1['cx'] + 50):
                    line2 = l
                    break
                    
            if line2 is not None:
                box1 = cv2.boxPoints(line1['rect'])
                
                d01 = np.linalg.norm(box1[0] - box1[1])
                d12 = np.linalg.norm(box1[1] - box1[2])
                
                if d01 > d12: 
                    vx = box1[1][0] - box1[0][0]
                    vy = box1[1][1] - box1[0][1]
                else:         
                    vx = box1[2][0] - box1[1][0]
                    vy = box1[2][1] - box1[1][1]
                    
                mag = np.sqrt(vx**2 + vy**2)
                if mag > 0:
                    vx, vy = vx / mag, vy / mag
                else:
                    vx, vy = 0, 1
                
                nx, ny = -vy, vx
                if nx < 0:
                    nx, ny = -nx, -ny
                
                dx = line2['cx'] - line1['cx']
                dy = line2['cy'] - line1['cy']
                
                center_distance = abs(dx * nx + dy * ny)
                
                offset1 = line1['thickness'] / 2
                offset2 = line2['thickness'] / 2
                pixel_gap = center_distance - offset1 - offset2
                
                # --- CALIBRATED SCALES ---
                cm_per_pixel_x = 0.200
                cm_per_pixel_y = 0.247  
                
                real_width_cm = int(pixel_gap * cm_per_pixel_x)

                if real_width_cm > 20:
                    start_x = int(line1['cx'] + nx * offset1)
                    start_y = int(line1['cy'] + ny * offset1)
                    
                    end_x = int(start_x + nx * pixel_gap)
                    end_y = int(start_y + ny * pixel_gap)
                    
                    # DRAWING: Yellow measuring tape
                    cv2.line(overlay, (start_x, start_y), (end_x, end_y), (0, 255, 255), 3)

                    # 1. FIND THE "FRONT GATE" 
                    target_x = start_x + (end_x - start_x) // 2
                    target_y = start_y + (end_y - start_y) // 2
                    
                    box2 = cv2.boxPoints(line2['rect'])
                    box1_sorted = sorted(box1, key=lambda pt: pt[1])
                    box2_sorted = sorted(box2, key=lambda pt: pt[1])
                    
                    l1_bottom_y = (box1_sorted[2][1] + box1_sorted[3][1]) / 2
                    l2_bottom_y = (box2_sorted[2][1] + box2_sorted[3][1]) / 2
                    
                    l1_bottom_x = (box1_sorted[2][0] + box1_sorted[3][0]) / 2
                    l2_bottom_x = (box2_sorted[2][0] + box2_sorted[3][0]) / 2
                    
                    gate_x = int((l1_bottom_x + l2_bottom_x) / 2)
                    gate_y = int(max(l1_bottom_y, l2_bottom_y)) 
                    
                    line_vx, line_vy = vx, vy
                    if line_vy > 0: 
                        line_vx, line_vy = -line_vx, -line_vy
                        
                    screen_angle_rad = math.atan2(line_vx, -line_vy)
                    tilt_degrees = math.degrees(screen_angle_rad)
                        
                    cv2.circle(overlay, (gate_x, gate_y), 8, (0, 0, 255), -1)
                    cv2.line(overlay, (target_x, target_y), (gate_x, gate_y), (0, 255, 0), 1)

                    # 2. CALCULATE KART DISTANCE TO FRONT GATE
                    dist_outward_cm = (480 - gate_y) * cm_per_pixel_y
                    dist_along_cm = (320 - gate_x) * cm_per_pixel_x
                    
                    cam_offset_y = -4.0   
                    cam_offset_x = 56.0   
                    
                    kart_y_fwd = dist_along_cm + cam_offset_y
                    kart_x_right = dist_outward_cm + cam_offset_x 
                    
                    k_y = int(kart_y_fwd)
                    k_x = int(kart_x_right)
                    k_tlt = int(tilt_degrees)
                    
                    # ==============================================================
                    # 3. PUBLISH THE DATA ARRAY (Start Point, End Point, Tilt)
                    # ==============================================================
                    
                    # ONLY publish if the spot is between 26cm and 85cm (Valid Spot)
                    if 26 <= real_width_cm <= 85:
                        metrics_msg = Float32MultiArray()
                        
                        car_start_x = 0.0
                        car_start_y = 0.0
                        end_target_x = float(k_x)
                        end_target_y = float(k_y)
                        kart_tilt = float(k_tlt)
                        
                        metrics_msg.data = [
                            car_start_x, 
                            car_start_y, 
                            end_target_x, 
                            end_target_y, 
                            kart_tilt
                        ]
                        
                        self.metrics_pub.publish(metrics_msg)
                    
                    # ==============================================================
                    # DISPLAY TEXT ON SCREEN
                    # ==============================================================
                    text_x = start_x + (end_x - start_x) // 2
                    text_y = start_y + (end_y - start_y) // 2 - 15
                    cv2.putText(overlay, f"WIDTH: {real_width_cm} cm", (text_x - 70, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    if 10 <= real_width_cm <= 85:
                        cv2.putText(overlay, "PERFECT SPOT!", (text_x - 70, text_y + 45), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.putText(overlay, f"WALL: X={k_x}cm, Y={k_y}cm", (20, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(overlay, f"TILT: {k_tlt} DEG", (20, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # ==============================================================
        # 5. SPATIAL AWARENESS: "AVERAGE PIXEL" ANOMALY DETECTION
        # ==============================================================
        
        safe_floor_mask = cv2.bitwise_and(floor_mask, cv2.bitwise_not(yellow_mask))
        avg_bgr = cv2.mean(blurred_bev, mask=safe_floor_mask)[:3]
        avg_background = np.full(bev_frame.shape, avg_bgr, dtype=np.uint8)
        
        diff = cv2.absdiff(blurred_bev, avg_background)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        _, object_mask = cv2.threshold(diff_gray, 40, 255, cv2.THRESH_BINARY)
        
        object_mask = cv2.bitwise_and(object_mask, safe_floor_mask)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel_eraser)
        
        obj_contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # ---> SAFETY BYPASS <---
        # Set to False to stop the red "DANGER" boxes from drawing and blocking testing
        run_safety_check = False 
        
        if run_safety_check and line1 is not None and line2 is not None:
            left_boundary_x = line1['cx']
            right_boundary_x = line2['cx']
            
            for ocnt in obj_contours:
                if cv2.contourArea(ocnt) > 800: 
                    M = cv2.moments(ocnt)
                    if M["m00"] > 0:
                        obj_cx = int(M["m10"] / M["m00"])
                        obj_cy = int(M["m01"] / M["m00"])
                        
                        if obj_cx < left_boundary_x:
                            zone_text = "LEFT SIDE LOT"
                            color = (255, 100, 0) 
                            
                        elif obj_cx > right_boundary_x:
                            zone_text = "RIGHT SIDE LOT"
                            color = (255, 100, 0) 
                            
                        else:
                            zone_text = "DANGER: INSIDE LOT!"
                            color = (0, 0, 255) 
                            
                        x, y, w, h = cv2.boundingRect(ocnt)
                        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 3)
                        cv2.circle(overlay, (obj_cx, obj_cy), 5, color, -1)
                        cv2.putText(overlay, zone_text, (x - 20, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.addWeighted(overlay, 0.4, bev_frame, 0.6, 0, bev_frame)

        # 6. SHOW THE FEEDS
        cv2.imshow("1. Normal img", frame)
        cv2.imshow("2. BEV img", bev_frame)
        cv2.imshow("3. Line Mask", yellow_mask)
        cv2.imshow("4. Anomaly Mask", object_mask) 
        cv2.waitKey(1)
        
def main(args=None):
    rclpy.init(args=args)
    node = LotDetectorSim()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
