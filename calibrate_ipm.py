#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np

# ==========================
# Checkerboard Parameters
# ==========================
CHECKERBOARD = (8, 8)      # inner corners
SQUARE_SIZE = 24.0         # mm

# ==========================
# Zoom Parameters
# ==========================
ZOOM_SIZE = 40             # pixels around cursor
ZOOM_SCALE = 8             # magnification factor

clicked_points = []
mouse_x = 0
mouse_y = 0


class CornerPicker(Node):

    def __init__(self):
        super().__init__('corner_picker')

        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image,
            '/rear_cam/image_raw',
            self.image_callback,
            10
        )

        self.frame = None

        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Zoom", cv2.WINDOW_NORMAL)

        cv2.setMouseCallback(
            "Calibration",
            self.mouse_callback
        )

        self.get_logger().info(
            "Click checkerboard corners in row-major order "
            "(left->right, top->bottom)"
        )

    def mouse_callback(self, event, x, y, flags, param):

        global clicked_points
        global mouse_x, mouse_y

        mouse_x = x
        mouse_y = y

        if event == cv2.EVENT_LBUTTONDOWN:

            clicked_points.append([x, y])

            print(
                f"Point {len(clicked_points):02d}: "
                f"({x}, {y})"
            )

    def image_callback(self, msg):

        global mouse_x, mouse_y

        frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

        # Rotate 180° if rear camera is mounted upside-down
        frame = cv2.flip(frame, -1)

        # Calibration resolution
        frame = cv2.resize(
            frame,
            (1280, 720),
            interpolation=cv2.INTER_AREA
        )
        
        display = frame.copy()

        # -------------------------------------------------
        # Draw selected points
        # -------------------------------------------------
        for i, p in enumerate(clicked_points):

            x = int(p[0])
            y = int(p[1])

            cv2.circle(
                display,
                (x, y),
                4,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                display,
                str(i + 1),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

        # -------------------------------------------------
        # Draw current mouse crosshair
        # -------------------------------------------------
        cv2.line(
            display,
            (mouse_x, 0),
            (mouse_x, display.shape[0]),
            (0, 255, 0),
            1
        )

        cv2.line(
            display,
            (0, mouse_y),
            (display.shape[1], mouse_y),
            (0, 255, 0),
            1
        )

        # -------------------------------------------------
        # Zoom Window
        # -------------------------------------------------
        h, w = display.shape[:2]

        x1 = max(0, mouse_x - ZOOM_SIZE)
        x2 = min(w, mouse_x + ZOOM_SIZE)

        y1 = max(0, mouse_y - ZOOM_SIZE)
        y2 = min(h, mouse_y + ZOOM_SIZE)

        roi = display[y1:y2, x1:x2]

        if roi.size > 0:

            zoom = cv2.resize(
                roi,
                None,
                fx=ZOOM_SCALE,
                fy=ZOOM_SCALE,
                interpolation=cv2.INTER_NEAREST
            )

            zh, zw = zoom.shape[:2]

            # center crosshair
            cv2.line(
                zoom,
                (zw // 2, 0),
                (zw // 2, zh),
                (0, 255, 0),
                1
            )

            cv2.line(
                zoom,
                (0, zh // 2),
                (zw, zh // 2),
                (0, 255, 0),
                1
            )

            cv2.putText(
                zoom,
                f"({mouse_x},{mouse_y})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("Zoom", zoom)

        # -------------------------------------------------
        # Status text
        # -------------------------------------------------
        cv2.putText(
            display,
            f"Points: {len(clicked_points)}/64",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

        cv2.imshow("Calibration", display)

        key = cv2.waitKey(1) & 0xFF

        # Undo last point
        if key == ord('z'):
            if len(clicked_points) > 0:
                removed = clicked_points.pop()
                print("Removed:", removed)

        # Save points
        if key == ord('s'):

            if len(clicked_points) != 64:
                print(
                    f"Need 64 points, "
                    f"currently {len(clicked_points)}"
                )
                return

            img_points = np.array(
                clicked_points,
                dtype=np.float32
            )

            objp = np.zeros(
                (64, 3),
                np.float32
            )

            objp[:, :2] = np.mgrid[
                0:CHECKERBOARD[0],
                0:CHECKERBOARD[1]
            ].T.reshape(-1, 2)

            objp *= SQUARE_SIZE

            np.save(
                "img_points.npy",
                img_points
            )

            np.save(
                "obj_points.npy",
                objp
            )

            print()
            print("Saved:")
            print("  img_points.npy")
            print("  obj_points.npy")
            print()

        self.frame = display


def main():

    rclpy.init()

    node = CornerPicker()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()