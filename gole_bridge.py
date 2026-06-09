import cv2

# Load the calibration image saved by your ROS 2 node
img = cv2.imread('/tmp/cal.png')

if img is None:
    print("Error: Could not open /tmp/cal.png. Make sure your ROS node is running and hits frame #30!")
    exit()

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked coordinate: [{float(x)}, {float(y)}]")
        # Draw a little circle and the text on the image so you see where you clicked
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, f"{x},{y}", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('Calibration Window', img)

print("INSTRUCTIONS: Click the 4 corners of your real-world floor rectangle in order:")
print("1. Bottom-Left (BL) -> 2. Bottom-Right (BR) -> 3. Top-Left (TL) -> 4. Top-Right (TR)")

cv2.imshow('Calibration Window', img)
cv2.setMouseCallback('Calibration Window', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()