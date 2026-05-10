Real-car open-loop adaptation
=============================

What was added:
- src/autopark_system package copied into this workspace
- open-loop motion executor that executes planner motion segments directly
- launch file: `realcar_openloop.launch.py`
- config file with serial ports for drive/ultrasonic ESP32 boards

How to run on RDK:
1. cd ~/AutoPark_ws
2. source /opt/ros/humble/setup.bash
3. rm -rf build install log
4. colcon build --symlink-install
5. source install/setup.bash
6. ros2 launch autopark_system realcar_openloop.launch.py

What it does:
- grayscale camera node starts
- lot_detector starts and publishes /parking_metrics
- serial_bridge reads /dev/ttyUSB0 and /dev/ttyUSB1
- autopark_master waits for start switch and publishes motion plan
- motion_executor runs the planner motions open-loop for the real car

Important:
- planner mode is currently both_sides by default in config
- this is open-loop motion execution, so tune open_loop_speed_mps on the real car carefully
- keep wheels lifted for first tests
