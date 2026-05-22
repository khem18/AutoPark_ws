import serial
import time
import json

PORT = "/dev/ttyUSB1"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=0.2)

# prevent some reset behavior
ser.setDTR(False)
ser.setRTS(False)

print("Waiting ESP32 boot...")
time.sleep(8)

def read_status(sec=2):
    t0 = time.time()
    while time.time() - t0 < sec:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print("ESP:", line)

def send(cmd):
    line = json.dumps(cmd)
    print("SEND:", line)
    ser.write((line + "\n").encode())
    ser.flush()
    read_status(3)

print("Read before command:")
read_status(3)

send({"type":"drive","speed_mps":0.0,"gear":1,"steer_deg":0.0})
send({"type":"drive","speed_mps":0.0,"gear":1,"steer_deg":8.0})
send({"type":"drive","speed_mps":0.0,"gear":1,"steer_deg":-8.0})
send({"type":"drive","speed_mps":0.0,"gear":1,"steer_deg":0.0})

send({"type":"stop","reason":"rdk_steer_test"})

ser.close()
