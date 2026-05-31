#!/bin/bash
# ============================================================
#  find_serial_port.sh  —  Find which port connects to Arduino
#
#  Usage:  sudo bash find_serial_port.sh
# ============================================================

echo "=== Serial Port Finder ==="
echo ""

# ── 1. Show all ports found ──────────────────────────────────
echo "[1] Available serial ports:"
ls /dev/ttyS* /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
echo ""

# ── 2. Check which port AutoPark serial_bridge uses ──────────
echo "[2] Checking AutoPark config for serial port..."
CONFIG_FILE=~/AutoPark_ws/src/autopark_system/config/autopark_params.yaml
if [ -f "$CONFIG_FILE" ]; then
    grep -i "serial\|port\|uart\|tty" "$CONFIG_FILE"
else
    echo "    Config not found at: $CONFIG_FILE"
    echo "    Try: find ~/AutoPark_ws -name '*.yaml' | xargs grep -l 'tty' 2>/dev/null"
fi
echo ""

# ── 3. Check dmesg for USB serial connections ────────────────
echo "[3] Recent USB-serial connections (dmesg):"
dmesg | grep -E 'tty|usb|serial|ch34|cp21|ftdi|arduino' | tail -20
echo ""

# ── 4. Try to read from each USB port ────────────────────────
echo "[4] Testing USB serial ports for Arduino response..."
for PORT in /dev/ttyUSB0 /dev/ttyUSB1 /dev/ttyACM0 /dev/ttyACM1; do
    if [ -e "$PORT" ]; then
        echo ""
        echo "  Testing $PORT ..."
        # Set baud rate and try to read
        stty -F $PORT 115200 raw -echo 2>/dev/null
        # Send arm command and wait 1s for response
        echo '{"type":"status"}' > $PORT 2>/dev/null
        sleep 0.5
        REPLY=$(timeout 1 cat $PORT 2>/dev/null | head -1)
        if [ -n "$REPLY" ]; then
            echo "  ✓ GOT RESPONSE on $PORT: $REPLY"
            echo ""
            echo "  >>> USE THIS PORT: $PORT <<<"
        else
            echo "  ✗ No response on $PORT"
        fi
    fi
done

# ── 5. Check if serial_bridge is already running ─────────────
echo ""
echo "[5] Is AutoPark serial_bridge already using a port?"
ps aux | grep serial_bridge | grep -v grep
ls -la /proc/$(pgrep -f serial_bridge 2>/dev/null)/fd 2>/dev/null | grep tty

echo ""
echo "=== Done ==="
echo "Edit SERIAL_PORT in closed_loop_demo.cpp to the correct port."
