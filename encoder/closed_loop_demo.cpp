// ============================================================
//  closed_loop_demo.cpp  —  Closed-Loop Distance Demo
//
//  Build:
//    g++ -std=c++17 -O2 -o closed_loop_demo closed_loop_demo.cpp \
//        -lgpiod -lpthread
//
//  Run:
//    sudo ./closed_loop_demo
//
//  What it does:
//    Move 1: forward  0.50 m  at 0.08 m/s
//    Move 2: pause 1 s
//    Move 3: reverse  0.30 m  at 0.06 m/s
// ============================================================

#include "diff_encoder.hpp"
#include "closed_loop_driver.hpp"

#include <csignal>
#include <cstdio>
#include <thread>
#include <chrono>

static ClosedLoopDriver* g_driver = nullptr;

void on_signal(int) {
    if (g_driver) g_driver->abort();
}

int main() {
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    // ── Encoder config ────────────────────────────────────────
    DiffDriveConfig enc_cfg;
    enc_cfg.chip_name    = "gpiochip4";  // from hb_gpioinfo
    enc_cfg.left_c1      = 27;
    enc_cfg.left_c2      = 17;
    enc_cfg.right_c1     = 24;
    enc_cfg.right_c2     = 23;
    enc_cfg.ppr          = 1000;         // ← your encoder PPR
    enc_cfg.gear_ratio   = 74.25f;       // 74.25 : 1 gearbox
    enc_cfg.wheel_circ_m = 0.20f;        // ← wheel circumference [m]

    // ── Serial port to Arduino ────────────────────────────────
    // Find port: ls /dev/ttyS* /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
    const char* SERIAL_PORT = "/dev/ttyUSB1";

    // ── Closed-loop config ────────────────────────────────────
    ClosedLoopConfig cl_cfg;
    cl_cfg.slowdown_dist_m  = 0.10f;  // Slow down 10 cm before target
    cl_cfg.stop_thresh_m    = 0.02f;  // Stop when within 2 cm
    cl_cfg.min_speed_mps    = 0.03f;  // Min speed during slowdown
    cl_cfg.kp_drift         = 40.0f;  // Drift correction gain
    cl_cfg.drift_deadband_m = 0.005f; // Ignore drift < 5 mm
    cl_cfg.loop_ms          = 20;     // Control loop 20 ms
    cl_cfg.enc_check_time_s = 3.0f;   // Check encoder responds within 3 s

    printf("=== AutoPark Closed-Loop Drive Demo ===\n");
    printf("Serial : %s\n", SERIAL_PORT);
    printf("Encoder: gpiochip4  L=GPIO%u/%u  R=GPIO%u/%u\n",
           enc_cfg.left_c1, enc_cfg.left_c2,
           enc_cfg.right_c1, enc_cfg.right_c2);
    printf("PPR=%d  Gear=%.2f:1  Wheel=%.3f m\n\n",
           enc_cfg.ppr, enc_cfg.gear_ratio, enc_cfg.wheel_circ_m);

    // ── Initialise encoder + serial ───────────────────────────
    DiffDriveEncoder enc(enc_cfg);
    SerialPort       serial(SERIAL_PORT);
    ClosedLoopDriver driver(enc, serial, cl_cfg);
    g_driver = &driver;

    // ── Move 1: Forward 0.50 m ────────────────────────────────
    printf("--- Move 1: FORWARD 0.50 m ---\n");
    DriveResult r1 = driver.driveDistance(
        0.50f,   // target distance [m]
        true,    // forward
        0.08f,   // speed [m/s]
        0.0f,    // steer angle [deg]
        15.0f    // timeout [s]
    );
    printf("Move 1 result: %s  (L=%.4f m  R=%.4f m)\n\n",
           driveResultName(r1), driver.leftDist(), driver.rightDist());

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // ── Move 2: Reverse 0.30 m ────────────────────────────────
    printf("--- Move 2: REVERSE 0.30 m ---\n");
    DriveResult r2 = driver.driveDistance(
        0.30f,   // target distance [m]
        false,   // reverse
        0.06f,   // speed [m/s]
        0.0f,    // steer angle [deg]
        12.0f    // timeout [s]
    );
    printf("Move 2 result: %s  (L=%.4f m  R=%.4f m)\n\n",
           driveResultName(r2), driver.leftDist(), driver.rightDist());

    printf("=== Demo complete ===\n");
    return 0;
}
