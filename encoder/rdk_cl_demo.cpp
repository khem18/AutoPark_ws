// ============================================================
//  rdk_cl_demo.cpp  —  Single-Encoder Straight Drive
//
//  Build:
//    g++ -std=c++17 -O2 -o rdk_cl_demo rdk_cl_demo.cpp -lpthread
//
//  Run (kill autopark first):
//    pkill -f serial_bridge; pkill -f autopark_master; sleep 2
//    sudo ./rdk_cl_demo /dev/ttyUSB2 /dev/ttyUSB0
// ============================================================

#include "enc_serial_reader.hpp"
#include "rdk_closed_loop.hpp"

#include <csignal>
#include <cstdio>
#include <thread>
#include <chrono>

static StraightDriver* g_driver = nullptr;
void on_signal(int) { if (g_driver) g_driver->abort(); }

int main(int argc, char** argv) {
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    const char* ENC_PORT   = (argc > 1) ? argv[1] : "/dev/ttyUSB2";
    const char* DRIVE_PORT = (argc > 2) ? argv[2] : "/dev/ttyUSB0";

    printf("=== Single-Encoder Straight Drive ===\n");
    printf("Encoder : %s  (right encoder only)\n", ENC_PORT);
    printf("Motor   : %s\n\n", DRIVE_PORT);

    StraightDriveConfig cfg;
    cfg.slowdown_m    = 0.30f;   // ramp-down starts 30 cm before target
    cfg.stop_thresh_m = 0.02f;   // stop within 2 cm of target
    cfg.min_speed_mps = 0.02f;   // minimum speed during ramp-down
    cfg.speed_kp      = 0.4f;
    cfg.speed_ki      = 0.08f;
    cfg.enc_check_s   = 3.0f;
    cfg.enc_check_m   = 0.005f;
    cfg.arm_wait_ms   = 4000.0f;

    EncSerialReader enc(ENC_PORT);
    DriveSerial     drive(DRIVE_PORT);
    StraightDriver  driver(enc, drive, cfg);
    g_driver = &driver;

    // Reset encoder
    {
        int fd = ::open(ENC_PORT, O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd >= 0) { ::write(fd, "r\n", 2); ::close(fd); }
    }

    printf("Waiting for encoder data...\n");
    for (int i = 0; i < 60 && !enc.isValid(); i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (!enc.isValid()) {
        printf("ERROR: No data on %s\n", ENC_PORT);
        return 1;
    }
    auto snap = enc.getSnapshot();
    printf("Encoder OK  pkts=%u  rc=%ld  rd=%.4fm  rRPM=%.2f\n\n",
           enc.packetCount(), snap.rightCount, snap.rightDistM, snap.rightWRpm);

    // ── Move 1: Forward ─────────────────────────────────────
    // Target = 0.50m.
    // Car coasts ~40cm after stop at speed 0.236 m/s.
    // So set actual encoder target = 0.50m.
    // After PID fix (rightSpeedMs now uses correct circumference),
    // ramp-down should slow car enough to stop close to 0.50m.
    printf("=== Move 1: FORWARD 0.50 m ===\n");
    StraightResult r1 = driver.driveStraight(
        0.50f, true, 0.06f, 15.0f);  // reduced speed 0.08→0.06
    printf("Result: %s  dist=%.4fm\n\n",
           straightResultName(r1), driver.lastDist());

    std::this_thread::sleep_for(std::chrono::seconds(2));

    // ── Move 3: Reverse ──────────────────────────────────────
    printf("=== Move 3: REVERSE 0.30 m ===\n");
    StraightResult r3 = driver.driveStraight(
        0.30f, false, 0.04f, 12.0f);  // reduced speed 0.06→0.04
    printf("Result: %s  dist=%.4fm\n\n",
           straightResultName(r3), driver.lastDist());

    printf("=== Done ===\n");
    return 0;
}
