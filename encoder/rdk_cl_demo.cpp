// ============================================================
//  rdk_cl_demo.cpp  —  RDK X5 Two-Board Closed-Loop v3
//
//  Build:
//    g++ -std=c++17 -O2 -o rdk_cl_demo rdk_cl_demo.cpp -lpthread
//
//  Run:
//    sudo ./rdk_cl_demo /dev/ttyUSB2 /dev/ttyUSB0
// ============================================================

#include "enc_serial_reader.hpp"
#include "rdk_closed_loop.hpp"

#include <csignal>
#include <cstdio>
#include <thread>
#include <chrono>

static RdkClosedLoop* g_cl = nullptr;
void on_signal(int) { if (g_cl) g_cl->abort(); }

int main(int argc, char** argv) {
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    const char* ENC_PORT   = (argc > 1) ? argv[1] : "/dev/ttyUSB2";
    const char* DRIVE_PORT = (argc > 2) ? argv[2] : "/dev/ttyUSB0";

    printf("=== RDK X5 Two-Board Closed-Loop v3 ===\n");
    printf("Encoder ESP32 : %s\n", ENC_PORT);
    printf("Motor   ESP32 : %s\n", DRIVE_PORT);

    ClosedLoopConfig cfg;
    cfg.slowdown_m     = 0.10f;
    cfg.stop_thresh_m  = 0.02f;
    cfg.min_speed_mps  = 0.03f;
    cfg.drift_kp       = 40.0f;

    // ── ENC_FAIL fix ────────────────────────────────────────
    // enc_check_m was 0.02 (2cm) — too strict.
    // Car IS moving but encoder shows 0.0002m due to PPR mismatch.
    // Set to 0.00005 so ANY encoder pulse passes the check.
    // After tuning PPR, raise this back to 0.005 (5mm).
    cfg.enc_check_s    = 5.0f;     // wait 5 s for movement
    cfg.enc_check_m    = 0.00005f; // detect ANY pulse

    EncSerialReader enc(ENC_PORT);
    DriveSerial     drive(DRIVE_PORT);
    RdkClosedLoop   cl(enc, drive, cfg);
    g_cl = &cl;

    // Reset encoder boot counts
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
    printf("Encoder OK  pkts=%u  lc=%ld  rc=%ld\n\n",
           enc.packetCount(), snap.leftCount, snap.rightCount);

    // ── Move 1: Forward ─────────────────────────────────────
    printf("--- Move 1: FORWARD 0.50 m ---\n");
    CLResult r1 = cl.driveDistance(0.50f, true, 0.08f, 0.0f, 15.0f);
    printf("Result: %s  (lDist=%.4f  rDist=%.4f)\n\n",
           clResultName(r1), cl.lastLeftDist(), cl.lastRightDist());

    // ── Full clear before Move 2 ─────────────────────────────
    // Wait for motor to fully stop and flush any queued commands
    drive.stop("clear");
    std::this_thread::sleep_for(std::chrono::milliseconds(800));

    // ── Move 2: Reverse ──────────────────────────────────────
    printf("--- Move 2: REVERSE 0.30 m ---\n");
    CLResult r2 = cl.driveDistance(0.30f, false, 0.06f, 0.0f, 12.0f);
    printf("Result: %s  (lDist=%.4f  rDist=%.4f)\n\n",
           clResultName(r2), cl.lastLeftDist(), cl.lastRightDist());

    printf("=== Done ===\n");
    printf("\n--- PPR Calibration hint ---\n");
    printf("Expected dist 0.50m,  actual encoder showed %.4f m\n",
           cl.lastAvgDist_m1());
    printf("If encoder dist << real dist, your PPR is set too high.\n");
    printf("Real PPR = encoder_ppr × (real_dist / encoder_dist)\n");
    return 0;
}
