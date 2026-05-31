// ============================================================
//  diff_encoder_demo.cpp  —  RDK X5 Differential Drive Demo
//
//  Build:
//    g++ -std=c++17 -O2 -o diff_encoder_demo diff_encoder_demo.cpp \
//        -lgpiod -lpthread
//
//  Run:
//    sudo ./diff_encoder_demo
//    'r' + Enter to reset counters  |  Ctrl+C to quit
// ============================================================

#include "diff_encoder.hpp"
#include <chrono>
#include <cstdio>
#include <csignal>
#include <thread>

static volatile bool g_running = true;
void on_signal(int) { g_running = false; }

int main() {
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    DiffDriveConfig cfg;
    cfg.chip_name    = "gpiochip4";
    cfg.left_c1      = 27;
    cfg.left_c2      = 17;
    cfg.right_c1     = 24;
    cfg.right_c2     = 23;
    cfg.ppr          = 1000;       // ← set to your encoder PPR
    cfg.gear_ratio   = 74.25f;     // 74.25 : 1 gearbox
    cfg.wheel_circ_m = 0.20f;      // ← wheel circumference [m]

    printf("=== RDK X5 Differential Drive Encoder ===\n");
    printf("Chip  : %s\n",           cfg.chip_name);
    printf("Left  : C1=line%u  C2=line%u\n", cfg.left_c1,  cfg.left_c2);
    printf("Right : C1=line%u  C2=line%u\n", cfg.right_c1, cfg.right_c2);
    printf("PPR   : %d  |  Gear: %.2f:1  |  Wheel: %.3f m\n\n",
           cfg.ppr, cfg.gear_ratio, cfg.wheel_circ_m);

    DiffDriveEncoder enc(cfg);

    printf("%-4s | %-8s %-8s %-10s %-10s | %-8s %-8s %-10s %-10s\n",
           "Dir",
           "L-mRPM","L-wRPM","L-m/s","L-dist(m)",
           "R-mRPM","R-wRPM","R-m/s","R-dist(m)");
    printf("%.80s\n",
           "--------------------------------------------------------------------------------");

    std::thread input([&enc]() {
        char buf[8];
        while (fgets(buf, sizeof(buf), stdin))
            if (buf[0]=='r' || buf[0]=='R') {
                enc.resetAll();
                printf("\n[Reset] Counters zeroed.\n");
            }
    });
    input.detach();

    while (g_running) {
        const char* d = enc.leftDir() > 0 ? "FWD" :
                        enc.leftDir() < 0 ? "REV" : "---";

        printf("\r%-4s | %-8.1f %-8.2f %-10.4f %-10.4f | %-8.1f %-8.2f %-10.4f %-10.4f",
               d,
               enc.leftMotorRpm(),  enc.leftWheelRpm(),
               enc.leftSpeedMs(),   enc.leftDistM(),
               enc.rightMotorRpm(), enc.rightWheelRpm(),
               enc.rightSpeedMs(),  enc.rightDistM());
        fflush(stdout);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    printf("\n[Stopped]\n");
    return 0;
}
