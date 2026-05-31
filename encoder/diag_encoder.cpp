// ============================================================
//  diag_encoder.cpp  —  Encoder Pulse Diagnostic
//
//  DOES NOT need serial or Arduino.
//  Just checks if encoder pulses arrive on the GPIO pins.
//
//  HOW TO USE:
//    1. Build and run
//    2. SPIN EACH WHEEL BY HAND slowly
//    3. Watch the isr count go up
//       If count stays 0 → wiring problem on that pin
//       If count goes up  → encoder connected correctly
//
//  Build:
//    g++ -std=c++17 -O2 -o diag_encoder diag_encoder.cpp -lgpiod -lpthread
//
//  Run:
//    sudo ./diag_encoder
// ============================================================

#include <gpiod.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <csignal>
#include <stdexcept>
#include <thread>

static volatile bool g_running = true;
void on_signal(int) { g_running = false; }

// ── One pin monitor ──────────────────────────────────────────
struct PinMonitor {
    const char*       chip_name;
    unsigned          line_num;
    const char*       label;
    gpiod_chip*       chip    = nullptr;
    gpiod_line*       line    = nullptr;
    std::atomic<long> count   {0};
    std::thread       thread_;
    std::atomic<bool> running_{false};

    void start() {
        chip = gpiod_chip_open_by_name(chip_name);
        if (!chip) throw std::runtime_error("Cannot open chip");
        line = gpiod_chip_get_line(chip, line_num);
        if (!line) throw std::runtime_error("Cannot get line");
        if (gpiod_line_request_both_edges_events(line, label) < 0)
            throw std::runtime_error("Cannot request edges");
        running_ = true;
        thread_ = std::thread([this]() {
            struct timespec timeout{ 0, 50'000'000L };
            while (running_) {
                if (gpiod_line_event_wait(line, &timeout) > 0) {
                    struct gpiod_line_event ev;
                    if (gpiod_line_event_read(line, &ev) == 0) count++;
                }
            }
        });
    }

    void stop() {
        running_ = false;
        if (thread_.joinable()) thread_.join();
        if (line) { gpiod_line_release(line); line = nullptr; }
        if (chip) { gpiod_chip_close(chip);   chip = nullptr; }
    }
};

int main() {
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    printf("=== Encoder Pulse Diagnostic ===\n");
    printf("Spin each wheel BY HAND slowly.\n");
    printf("Count going up = encoder connected OK.\n");
    printf("Count stays 0  = wiring problem.\n");
    printf("Ctrl+C to quit.\n\n");

    // ── Define the 4 encoder pins ──────────────────────────────
    PinMonitor pins[4] = {
        {"gpiochip4", 27, "L-C1"},   // Left  motor Channel A
        {"gpiochip4", 17, "L-C2"},   // Left  motor Channel B
        {"gpiochip4", 24, "R-C1"},   // Right motor Channel A
        {"gpiochip4", 23, "R-C2"},   // Right motor Channel B
    };

    // Start all monitors
    for (auto& p : pins) {
        try {
            p.start();
            printf("[OK] Watching gpiochip4 line %-2u  (%s)\n",
                   p.line_num, p.label);
        } catch (const std::exception& e) {
            printf("[ERR] gpiochip4 line %u (%s): %s\n",
                   p.line_num, p.label, e.what());
        }
    }

    printf("\n%-10s %-10s %-10s %-10s   <- spin wheels to see count rise\n",
           "L-C1(27)", "L-C2(17)", "R-C1(24)", "R-C2(23)");
    printf("%-70s\n", "----------------------------------------------------------------------");

    while (g_running) {
        printf("\r%-10ld %-10ld %-10ld %-10ld",
               pins[0].count.load(),
               pins[1].count.load(),
               pins[2].count.load(),
               pins[3].count.load());
        fflush(stdout);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    printf("\n\nStopping...\n");
    for (auto& p : pins) p.stop();

    // ── Summary ───────────────────────────────────────────────
    printf("\n=== Summary ===\n");
    const char* motors[4] = {
        "Left  C1 GPIO27", "Left  C2 GPIO17",
        "Right C1 GPIO24", "Right C2 GPIO23"
    };
    for (int i = 0; i < 4; i++) {
        long c = pins[i].count.load();
        printf("  %-18s : %6ld pulses  %s\n",
               motors[i], c,
               c > 10 ? "OK ✓" : "NO SIGNAL - check wiring ✗");
    }
    return 0;
}
