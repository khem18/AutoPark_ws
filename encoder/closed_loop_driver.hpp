#pragma once
// ============================================================
//  closed_loop_driver.hpp  —  Encoder Closed-Loop Drive
//
//  What it does:
//    1. Send "drive" JSON command to Arduino via serial
//    2. Monitor encoder distance every 20 ms (control loop)
//    3. Ramp down speed in last 10 cm
//    4. Correct left/right drift with steering trim
//    5. Send "stop" JSON when target distance is reached
//
//  Result: car drives accurate real-world distances
//  instead of guessing from time × speed.
//
//  Integration:
//    Call driveDistance() from autopark_master instead of
//    sending a time-based "drive" command directly.
//
//  Build:
//    g++ -std=c++17 -O2 -o closed_loop_demo closed_loop_demo.cpp \
//        -lgpiod -lpthread
// ============================================================

#include "diff_encoder.hpp"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

// ============================================================
//  Drive result codes
// ============================================================
enum class DriveResult {
    SUCCESS,        // Reached target distance within threshold
    TIMEOUT,        // Timed out before reaching target
    ENCODER_FAIL,   // Encoder gave no pulses (wiring problem)
    ABORTED         // abort() was called externally
};

inline const char* driveResultName(DriveResult r) {
    switch (r) {
        case DriveResult::SUCCESS:      return "SUCCESS";
        case DriveResult::TIMEOUT:      return "TIMEOUT";
        case DriveResult::ENCODER_FAIL: return "ENCODER_FAIL";
        case DriveResult::ABORTED:      return "ABORTED";
        default:                        return "UNKNOWN";
    }
}

// ============================================================
//  SerialPort  —  POSIX serial to Arduino
//
//  Common ports on RDK X5:
//    /dev/ttyS3    — hardware UART (115200 default)
//    /dev/ttyUSB0  — USB-serial adapter
//    /dev/ttyACM0  — USB ACM (CDC)
//
//  Find yours: ls /dev/tty* | grep -E 'ttyS|USB|ACM'
// ============================================================
class SerialPort {
public:
    SerialPort(const char* port, int baud_const = B115200) {
        fd_ = open(port, O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_ < 0)
            throw std::runtime_error(
                std::string("Cannot open serial port: ") + port +
                "  (run with sudo, check port with: ls /dev/tty*)");

        struct termios tty{};
        tcgetattr(fd_, &tty);
        cfsetispeed(&tty, baud_const);
        cfsetospeed(&tty, baud_const);
        tty.c_cflag  = (tty.c_cflag & ~CSIZE) | CS8;
        tty.c_cflag |= CLOCAL | CREAD;
        tty.c_cflag &= ~(PARENB | PARODD | CSTOPB | CRTSCTS);
        tty.c_iflag  = IGNPAR;
        tty.c_oflag  = 0;
        tty.c_lflag  = 0;
        tty.c_cc[VMIN]  = 0;
        tty.c_cc[VTIME] = 0;
        tcsetattr(fd_, TCSANOW, &tty);
        tcflush(fd_, TCIOFLUSH);
    }

    ~SerialPort() { if (fd_ >= 0) close(fd_); }

    SerialPort(const SerialPort&)            = delete;
    SerialPort& operator=(const SerialPort&) = delete;

    // Write a JSON line (auto-appends '\n')
    void writeLine(const char* json) {
        write(fd_, json, strlen(json));
        write(fd_, "\n", 1);
    }

    // Read available bytes into buf (non-blocking)
    int readAvailable(char* buf, int maxlen) {
        return read(fd_, buf, maxlen);
    }

private:
    int fd_ = -1;
};


// ============================================================
//  ClosedLoopConfig  —  top-level struct (avoids GCC error)
// ============================================================
struct ClosedLoopConfig {
    // Distance thresholds
    float slowdown_dist_m  = 0.10f;  // Start slowing 10 cm before target
    float stop_thresh_m    = 0.02f;  // Stop when within 2 cm

    // Speed limits
    float min_speed_mps    = 0.03f;  // Slowest speed during ramp-down
    float max_steer_trim   = 8.0f;   // Maximum drift-correction steer angle [deg]

    // Drift correction
    float kp_drift         = 40.0f;  // Steer trim gain (deg per metre of drift)
    float drift_deadband_m = 0.005f; // Ignore drift < 5 mm

    // Timing
    int   loop_ms          = 20;     // Control loop period [ms]
    float drive_cmd_dur    = 0.3f;   // Duration field in each drive JSON [s]
                                     // (Arduino stops if no refresh in time)

    // Encoder validity check
    float enc_check_dist_m = 0.05f;  // After driving this far, verify encoder moved
    float enc_check_time_s = 3.0f;   // Seconds to wait for encoder check
};


// ============================================================
//  ClosedLoopDriver
// ============================================================
class ClosedLoopDriver {
public:
    using Config = ClosedLoopConfig;

    ClosedLoopDriver(DiffDriveEncoder& enc,
                     SerialPort&       serial,
                     Config            cfg = Config{})
        : enc_(enc), serial_(serial), cfg_(cfg), abort_(false) {}

    // ── driveDistance ──────────────────────────────────────────
    //
    //  target_m   : distance to drive [m]  (always positive)
    //  forward    : true = forward, false = reverse
    //  speed_mps  : target speed (e.g. 0.08 m/s)
    //  steer_deg  : base steering angle (0 = straight)
    //  timeout_s  : safety timeout [s]
    //
    DriveResult driveDistance(float target_m,
                               bool  forward,
                               float speed_mps,
                               float steer_deg  = 0.0f,
                               float timeout_s  = 15.0f)
    {
        abort_ = false;

        // Arm the Arduino auto mode
        serial_.writeLine("{\"type\":\"arm\"}");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Zero encoders at start of move
        enc_.resetAll();

        auto  t_start   = now_ms();
        auto  t_timeout = static_cast<uint64_t>(timeout_s * 1000.0f);
        int   gear      = forward ? 1 : -1;
        bool  enc_ok    = false;

        DriveResult result = DriveResult::TIMEOUT;

        printf("[CL] Start  target=%.3f m  gear=%+d  speed=%.3f m/s  steer=%.1f°\n",
               target_m, gear, speed_mps, steer_deg);

        while (!abort_) {
            // ── Distance measurement ──────────────────────────
            float dl = std::fabs(enc_.leftDistM());
            float dr = std::fabs(enc_.rightDistM());
            float dist      = (dl + dr) * 0.5f;   // average both wheels
            float remaining = target_m - dist;

            // ── Encoder validity check ────────────────────────
            // If we've been driving for enc_check_time_s and
            // neither wheel has moved enc_check_dist_m, report
            // ENCODER_FAIL (probably wiring issue).
            float elapsed_s = (now_ms() - t_start) / 1000.0f;
            if (!enc_ok && elapsed_s > cfg_.enc_check_time_s) {
                if (dl < cfg_.enc_check_dist_m && dr < cfg_.enc_check_dist_m) {
                    printf("[CL] ENCODER_FAIL — no pulses after %.1f s\n", elapsed_s);
                    result = DriveResult::ENCODER_FAIL;
                    break;
                }
                enc_ok = true;
            }

            // ── Reached target ────────────────────────────────
            if (dist > 0.001f && remaining <= cfg_.stop_thresh_m) {
                printf("[CL] SUCCESS  dist=%.4f m  remaining=%.4f m\n",
                       dist, remaining);
                result = DriveResult::SUCCESS;
                break;
            }

            // ── Timeout ───────────────────────────────────────
            if (now_ms() - t_start > t_timeout) {
                printf("[CL] TIMEOUT  dist=%.4f m  remaining=%.4f m\n",
                       dist, remaining);
                result = DriveResult::TIMEOUT;
                break;
            }

            // ── Speed ramp-down near target ───────────────────
            float speed = speed_mps;
            if (remaining < cfg_.slowdown_dist_m) {
                float ratio = remaining / cfg_.slowdown_dist_m;
                speed = cfg_.min_speed_mps +
                        ratio * (speed_mps - cfg_.min_speed_mps);
                speed = std::max(speed, cfg_.min_speed_mps);
            }

            // ── Drift correction ──────────────────────────────
            // drift > 0: left wheel ahead → car drifting right → steer left (-)
            // drift < 0: right wheel ahead → car drifting left → steer right (+)
            float steer_cmd = steer_deg;
            float drift     = dl - dr;

            if (std::fabs(drift) > cfg_.drift_deadband_m) {
                float trim = -drift * cfg_.kp_drift;  // negative: correct back
                trim = std::max(-cfg_.max_steer_trim,
                           std::min( cfg_.max_steer_trim, trim));
                steer_cmd += trim;
            }

            // ── Send drive command ────────────────────────────
            char json[320];
            snprintf(json, sizeof(json),
                "{\"type\":\"drive\","
                "\"speed_mps\":%.4f,"
                "\"gear\":%d,"
                "\"steer_deg\":%.2f,"
                "\"duration\":%.2f,"
                "\"steer_active_hold\":true}",
                speed, gear, steer_cmd, cfg_.drive_cmd_dur);
            serial_.writeLine(json);

            // ── Debug print every 200 ms ──────────────────────
            static uint64_t last_print = 0;
            if (now_ms() - last_print > 200) {
                printf("[CL] dist=%.4f  rem=%.4f  speed=%.3f  drift=%.4f  steer=%.1f\n",
                       dist, remaining, speed, drift, steer_cmd);
                last_print = now_ms();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.loop_ms));
        }

        // Always stop
        stop();
        return result;
    }

    // ── Immediate stop ─────────────────────────────────────────
    void stop(const char* reason = "distance_reached") {
        char json[128];
        snprintf(json, sizeof(json),
                 "{\"type\":\"stop\",\"reason\":\"%s\"}", reason);
        serial_.writeLine(json);
    }

    // ── Abort from another thread ─────────────────────────────
    void abort() { abort_ = true; }

    // ── Convenience getters ───────────────────────────────────
    float leftDist()  const { return enc_.leftDistM(); }
    float rightDist() const { return enc_.rightDistM(); }
    float avgDist()   const { return (leftDist() + rightDist()) * 0.5f; }
    float leftRpm()   const { return enc_.leftWheelRpm(); }
    float rightRpm()  const { return enc_.rightWheelRpm(); }

private:
    static uint64_t now_ms() {
        using namespace std::chrono;
        return static_cast<uint64_t>(
            duration_cast<milliseconds>(
                steady_clock::now().time_since_epoch()).count());
    }

    DiffDriveEncoder& enc_;
    SerialPort&       serial_;
    Config            cfg_;
    std::atomic<bool> abort_;
};
