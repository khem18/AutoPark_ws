#pragma once
// ============================================================
//  rdk_closed_loop.hpp  —  v4  Single-Encoder Straight Drive
//
//  Uses RIGHT encoder only (PPR=11, CHANGE, ~1610 counts/rev).
//  Both motors run at same PWM via steer_deg=0 → equal speed.
//  Speed PID holds target RPM using encoder feedback.
//
//  Designed for Move 1 (forward) and Move 3 (reverse) which
//  are both straight lines in the autopark sequence.
// ============================================================

#include "enc_serial_reader.hpp"

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
//  DriveSerial — sends JSON to Motor ESP32
// ============================================================
class DriveSerial {
public:
    explicit DriveSerial(const char* port, int baud = B115200) {
        fd_ = ::open(port, O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_ < 0)
            throw std::runtime_error(
                std::string("DriveSerial: cannot open ") + port);
        struct termios tty{};
        tcgetattr(fd_, &tty);
        cfsetispeed(&tty, baud); cfsetospeed(&tty, baud);
        tty.c_cflag  = (tty.c_cflag & ~CSIZE) | CS8;
        tty.c_cflag |= CLOCAL | CREAD;
        tty.c_cflag &= ~(PARENB | PARODD | CSTOPB | CRTSCTS);
        tty.c_iflag  = IGNPAR; tty.c_oflag = 0; tty.c_lflag = 0;
        tty.c_cc[VMIN] = 0; tty.c_cc[VTIME] = 0;
        tcsetattr(fd_, TCSANOW, &tty);
        tcflush(fd_, TCIOFLUSH);
    }
    ~DriveSerial() { if (fd_ >= 0) ::close(fd_); }

    void writeLine(const char* json) {
        ::write(fd_, json, strlen(json));
        ::write(fd_, "\n", 1);
    }
    void arm()  { writeLine("{\"type\":\"arm\"}"); }
    void stop(const char* reason = "distance_reached") {
        char buf[128];
        snprintf(buf, sizeof(buf),
                 "{\"type\":\"stop\",\"reason\":\"%s\"}", reason);
        writeLine(buf);
    }
    // Straight drive — steer_deg always 0 for equal wheel speed
    void driveStraight(float speedMps, int gear, float durationS) {
        driveWithSteer(speedMps, gear, 0.0f, durationS);
    }
    // Drive with explicit steer angle (for curved Move1)
    void driveWithSteer(float speedMps, int gear, float steerDeg, float durationS) {
        char buf[256];
        snprintf(buf, sizeof(buf),
            "{\"type\":\"drive\","
            "\"speed_mps\":%.4f,\"gear\":%d,"
            "\"steer_deg\":%.2f,\"duration\":%.2f,"
            "\"steer_active_hold\":true}",
            speedMps, gear, steerDeg, durationS);
        writeLine(buf);
    }
private:
    int fd_ = -1;
};


// ============================================================
//  StraightDriveConfig
// ============================================================
struct StraightDriveConfig {
    // Distance control
    float slowdown_m    = 0.10f;   // Start ramp-down 10 cm before target
    float stop_thresh_m = 0.02f;   // Stop within 2 cm of target
    float min_speed_mps = 0.03f;   // Minimum speed during ramp-down

    // Speed PID (keeps both wheels at same real speed via right encoder)
    float speed_kp      = 0.4f;    // Proportional gain
    float speed_ki      = 0.08f;   // Integral gain
    float speed_max_adj = 0.02f;   // Max speed adjustment per loop [m/s]

    // Timing
    int   loop_ms       = 20;      // Control loop period [ms]
    float cmd_duration  = 0.3f;    // Drive cmd duration field [s]

    // Safety
    float enc_check_s   = 3.0f;    // Fail if no movement after this many seconds
    float enc_check_m   = 0.005f;  // Minimum movement to pass encoder check [m]
    float arm_wait_ms   = 500.0f;  // Wait after arm before driving [ms]
};


// ============================================================
//  Result codes
// ============================================================
enum class StraightResult { SUCCESS, TIMEOUT, ENC_FAIL, ABORTED };
inline const char* straightResultName(StraightResult r) {
    switch (r) {
        case StraightResult::SUCCESS:  return "SUCCESS";
        case StraightResult::TIMEOUT:  return "TIMEOUT";
        case StraightResult::ENC_FAIL: return "ENC_FAIL";
        case StraightResult::ABORTED:  return "ABORTED";
        default:                       return "UNKNOWN";
    }
}


// ============================================================
//  StraightDriver — single-encoder closed-loop for straight moves
// ============================================================
class StraightDriver {
public:
    using Config = StraightDriveConfig;

    StraightDriver(EncSerialReader& enc,
                   DriveSerial&     drive,
                   Config           cfg = Config{})
        : enc_(enc), drive_(drive), cfg_(cfg), abort_(false) {}

    // ── driveStraight ──────────────────────────────────────────
    //  targetM    : distance [m] (positive always)
    //  forward    : true = forward, false = reverse
    //  speedMps   : target speed [m/s]
    //  timeoutS   : safety timeout [s]
    //
    StraightResult driveStraight(float targetM,
                                  bool  forward,
                                  float speedMps,
                                  float timeoutS = 15.0f)
    {
        abort_      = false;
        lastDistM_  = 0.0f;
        lastRightM_ = 0.0f;
        speedIntegral_ = 0.0f;

        if (!enc_.isValid()) {
            printf("[CL] No encoder data\n");
            return StraightResult::ENC_FAIL;
        }

        // Arm + wait
        drive_.arm();
        std::this_thread::sleep_for(
            std::chrono::milliseconds((int)cfg_.arm_wait_ms));

        // Record start position (right encoder only)
        EncSnapshot start = enc_.getSnapshot();
        float startR = start.rightDistM;

        int   gear    = forward ? 1 : -1;
        auto  tStart  = nowMs();
        auto  tTimeout= static_cast<uint64_t>(timeoutS * 1000.0f);
        bool  encOk   = false;

        printf("[CL] Straight %s  target=%.3fm  speed=%.3fm/s\n",
               forward ? "FORWARD" : "REVERSE", targetM, speedMps);

        StraightResult result = StraightResult::TIMEOUT;
        float currentSpeed = speedMps;

        while (!abort_) {
            auto loopStart = nowMs();

            // ── Distance (right encoder only) ─────────────────
            EncSnapshot s = enc_.getSnapshot();
            float dist      = std::fabs(s.rightDistM - startR);
            float remaining = targetM - dist;
            lastDistM_  = dist;
            lastRightM_ = s.rightDistM;

            // ── Encoder check ──────────────────────────────────
            float elapsedS = (nowMs() - tStart) / 1000.0f;
            if (!encOk && elapsedS > cfg_.enc_check_s) {
                if (dist < cfg_.enc_check_m) {
                    printf("[CL] ENC_FAIL  no movement after %.1fs "
                           "(dist=%.5fm)\n", elapsedS, dist);
                    result = StraightResult::ENC_FAIL;
                    break;
                }
                encOk = true;
            }

            // ── Target reached ─────────────────────────────────
            if (dist > 0.001f && remaining <= cfg_.stop_thresh_m) {
                printf("[CL] SUCCESS  dist=%.4fm  remaining=%.4fm\n",
                       dist, remaining);
                result = StraightResult::SUCCESS;
                break;
            }

            // ── Timeout ────────────────────────────────────────
            if (nowMs() - tStart > tTimeout) {
                printf("[CL] TIMEOUT  dist=%.4fm  remaining=%.4fm\n",
                       dist, remaining);
                result = StraightResult::TIMEOUT;
                break;
            }

            // ── Speed ramp-down ────────────────────────────────
            float targetSpeed = speedMps;
            if (remaining < cfg_.slowdown_m && remaining > 0.0f) {
                float ratio = remaining / cfg_.slowdown_m;
                targetSpeed = std::max(cfg_.min_speed_mps,
                    cfg_.min_speed_mps + ratio*(speedMps - cfg_.min_speed_mps));
            }

            // ── Speed PID (right encoder feedback) ────────────
            // Measures actual wheel RPM and adjusts commanded speed
            // so both wheels maintain the same real-world speed.
            float actualMs   = s.rightSpeedMs;   // m/s from right encoder
            float speedErr   = targetSpeed - actualMs;
            speedIntegral_  += speedErr * (cfg_.loop_ms / 1000.0f);
            speedIntegral_   = std::max(-0.05f,
                               std::min( 0.05f, speedIntegral_));

            float adj = cfg_.speed_kp * speedErr
                      + cfg_.speed_ki * speedIntegral_;
            adj = std::max(-cfg_.speed_max_adj,
                  std::min( cfg_.speed_max_adj, adj));
            currentSpeed = std::max(cfg_.min_speed_mps,
                           std::min(speedMps, targetSpeed + adj));

            // ── Send drive — steer_deg=0 → equal wheel PWM ────
            drive_.driveStraight(currentSpeed, gear, cfg_.cmd_duration);

            // ── Debug every 200 ms ─────────────────────────────
            static uint64_t dbg = 0;
            if (nowMs() - dbg > 200) {
                dbg = nowMs();
                printf("[CL] dist=%.4f  rem=%.4f  "
                       "cmd=%.3f  actual=%.3f  rRPM=%.2f\n",
                       dist, remaining,
                       currentSpeed, actualMs, s.rightWRpm);
            }

            // ── Wait remainder of loop ─────────────────────────
            uint64_t elapsed = nowMs() - loopStart;
            if (elapsed < (uint64_t)cfg_.loop_ms)
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(cfg_.loop_ms - elapsed));
        }

        if (abort_) result = StraightResult::ABORTED;
        drive_.stop("distance_reached");
        return result;
    }

    void abort() { abort_ = true; }

    float lastDist()  const { return lastDistM_; }

private:
    static uint64_t nowMs() {
        using namespace std::chrono;
        return static_cast<uint64_t>(
            duration_cast<milliseconds>(
                steady_clock::now().time_since_epoch()).count());
    }

    EncSerialReader&  enc_;
    DriveSerial&      drive_;
    Config            cfg_;
    std::atomic<bool> abort_;
    float             lastDistM_   = 0.0f;
    float             lastRightM_  = 0.0f;
    float             speedIntegral_ = 0.0f;
};
