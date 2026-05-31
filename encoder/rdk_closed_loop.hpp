#pragma once
// ============================================================
//  rdk_closed_loop.hpp  —  v3
//  Key changes:
//    - arm wait: 150ms → 500ms (prevents move2 direction bug)
//    - Expose lastLeftDist / lastRightDist / lastAvgDist_m1
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
    void drive(float speedMps, int gear, float steerDeg,
               float durationS, bool activeHold) {
        char buf[320];
        snprintf(buf, sizeof(buf),
            "{\"type\":\"drive\","
            "\"speed_mps\":%.4f,\"gear\":%d,"
            "\"steer_deg\":%.2f,\"duration\":%.2f,"
            "\"steer_active_hold\":%s}",
            speedMps, gear, steerDeg, durationS,
            activeHold ? "true" : "false");
        writeLine(buf);
    }
private:
    int fd_ = -1;
};

struct ClosedLoopConfig {
    float slowdown_m    = 0.10f;
    float stop_thresh_m = 0.02f;
    float min_speed_mps = 0.03f;
    float drift_kp      = 40.0f;
    float drift_db_m    = 0.005f;
    float max_trim_deg  = 8.0f;
    int   loop_ms       = 20;
    float cmd_duration  = 0.3f;
    float enc_check_s   = 5.0f;
    float enc_check_m   = 0.00005f; // detect ANY movement
};

enum class CLResult { SUCCESS, TIMEOUT, ENC_FAIL, ABORTED };
inline const char* clResultName(CLResult r) {
    switch (r) {
        case CLResult::SUCCESS:  return "SUCCESS";
        case CLResult::TIMEOUT:  return "TIMEOUT";
        case CLResult::ENC_FAIL: return "ENC_FAIL";
        case CLResult::ABORTED:  return "ABORTED";
        default:                 return "UNKNOWN";
    }
}

class RdkClosedLoop {
public:
    using Config = ClosedLoopConfig;

    RdkClosedLoop(EncSerialReader& enc, DriveSerial& drive, Config cfg = Config{})
        : enc_(enc), drive_(drive), cfg_(cfg), abort_(false) {}

    CLResult driveDistance(float targetM, bool forward,
                           float speedMps, float steerDeg = 0.0f,
                           float timeoutS = 15.0f)
    {
        abort_      = false;
        lastLDist_  = 0.0f;
        lastRDist_  = 0.0f;

        if (!enc_.isValid()) {
            printf("[CL] No encoder data\n");
            return CLResult::ENC_FAIL;
        }

        // ── Arm motor ESP32 ─────────────────────────────────
        // 500 ms gives the motor ESP32 enough time to:
        //   1. receive arm JSON
        //   2. process it (set autoLatched=true)
        //   3. enter MODE_AUTO
        // 150 ms was too fast → move 2 received stale gear=+1 commands
        drive_.arm();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // Record start distances
        EncSnapshot start = enc_.getSnapshot();
        float startL = start.leftDistM;
        float startR = start.rightDistM;

        int   gear     = forward ? 1 : -1;
        auto  tStart   = nowMs();
        auto  tTimeout = static_cast<uint64_t>(timeoutS * 1000.0f);
        bool  encOk    = false;

        printf("[CL] Start  target=%.3fm  gear=%+d  speed=%.3fm/s  steer=%.1f°\n",
               targetM, gear, speedMps, steerDeg);

        CLResult result = CLResult::TIMEOUT;

        while (!abort_) {
            auto loopStart = nowMs();

            EncSnapshot s  = enc_.getSnapshot();
            float dl       = std::fabs(s.leftDistM  - startL);
            float dr       = std::fabs(s.rightDistM - startR);
            float dist     = (dl + dr) * 0.5f;
            float remaining = targetM - dist;

            lastLDist_ = dl;
            lastRDist_ = dr;

            // Encoder check
            float elapsedS = (nowMs() - tStart) / 1000.0f;
            if (!encOk && elapsedS > cfg_.enc_check_s) {
                if (dl < cfg_.enc_check_m && dr < cfg_.enc_check_m) {
                    printf("[CL] ENC_FAIL — no movement after %.1f s"
                           "  (dl=%.6f dr=%.6f threshold=%.6f)\n",
                           elapsedS, dl, dr, cfg_.enc_check_m);
                    result = CLResult::ENC_FAIL; break;
                }
                encOk = true;
            }

            // Reached target
            if (dist > 0.001f && remaining <= cfg_.stop_thresh_m) {
                printf("[CL] SUCCESS  dist=%.4fm  remaining=%.4fm\n", dist, remaining);
                result = CLResult::SUCCESS; break;
            }

            // Timeout
            if (nowMs() - tStart > tTimeout) {
                printf("[CL] TIMEOUT  dist=%.4fm  remaining=%.4fm\n", dist, remaining);
                result = CLResult::TIMEOUT; break;
            }

            // Speed ramp-down
            float speed = speedMps;
            if (remaining < cfg_.slowdown_m && remaining > 0.0f) {
                float ratio = remaining / cfg_.slowdown_m;
                speed = std::max(cfg_.min_speed_mps,
                                 cfg_.min_speed_mps + ratio*(speedMps-cfg_.min_speed_mps));
            }

            // Drift correction
            float steerCmd = steerDeg;
            float drift = dl - dr;
            if (std::fabs(drift) > cfg_.drift_db_m) {
                float trim = std::max(-cfg_.max_trim_deg,
                             std::min( cfg_.max_trim_deg, -drift*cfg_.drift_kp));
                steerCmd += trim;
            }

            // Send drive
            drive_.drive(speed, gear, steerCmd, cfg_.cmd_duration, true);

            // Debug every 200 ms
            static uint64_t dbg = 0;
            if (nowMs() - dbg > 200) {
                dbg = nowMs();
                printf("[CL] dist=%.4f  rem=%.4f  speed=%.3f  "
                       "drift=%.5f  steer=%.1f  pkts=%u\n",
                       dist, remaining, speed, drift, steerCmd, enc_.packetCount());
            }

            uint64_t elapsed = nowMs() - loopStart;
            if (elapsed < (uint64_t)cfg_.loop_ms)
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(cfg_.loop_ms - elapsed));
        }

        if (abort_) result = CLResult::ABORTED;
        drive_.stop("distance_reached");
        lastAvgM1_ = (lastLDist_ + lastRDist_) * 0.5f;
        return result;
    }

    void abort() { abort_ = true; }

    float lastLeftDist()  const { return lastLDist_; }
    float lastRightDist() const { return lastRDist_; }
    float lastAvgDist_m1() const { return lastAvgM1_; }

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
    float lastLDist_  = 0.0f;
    float lastRDist_  = 0.0f;
    float lastAvgM1_  = 0.0f;
};
