#pragma once
// ============================================================
//  rdk_closed_loop.hpp  —  v6  Dual-Encoder Straight Drive
//
//  CHANGES vs v5:
//  [v6] Dual-encoder distance tracking.
//       Uses AVERAGE of left + right encoder distances for target tracking
//       instead of right-only.  Accurate even when wheels turn at different
//       speeds (motor imbalance).  Speed PID still uses rightSpeedMs.
//       A lr_delta diagnostic is printed every 200 ms so imbalance magnitude
//       can be read from logs and fed into left_pwm_boost tuning in firmware.
//
//  [v5] Stuck-detection: boost speedMps instead of ENC_FAIL under heavy load.
//       speedMps is passed by REFERENCE → session variable updates in-place.
//
//  Left/right motor imbalance note:
//       The primary fix is left_pwm_boost in esp32_drive_controller.ino.
//       Dual-encoder averaging here ensures distance measurement is accurate
//       regardless of wheel balance.
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
    float enc_check_s   = 3.0f;    // (kept for reference — now used as first stuck_check_s)
    float enc_check_m   = 0.005f;  // (kept for reference — now used as stuck_min_move_m)
    float arm_wait_ms   = 500.0f;  // Wait after arm before driving [ms]

    // ── [v5] Stuck detection / session speed boost ────────────────────────
    // When the car carries passengers and the motor stalls at the calibrated
    // speed, driveStraight() detects "stuck" (encoder distance doesn't increase)
    // and adds stuck_boost_mps to the speed instead of returning ENC_FAIL.
    // speedMps is passed by reference, so the caller's session variable updates
    // in-place and all later moves in the same round use the higher speed.
    // If still stuck, the boost repeats every stuck_check_s until the cap is hit.
    float stuck_boost_mps     = 0.010f;  // Speed added per stuck event [m/s]
    float stuck_max_speed_mps = 0.150f;  // Hard cap on session speed [m/s]
    float stuck_check_s       = 3.0f;    // Seconds between stuck checks [s]
    float stuck_min_move_m    = 0.005f;  // Min encoder distance per check to NOT be stuck [m]
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

    // ── Live steer update ──────────────────────────────────────
    // Called while driveStraight() is running (from another thread).
    // autopark_master sends a "steer_update" cmd when IMU detects drift
    // during Move3 — encoder_bridge relays it here so the drive loop
    // sends the corrected angle to ESP32 on the next 20 ms tick.
    void setLiveSteer(float deg) { live_steer_.store(deg); }
    float getLiveSteer()   const { return live_steer_.load(); }

    // ── driveStraight ──────────────────────────────────────────
    //  targetM    : distance [m] (positive always)
    //  forward    : true = forward, false = reverse
    //  speedMps   : [v5] REFERENCE to caller's session speed variable [m/s].
    //               When the car is stuck, this value is INCREASED by
    //               cfg_.stuck_boost_mps so the caller's session variable
    //               reflects the boosted speed for subsequent moves.
    //  timeoutS   : safety timeout [s]
    //  steerDeg   : steer angle to hold throughout the drive [degrees]
    //  skipArm    : skip the arm() call when the motor is already active
    //
    StraightResult driveStraight(float  targetM,
                                  bool   forward,
                                  float& speedMps,   // [v5] by reference — updates session speed
                                  float  timeoutS = 15.0f,
                                  float  steerDeg = 0.0f,
                                  bool   skipArm  = false)
    {
        abort_         = false;
        lastDistM_     = 0.0f;
        lastRightM_    = 0.0f;
        speedIntegral_ = 0.0f;
        live_steer_.store(steerDeg);   // initialise; may be updated live by setLiveSteer()

        if (!enc_.isValid()) {
            printf("[CL] No encoder data\n");
            return StraightResult::ENC_FAIL;
        }

        // Arm + wait — SKIP when motor is already active (settle CMD still running).
        if (!skipArm) {
            drive_.arm();
            std::this_thread::sleep_for(
                std::chrono::milliseconds((int)cfg_.arm_wait_ms));
        }

        // [v6] Record start position for BOTH encoders.
        // Average left+right distance is used for target tracking so that
        // a motor imbalance (left slower than right) does not cause the car
        // to stop short or overshoot depending on which side runs faster.
        EncSnapshot start = enc_.getSnapshot();
        float startR = start.rightDistM;
        float startL = start.leftDistM;   // [v6] dual-encoder start

        int   gear     = forward ? 1 : -1;
        auto  tStart   = nowMs();
        auto  tTimeout = static_cast<uint64_t>(timeoutS * 1000.0f);

        // ── [v5] Stuck detection state ─────────────────────────────────────
        // Check every stuck_check_s.  First check fires after stuck_check_s.
        // If encoder hasn't moved stuck_min_move_m since last check → boost.
        uint64_t stuckNextCheckMs = tStart + static_cast<uint64_t>(cfg_.stuck_check_s * 1000.0f);
        float    stuckRefDist     = 0.0f;   // dist at last check (start = 0)
        int      stuckBoostCount  = 0;      // how many boosts applied this drive

        printf("[CL] Straight %s  target=%.3fm  speed=%.3fm/s\n",
               forward ? "FORWARD" : "REVERSE", targetM, speedMps);

        StraightResult result = StraightResult::TIMEOUT;
        float currentSpeed = speedMps;

        while (!abort_) {
            auto loopStart = nowMs();

            EncSnapshot s = enc_.getSnapshot();   // [v6] read snapshot once per tick

            // [v6] Dual-encoder distance: average left + right.
            // If the left encoder is not yet valid (lc==0), fall back to right-only
            // so the system still works if one encoder is missing or noisy.
            float distR = std::fabs(s.rightDistM - startR);
            float distL = std::fabs(s.leftDistM  - startL);
            bool  leftValid = (std::fabs(s.leftDistM) > 0.0001f ||
                               std::fabs(startL)      > 0.0001f ||
                               s.leftCount != 0);
            float dist      = leftValid ? (distR + distL) / 2.0f : distR;
            float remaining = targetM - dist;
            lastDistM_  = dist;
            lastRightM_ = s.rightDistM;

            // ── [v5] Stuck detection — replaces ENC_FAIL ───────
            // Every stuck_check_s seconds: if encoder hasn't moved
            // stuck_min_move_m → boost speedMps (session speed ref) and continue.
            // Repeats until stuck_max_speed_mps cap is reached.
            if (nowMs() >= stuckNextCheckMs) {
                float moved = dist - stuckRefDist;
                if (moved < cfg_.stuck_min_move_m) {
                    if (speedMps < cfg_.stuck_max_speed_mps) {
                        float oldSpeed = speedMps;
                        speedMps = std::min(speedMps + cfg_.stuck_boost_mps,
                                            cfg_.stuck_max_speed_mps);
                        currentSpeed   = speedMps;
                        speedIntegral_ = 0.0f;   // reset PID integral on boost
                        stuckBoostCount++;
                        printf("[CL] STUCK #%d (moved=%.5fm in %.1fs)"
                               " → boost %.4f→%.4f m/s (session updated)\n",
                               stuckBoostCount, moved,
                               cfg_.stuck_check_s, oldSpeed, speedMps);
                    } else {
                        // Already at max speed — still not moving; will TIMEOUT
                        printf("[CL] STUCK at max speed %.4f m/s"
                               " (moved=%.5fm) — will timeout\n",
                               speedMps, moved);
                    }
                }
                stuckRefDist      = dist;
                stuckNextCheckMs += static_cast<uint64_t>(cfg_.stuck_check_s * 1000.0f);
            }

            // ── Target reached ─────────────────────────────────
            if (dist > 0.001f && remaining <= cfg_.stop_thresh_m) {
                printf("[CL] SUCCESS  dist=%.4fm  remaining=%.4fm"
                       "  boosts=%d  final_speed=%.4f\n",
                       dist, remaining, stuckBoostCount, speedMps);
                result = StraightResult::SUCCESS;
                break;
            }

            // ── Timeout ────────────────────────────────────────
            if (nowMs() - tStart > tTimeout) {
                printf("[CL] TIMEOUT  dist=%.4fm  remaining=%.4fm"
                       "  boosts=%d  final_speed=%.4f\n",
                       dist, remaining, stuckBoostCount, speedMps);
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
            float actualMs   = s.rightSpeedMs;
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

            // ── Send drive — hold commanded steer angle each tick ─
            // ── Send drive — hold live steer angle each tick ──────────────
            // live_steer_ starts at steerDeg but can be updated mid-drive by
            // autopark_master via "steer_update" → encoder_bridge.setLiveSteer()
            // → corrects IMU drift while encoder controls distance.
            drive_.driveWithSteer(currentSpeed, gear, live_steer_.load(), cfg_.cmd_duration);

            // ── Debug every 200 ms ─────────────────────────────
            static uint64_t dbg = 0;
            if (nowMs() - dbg > 200) {
                dbg = nowMs();
                // [v6] lr_delta: speed difference between wheels.
                // If lr_delta > 0.010 m/s consistently, the left motor is
                // significantly slower and left_pwm_boost in the ESP32
                // firmware should be increased.
                float lr_delta = s.rightSpeedMs - s.leftSpeedMs;
                printf("[CL] dist=%.4f  rem=%.4f  "
                       "cmd=%.3f  actual=%.3f  rRPM=%.2f  "
                       "lr_delta=%+.3f m/s  leftValid=%d\n",
                       dist, remaining,
                       currentSpeed, s.rightSpeedMs, s.rightWRpm,
                       lr_delta, (int)leftValid);
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
    std::atomic<bool>  abort_;
    std::atomic<float> live_steer_{0.0f};  // updated by setLiveSteer() for IMU correction
    float             lastDistM_   = 0.0f;
    float             lastRightM_  = 0.0f;
    float             speedIntegral_ = 0.0f;
};
