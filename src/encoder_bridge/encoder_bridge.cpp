// ============================================================
//  encoder_bridge.cpp  —  ROS2 encoder closed-loop bridge  v4
//
//  FIXES vs v4:
//  1. Reads "target_dist_m" (autopark_master field) in addition
//     to "dist_m", so the planner distance is always used.
//  2. Uses ROS2 params enc_fwd_speed_mps / enc_rev_speed_mps
//     (calibrated from closed_loop_demo) instead of the
//     autopark speed_scale value (which is too slow).
//  3. arm_wait_ms reduced 1500→300 ms: steer is already settled
//     before the drive command arrives (autopark_master waits
//     steer_settle_pause_s before sending drive cmd).  The old
//     1500 ms caused the car to coast ~1.5s under serial_bridge's
//     command before encoder-loop kicked in — untracked distance.
//  4. Publishes /enc_busy (Bool) so serial_bridge can skip
//     forwarding straight commands while encoder is driving.
//  [v5] Fix jsonStr: Python json.dumps uses "key": "val" (space);
//       v4 searched "key":"val" → type always "" → never intercepted.
//  [v5] Guard: if encoder isValid()=False, don't intercept (fallback
//       to serial_bridge time-based stop until encoder is ready).
// ============================================================

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>

#include "enc_serial_reader.hpp"
#include "rdk_closed_loop.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <atomic>

// ── JSON helpers ─────────────────────────────────────────────
static float jsonFloat(const std::string& s, const char* key, float def = 0.0f) {
    std::string k = std::string("\"") + key + "\":";
    auto pos = s.find(k);
    if (pos == std::string::npos) return def;
    size_t vp = pos + k.size();
    while (vp < s.size() && s[vp] == ' ') vp++;
    try { return std::stof(s.substr(vp)); } catch (...) { return def; }
}

static int jsonInt(const std::string& s, const char* key, int def = 0) {
    std::string k = std::string("\"") + key + "\":";
    auto pos = s.find(k);
    if (pos == std::string::npos) return def;
    size_t vp = pos + k.size();
    while (vp < s.size() && s[vp] == ' ') vp++;
    try { return std::stoi(s.substr(vp)); } catch (...) { return def; }
}

static std::string jsonStr(const std::string& s, const char* key,
                            const std::string& def = "") {
    // FIX v5: skip optional space after colon — Python json.dumps produces
    // "key": "value" (with space) but v4 searched for "key":"value" (no space).
    // This caused type="" for ALL commands → encoder_bridge returned immediately.
    std::string k = std::string("\"") + key + "\":";
    auto pos = s.find(k);
    if (pos == std::string::npos) return def;
    size_t vp = pos + k.size();
    while (vp < s.size() && s[vp] == ' ') vp++;   // skip optional space
    if (vp >= s.size() || s[vp] != '"') return def; // must be opening quote
    auto start = vp + 1;
    auto end   = s.find('"', start);
    if (end == std::string::npos) return def;
    return s.substr(start, end - start);
}

static bool jsonBool(const std::string& s, const char* key) {
    std::string k = std::string("\"") + key + "\":";
    auto pos = s.find(k);
    if (pos == std::string::npos) return false;
    size_t vp = pos + k.size();
    while (vp < s.size() && s[vp] == ' ') vp++;
    return s.substr(vp, 4) == "true";
}


// ── EncoderBridgeNode ─────────────────────────────────────────
class EncoderBridgeNode : public rclcpp::Node {
public:
    EncoderBridgeNode()
        : Node("encoder_bridge"),
          drive_(nullptr), enc_(nullptr), driver_(nullptr)
    {
        declare_parameter("enc_port",              "/dev/ttyUSB1");
        declare_parameter("drive_port",            "/dev/ttyUSB2");
        declare_parameter("straight_steer_thresh", 5.0);
        declare_parameter("speed_scale",           0.01);
        // ── v4: calibrated driving speeds (from closed_loop_demo.cpp)
        // Move 1 fwd_setup  → 0.06 m/s  (calibrated: real dist = 0.50m ✓)
        // Move 3 rev_d4     → 0.04 m/s  (calibrated: real dist = 0.30m ✓)
        declare_parameter("enc_fwd_speed_mps",     0.06);
        declare_parameter("enc_rev_speed_mps",     0.04);

        enc_port_         = get_parameter("enc_port").as_string();
        drive_port_       = get_parameter("drive_port").as_string();
        steer_thresh_     = (float)get_parameter("straight_steer_thresh").as_double();
        speed_scale_      = (float)get_parameter("speed_scale").as_double();
        enc_fwd_speed_    = (float)get_parameter("enc_fwd_speed_mps").as_double();
        enc_rev_speed_    = (float)get_parameter("enc_rev_speed_mps").as_double();

        RCLCPP_INFO(get_logger(),
            "enc=%s  drive=%s  thresh=%.1f°  speed_scale=%.4f"
            "  fwd_spd=%.3f  rev_spd=%.3f",
            enc_port_.c_str(), drive_port_.c_str(),
            steer_thresh_, speed_scale_,
            enc_fwd_speed_, enc_rev_speed_);

        try {
            drive_ = std::make_unique<DriveSerial>(drive_port_.c_str());
            enc_   = std::make_unique<EncSerialReader>(enc_port_.c_str());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Serial open failed: %s", e.what());
            throw;
        }

        StraightDriveConfig cfg;
        cfg.slowdown_m    = 0.15f;   // start ramp-down 15 cm before target
        cfg.stop_thresh_m = 0.02f;
        cfg.min_speed_mps = 0.025f;
        cfg.speed_kp      = 0.4f;
        cfg.speed_ki      = 0.08f;
        cfg.enc_check_s   = 4.0f;
        cfg.enc_check_m   = 0.005f;
        // v4: 300 ms — steer is already at target before drive cmd arrives.
        // The old 1500 ms caused ~1.5 s of untracked motion from serial_bridge.
        cfg.arm_wait_ms   = 300.0f;

        driver_ = std::make_unique<StraightDriver>(*enc_, *drive_, cfg);

        sub_ = create_subscription<std_msgs::msg::String>(
            "/autopark/cmd_json", 10,
            [this](const std_msgs::msg::String::SharedPtr msg) {
                handle_command(msg->data);
            });

        pub_status_ = create_publisher<std_msgs::msg::String>("/enc_status", 10);
        pub_result_ = create_publisher<std_msgs::msg::String>("/enc_result", 10);
        // v4: publish busy flag so serial_bridge can skip forwarding straight cmds
        pub_busy_   = create_publisher<std_msgs::msg::Bool>("/enc_busy", 10);

        // Subscribe to steer_ready: used by monitor thread to confirm
        // steer servo has physically settled AFTER the drive CMD arrives.
        create_subscription<std_msgs::msg::Bool>(
            "/autopark/esp32_steer_ready", 20,
            [this](const std_msgs::msg::Bool::SharedPtr msg) {
                if (msg->data) steer_ready_flag_.store(true);
            });

        timer_ = create_wall_timer(
            std::chrono::milliseconds(100),
            [this]() { publish_status(); });

        // Reset encoder boot counts
        {
            int fd = ::open(enc_port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
            if (fd >= 0) { ::write(fd, "r\n", 2); ::close(fd); }
        }

        RCLCPP_INFO(get_logger(), "Encoder bridge ready");
    }

private:
    void set_busy(bool b) {
        auto msg = std_msgs::msg::Bool();
        msg.data = b;
        pub_busy_->publish(msg);
    }

    void handle_command(const std::string& json) {
        std::string type = jsonStr(json, "type", "");

        // ── Non-drive: abort drive, let serial_bridge handle ───
        if (type != "drive") {
            if (type == "stop" || type == "disarm" || type == "manual") {
                if (monitor_thread_.joinable()) {
                    monitor_abort_.store(true);
                    monitor_thread_.join();
                    monitor_abort_.store(false);
                }
                if (drive_thread_.joinable()) {
                    driver_->abort();
                    drive_thread_.join();
                }
                set_busy(false);
                RCLCPP_INFO(get_logger(), "Drive aborted by: %s", type.c_str());
            }
            return;
        }

        // ── Parse drive fields ─────────────────────────────────
        float speed_mps  = std::fabs(jsonFloat(json, "speed_mps", 0.0f));
        int   gear       = jsonInt  (json, "gear", 0);
        float steer_deg  = jsonFloat(json, "steer_deg", 0.0f);
        float duration   = jsonFloat(json, "duration", 0.0f);

        // v4: read both "dist_m" (standalone) and "target_dist_m" (autopark_master)
        float dist_m_raw = jsonFloat(json, "dist_m", -1.0f);
        if (dist_m_raw < 0.001f)
            dist_m_raw = jsonFloat(json, "target_dist_m", -1.0f);

        bool  act_hold   = jsonBool (json, "steer_active_hold");

        RCLCPP_INFO(get_logger(),
            "CMD  gear=%+d speed=%.4f steer=%.1f° dur=%.2f "
            "dist_m=%.4f active_hold=%d",
            gear, speed_mps, steer_deg, duration, dist_m_raw, (int)act_hold);

        // ── Straight move condition ────────────────────────────
        bool speed_ok = speed_mps > 0.0f;
        bool gear_ok  = gear != 0;
        bool dist_ok  = dist_m_raw > 0.001f;   // explicit target distance required
        bool steer_ok = std::fabs(steer_deg) <= steer_thresh_;

        // Intercept rule:
        //   Forward (gear>0): encoder handles ALL moves by distance,
        //     including Move1 curved setup (steer≈-23°). IMU NOT used for Move1.
        //   Reverse (gear<0): only |steer|≤thresh (straight reverses, i.e. Move3).
        //     Move2 arc (steer=+30°) is excluded here → IMU stops it. ✓
        bool intercept = speed_ok && gear_ok && dist_ok &&
                         (gear > 0 || steer_ok);

        if (!intercept) {
            RCLCPP_INFO(get_logger(),
                "  → %s (gear=%+d speed_ok=%d dist_ok=%d steer_ok=%d) — serial_bridge handles",
                (gear < 0 && !steer_ok) ? "Arc reverse — IMU handles"
                                        : "No target dist or zero speed",
                gear, (int)speed_ok, (int)dist_ok, (int)steer_ok);
            // FIX: do NOT abort running drive when rejecting a command.
            // If this is a forward steer-settle CMD (speed=0, gear>0),
            // reset steer_ready_flag HERE so the drive_thread_ (started on
            // the NEXT drive CMD) waits for a fresh signal from this settle.
            if (!speed_ok && gear > 0) {
                steer_ready_flag_.store(false);
                RCLCPP_INFO(get_logger(),
                    "  → Fwd settle CMD: steer_ready_flag reset, awaiting servo");
            }
            return;
        }

        // ── Compute target distance ────────────────────────────
        float target_m;

        if (dist_m_raw > 0.001f) {
            // Priority 1: planner value (target_dist_m or dist_m)
            target_m = dist_m_raw;
            RCLCPP_INFO(get_logger(),
                "  → Straight (dist from cmd): target=%.4fm", target_m);

        } else {
            // Priority 2: speed × duration, corrected for speed_scale
            float base_speed = (speed_scale_ > 0.0001f)
                                ? speed_mps / speed_scale_
                                : speed_mps;
            target_m = base_speed * duration;

            RCLCPP_INFO(get_logger(),
                "  → Straight: speed=%.4f/scale%.4f=%.4f × dur%.2f"
                " = target=%.4fm",
                speed_mps, speed_scale_, base_speed, duration, target_m);
        }

        if (target_m > 10.0f || target_m < 0.001f) {
            RCLCPP_WARN(get_logger(),
                "  → Target %.4fm out of range [0.001, 10] m — check dist_m/target_dist_m",
                target_m);
            return;
        }

        // ── Guard: encoder must have valid data ────────────────
        // If the encoder ESP32 hasn't sent data yet (isValid=False),
        // don't intercept — return and let serial_bridge handle with
        // time-based stop as fallback.  This also logs the pktCount
        // so you can see if data is arriving at all.
        if (!enc_ || !enc_->isValid()) {
            RCLCPP_WARN(get_logger(),
                "  → Encoder NOT valid (pkts=%u) — serial_bridge handles (time-based fallback)",
                enc_ ? enc_->packetCount() : 0u);
            return;
        }

        bool forward = (gear > 0);

        // ── MONITOR MODE: gear=+1 (Move1 forward curved) ─────────────────
        // serial_bridge + ESP32 control speed & steer. encoder_bridge only
        // monitors wheel distance and publishes enc_result when target reached.
        // enc_busy stays FALSE so serial_bridge does NOT filter drive cmds,
        // ensuring ESP32 keeps receiving the correct steer angle (-14.5°).
        if (forward) {
            // ── DRIVE MODE for forward/curved Move1 ──────────────────────
            // Same as Move3 Drive mode:
            //   enc_busy=True → serial_bridge BLOCKS the drive CMD
            //   → ESP32 NEVER receives the drive CMD
            //   → ESP32 steer stays from the settle CMD (no transition!)
            //   → DriveSerial sends speed + CORRECT steer angle
            //
            // Root cause of old Monitor mode failure:
            //   Monitor mode left enc_busy=False → serial_bridge forwarded
            //   the drive CMD to ESP32 → ESP32 transitioned steer → brief
            //   steer deviation at drive start → partial curve in Move1.
            //
            // Fix: block the drive CMD (enc_busy=True) and use driveWithSteer()
            //   which passes the actual steer angle from the motion (-14.5°).
            //   Settle CMD (still active, 12s duration) keeps steer on ESP32.
            //   No transition, no steer deviation. Exactly like Move3. ✓
            if (monitor_thread_.joinable()) {
                monitor_abort_.store(true);
                monitor_thread_.join();
                monitor_abort_.store(false);
            }
            if (drive_thread_.joinable()) {
                driver_->abort();
                drive_thread_.join();
            }

            float timeout    = target_m / 0.025f + 30.0f;
            float drive_speed = enc_fwd_speed_;   // calibrated forward speed (0.06 m/s)

            RCLCPP_INFO(get_logger(),
                "  Safety timeout=%.1fs  (%.3fm / 0.025 m/s + 30s buffer)",
                timeout, target_m);
            RCLCPP_INFO(get_logger(),
                "  → Encoder drive: target=%.4fm  speed=%.3f m/s  steer=%.1f°  gear=+1",
                target_m, drive_speed, steer_deg);

            // ── MONITOR MODE for Move1 (gear=+1, curved) ──────────────────
            // enc_busy stays FALSE → serial_bridge forwards the drive CMD to ESP32
            // using the SAME channel as the settle CMD.
            // ESP32 transitions: speed 0→0.01, SAME steer → no servo re-init,
            // no channel switch, no spring transient. ✓
            // encoder_bridge just monitors encoder distance and publishes enc_result.
            if (monitor_thread_.joinable()) {
                monitor_abort_.store(true);
                monitor_thread_.join();
                monitor_abort_.store(false);
            }
            if (drive_thread_.joinable()) {
                driver_->abort();
                drive_thread_.join();
            }

            float safety_s = target_m / 0.025f + 30.0f;
            RCLCPP_INFO(get_logger(),
                "  → Monitor mode (serial_bridge→ESP32): target=%.4fm  safety=%.1fs",
                target_m, safety_s);

            monitor_thread_ = std::thread(
                [this, target_m, safety_s]() {
                try {
                    for (int i = 0; i < 30 && !enc_->isValid(); i++)
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));

                    float start_dist = enc_->getSnapshot().rightDistM;
                    auto  t_start    = std::chrono::steady_clock::now();

                    while (!monitor_abort_.load()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                        float dist    = std::fabs(enc_->getSnapshot().rightDistM - start_dist);
                        float elapsed = std::chrono::duration<float>(
                            std::chrono::steady_clock::now() - t_start).count();

                        if (dist < target_m && elapsed < safety_s) continue;

                        const char* res = (dist >= target_m) ? "SUCCESS" : "TIMEOUT";
                        RCLCPP_INFO(rclcpp::get_logger("encoder_bridge"),
                            "Monitor done: %s  dist=%.4fm  target=%.4fm  t=%.1fs",
                            res, dist, target_m, elapsed);

                        auto msg = std_msgs::msg::String();
                        char buf[256];
                        snprintf(buf, sizeof(buf),
                            "{\"enc_result\":\"%s\",\"dist\":%.4f,\"target\":%.4f}",
                            res, dist, target_m);
                        msg.data = buf;
                        pub_result_->publish(msg);
                        return;
                    }
                } catch (const std::exception & e) {
                    RCLCPP_ERROR(rclcpp::get_logger("encoder_bridge"),
                        "Monitor thread exception: %s — publishing TIMEOUT", e.what());
                    try {
                        auto msg = std_msgs::msg::String();
                        char buf[256];
                        snprintf(buf, sizeof(buf),
                            "{\"enc_result\":\"ERROR\",\"dist\":0.0,\"target\":%.4f}",
                            target_m);
                        msg.data = buf;
                        pub_result_->publish(msg);
                    } catch (...) {}
                } catch (...) {
                    RCLCPP_ERROR(rclcpp::get_logger("encoder_bridge"),
                        "Monitor thread unknown exception");
                }
            });
            return;  // enc_busy stays FALSE — serial_bridge forwards drive CMD
        }

        // ── DRIVE MODE: gear=-1 (Move3 reverse straight) ─────────────────
        // encoder_bridge controls speed+distance via driveStraight().
        // enc_busy=True blocks serial_bridge from forwarding speed commands.
        // Clean up monitor_thread_ from Move1 before starting drive thread.
        if (monitor_thread_.joinable()) {
            monitor_abort_.store(true);
            monitor_thread_.join();
            monitor_abort_.store(false);
        }
        if (drive_thread_.joinable()) {
            driver_->abort();
            drive_thread_.join();
        }

        float timeout    = target_m / 0.025f + 30.0f;
        float drive_speed = enc_rev_speed_;

        RCLCPP_INFO(get_logger(),
            "  Safety timeout=%.1fs  (%.3fm / 0.025 m/s + 30s buffer)",
            timeout, target_m);
        RCLCPP_INFO(get_logger(),
            "  → Encoder drive: target=%.4fm  speed=%.3f m/s  gear=-1",
            target_m, drive_speed);

        set_busy(true);

        drive_thread_ = std::thread(
            [this, target_m, drive_speed, timeout]() {
            StraightResult result = driver_->driveStraight(
                target_m, false, drive_speed, timeout);

            RCLCPP_INFO(rclcpp::get_logger("encoder_bridge"),
                "Straight done: %s  dist=%.4fm  target=%.4fm",
                straightResultName(result), driver_->lastDist(), target_m);

            set_busy(false);

            auto msg = std_msgs::msg::String();
            char buf[256];
            snprintf(buf, sizeof(buf),
                "{\"enc_result\":\"%s\",\"dist\":%.4f,\"target\":%.4f}",
                straightResultName(result), driver_->lastDist(), target_m);
            msg.data = buf;
            pub_result_->publish(msg);
        });
    }

    void publish_status() {
        if (!enc_ || !enc_->isValid()) return;
        auto snap = enc_->getSnapshot();
        char buf[256];
        snprintf(buf, sizeof(buf),
            "{\"rc\":%ld,\"rd\":%.4f,\"rrpm\":%.2f,\"rspd\":%.3f}",
            snap.rightCount, snap.rightDistM,
            snap.rightWRpm, snap.rightSpeedMs);
        auto msg = std_msgs::msg::String();
        msg.data = buf;
        pub_status_->publish(msg);
    }

    std::string enc_port_, drive_port_;
    float       steer_thresh_, speed_scale_;
    float       enc_fwd_speed_, enc_rev_speed_;  // v4

    std::unique_ptr<DriveSerial>      drive_;
    std::unique_ptr<EncSerialReader>  enc_;
    std::unique_ptr<StraightDriver>   driver_;

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr    pub_status_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr    pub_result_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr      pub_busy_;
    rclcpp::TimerBase::SharedPtr                           timer_;

    std::thread drive_thread_;
    std::thread monitor_thread_;        // forward move distance monitor
    std::atomic<bool> monitor_abort_{false};
    std::atomic<bool> steer_ready_flag_{false};  // set by /esp32_steer_ready
};


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<EncoderBridgeNode>());
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("encoder_bridge"), "Fatal: %s", e.what());
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
