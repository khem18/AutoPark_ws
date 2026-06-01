// ============================================================
//  encoder_bridge.cpp  —  ROS2 encoder closed-loop bridge  v3
//
//  FIXES vs v2:
//  1. steer_active_hold check handles both "key":true and "key": true
//  2. Logs ALL received drive commands (not just matched ones)
//     so you can see why a command was not intercepted
//  3. Relaxed straight condition — steer_active_hold no longer
//     required (some autopark_master versions omit it)
//  4. speed_scale corrects target when autopark scales speed
//     but not duration:
//       target = (speed_mps / speed_scale) × duration
//  5. dist_m priority — uses exact planner value if present
//  6. No concurrent serial writes — only encoder drive commands
//     go to motor; arm/stop/led handled by serial_bridge only
// ============================================================

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include "enc_serial_reader.hpp"
#include "rdk_closed_loop.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <thread>

// ── JSON helpers (handle both "key":val and "key": val) ───────
static float jsonFloat(const std::string& s, const char* key, float def = 0.0f) {
    std::string k = std::string("\"") + key + "\":";
    auto pos = s.find(k);
    if (pos == std::string::npos) return def;
    // skip optional space after colon
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
    std::string k = std::string("\"") + key + "\":\"";
    auto pos = s.find(k);
    if (pos == std::string::npos) return def;
    auto start = pos + k.size();
    auto end   = s.find('"', start);
    if (end == std::string::npos) return def;
    return s.substr(start, end - start);
}

// Robust bool check — handles both ":true" and ": true"
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

        enc_port_     = get_parameter("enc_port").as_string();
        drive_port_   = get_parameter("drive_port").as_string();
        steer_thresh_ = (float)get_parameter("straight_steer_thresh").as_double();
        speed_scale_  = (float)get_parameter("speed_scale").as_double();

        RCLCPP_INFO(get_logger(),
            "enc=%s  drive=%s  thresh=%.1f°  speed_scale=%.4f",
            enc_port_.c_str(), drive_port_.c_str(),
            steer_thresh_, speed_scale_);

        try {
            drive_ = std::make_unique<DriveSerial>(drive_port_.c_str());
            enc_   = std::make_unique<EncSerialReader>(enc_port_.c_str());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Serial open failed: %s", e.what());
            throw;
        }

        StraightDriveConfig cfg;
        cfg.slowdown_m    = 0.30f;
        cfg.stop_thresh_m = 0.02f;
        cfg.min_speed_mps = 0.02f;
        cfg.speed_kp      = 0.4f;
        cfg.speed_ki      = 0.08f;
        cfg.enc_check_s   = 5.0f;
        cfg.enc_check_m   = 0.005f;
        cfg.arm_wait_ms   = 1500.0f;  // extra wait for steer to reach 0°

        driver_ = std::make_unique<StraightDriver>(*enc_, *drive_, cfg);

        sub_ = create_subscription<std_msgs::msg::String>(
            "/autopark/cmd_json", 10,
            [this](const std_msgs::msg::String::SharedPtr msg) {
                handle_command(msg->data);
            });

        pub_status_ = create_publisher<std_msgs::msg::String>("/enc_status", 10);
        pub_result_ = create_publisher<std_msgs::msg::String>("/enc_result", 10);

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
    void handle_command(const std::string& json) {
        std::string type = jsonStr(json, "type", "");

        // ── Non-drive: abort drive, let serial_bridge handle ───
        if (type != "drive") {
            if (type == "stop" || type == "disarm" || type == "manual") {
                if (drive_thread_.joinable()) {
                    driver_->abort();
                    drive_thread_.join();
                }
                RCLCPP_INFO(get_logger(), "Drive aborted by: %s", type.c_str());
            }
            // arm/led/others → serial_bridge only (no write here)
            return;
        }

        // ── Parse drive fields ─────────────────────────────────
        float speed_mps  = std::fabs(jsonFloat(json, "speed_mps", 0.0f));
        int   gear       = jsonInt  (json, "gear", 0);
        float steer_deg  = jsonFloat(json, "steer_deg", 0.0f);
        float duration   = jsonFloat(json, "duration", 0.0f);
        float dist_m_raw = jsonFloat(json, "dist_m", -1.0f);
        bool  act_hold   = jsonBool (json, "steer_active_hold");

        // Log every drive command so you can diagnose missed intercepts
        RCLCPP_INFO(get_logger(),
            "CMD  gear=%+d speed=%.4f steer=%.1f° dur=%.2f "
            "dist_m=%.4f active_hold=%d",
            gear, speed_mps, steer_deg, duration, dist_m_raw, (int)act_hold);

        // ── Straight move condition ────────────────────────────
        // steer_active_hold is preferred but NOT required —
        // some autopark_master versions omit it for straight moves
        bool steer_ok  = std::fabs(steer_deg) <= steer_thresh_;
        bool speed_ok  = speed_mps > 0.0f;
        bool gear_ok   = gear != 0;
        bool dur_ok    = duration > 0.01f || dist_m_raw > 0.001f;

        if (!steer_ok || !speed_ok || !gear_ok || !dur_ok) {
            RCLCPP_INFO(get_logger(),
                "  → Not straight (steer_ok=%d speed_ok=%d "
                "gear_ok=%d dur_ok=%d) — serial_bridge handles",
                (int)steer_ok, (int)speed_ok, (int)gear_ok, (int)dur_ok);
            // Arc or invalid command — serial_bridge forwards it
            if (drive_thread_.joinable()) {
                driver_->abort();
                drive_thread_.join();
            }
            return;
        }

        // ── Compute target distance ────────────────────────────
        float target_m;

        if (dist_m_raw > 0.001f) {
            // Priority 1: exact planner value
            target_m = dist_m_raw;
            RCLCPP_INFO(get_logger(),
                "  → Straight (dist_m direct): target=%.4fm", target_m);

        } else {
            // Priority 2: speed × duration, corrected for speed_scale
            // autopark_master sends: speed = base × scale, duration = dist / base
            // so: dist = speed/scale × duration = base × duration
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
                "  → Target %.4fm out of range [0.001, 10] m"
                " — check speed_scale=%.4f",
                target_m, speed_scale_);
            return;
        }

        // ── Start encoder-controlled drive ─────────────────────
        if (drive_thread_.joinable()) {
            driver_->abort();
            drive_thread_.join();
        }

        bool  forward = (gear > 0);
        float timeout = std::max(target_m / 0.005f, 30.0f);

        drive_thread_ = std::thread(
            [this, target_m, forward, speed_mps, timeout]() {
            StraightResult result = driver_->driveStraight(
                target_m, forward, speed_mps, timeout);

            RCLCPP_INFO(rclcpp::get_logger("encoder_bridge"),
                "Straight done: %s  dist=%.4fm  target=%.4fm",
                straightResultName(result), driver_->lastDist(), target_m);

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

    std::unique_ptr<DriveSerial>      drive_;
    std::unique_ptr<EncSerialReader>  enc_;
    std::unique_ptr<StraightDriver>   driver_;

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr    pub_status_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr    pub_result_;
    rclcpp::TimerBase::SharedPtr                           timer_;

    std::thread drive_thread_;
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
