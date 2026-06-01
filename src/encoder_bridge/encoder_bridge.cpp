// ============================================================
//  encoder_bridge.cpp  —  ROS2 node for encoder closed-loop
//
//  Subscribes to /autopark/cmd_json (from autopark_master).
//  For STRAIGHT moves (steer_deg ≈ 0): uses encoder to stop
//    at the correct distance instead of time-based control.
//    Distance is computed from: dist = speed_mps × duration
//  For ARC moves (steer_deg ≠ 0): passes through unchanged.
//  For ARM / STOP / LED: passes through unchanged.
//
//  Publishes encoder status to /enc_status.
//
//  Works for ANY starting pose because autopark_master already
//  computes the correct duration/distance from the camera pose.
//  This node just ensures the car drives EXACTLY that distance.
//
//  Ports (set as ROS2 parameters):
//    enc_port   = /dev/ttyUSB0   (encoder ESP32)
//    drive_port = /dev/ttyUSB2   (motor ESP32)
//    straight_steer_thresh = 5.0  (deg, below = straight move)
// ============================================================

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include "enc_serial_reader.hpp"
#include "rdk_closed_loop.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <thread>

// ── Simple JSON field helpers (no ArduinoJSON on RDK) ─────────
static float jsonFloat(const std::string& s, const char* key, float def = 0.0f) {
    std::string k = std::string("\"") + key + "\":";
    auto pos = s.find(k);
    if (pos == std::string::npos) return def;
    try { return std::stof(s.substr(pos + k.size())); } catch (...) { return def; }
}

static int jsonInt(const std::string& s, const char* key, int def = 0) {
    std::string k = std::string("\"") + key + "\":";
    auto pos = s.find(k);
    if (pos == std::string::npos) return def;
    try { return std::stoi(s.substr(pos + k.size())); } catch (...) { return def; }
}

static std::string jsonStr(const std::string& s, const char* key, const std::string& def = "") {
    std::string k = std::string("\"") + key + "\":\"";
    auto pos = s.find(k);
    if (pos == std::string::npos) return def;
    auto start = pos + k.size();
    auto end   = s.find('"', start);
    if (end == std::string::npos) return def;
    return s.substr(start, end - start);
}


// ── EncoderBridgeNode ─────────────────────────────────────────
class EncoderBridgeNode : public rclcpp::Node {
public:
    EncoderBridgeNode()
        : Node("encoder_bridge"),
          drive_(nullptr), enc_(nullptr), driver_(nullptr)
    {
        // Parameters
        declare_parameter("enc_port",   "/dev/ttyUSB0");
        declare_parameter("drive_port", "/dev/ttyUSB2");
        declare_parameter("straight_steer_thresh", 5.0);

        enc_port_   = get_parameter("enc_port").as_string();
        drive_port_ = get_parameter("drive_port").as_string();
        steer_thresh_ = (float)get_parameter("straight_steer_thresh").as_double();

        RCLCPP_INFO(get_logger(), "enc_port=%s  drive_port=%s  thresh=%.1f°",
                    enc_port_.c_str(), drive_port_.c_str(), steer_thresh_);

        // Open serial ports
        try {
            drive_ = std::make_unique<DriveSerial>(drive_port_.c_str());
            enc_   = std::make_unique<EncSerialReader>(enc_port_.c_str());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Serial open failed: %s", e.what());
            throw;
        }

        // Closed-loop config (uses calibrated WHEEL_CIRCUM=1.041 in enc_serial_reader)
        StraightDriveConfig cfg;
        cfg.slowdown_m    = 0.30f;
        cfg.stop_thresh_m = 0.02f;
        cfg.min_speed_mps = 0.02f;
        cfg.speed_kp      = 0.4f;
        cfg.speed_ki      = 0.08f;
        cfg.enc_check_s   = 3.0f;
        cfg.enc_check_m   = 0.005f;
        cfg.arm_wait_ms   = 1000.0f;

        driver_ = std::make_unique<StraightDriver>(*enc_, *drive_, cfg);

        // Subscribe to autopark commands
        sub_ = create_subscription<std_msgs::msg::String>(
            "/autopark/cmd_json", 10,
            [this](const std_msgs::msg::String::SharedPtr msg) {
                handle_command(msg->data);
            });

        // Publish encoder status
        pub_ = create_publisher<std_msgs::msg::String>("/enc_status", 10);

        // Status publisher timer (10 Hz)
        timer_ = create_wall_timer(
            std::chrono::milliseconds(100),
            [this]() { publish_status(); });

        // Reset encoder on start
        {
            int fd = ::open(enc_port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
            if (fd >= 0) { ::write(fd, "r\n", 2); ::close(fd); }
        }

        RCLCPP_INFO(get_logger(), "Encoder bridge ready");
    }

private:
    // ── Command handler ─────────────────────────────────────────
    void handle_command(const std::string& json) {
        std::string type = jsonStr(json, "type", "");

        // ── Passthrough: arm, stop, led, manual ────────────────
        if (type != "drive") {
            if (type == "arm" || type == "disarm" || type == "stop"
                || type == "led" || type == "manual") {
                // Abort any running drive first
                if (drive_thread_.joinable()) {
                    driver_->abort();
                    drive_thread_.join();
                }
                drive_->writeLine(json.c_str());
                RCLCPP_INFO(get_logger(), "Passthrough: %s", type.c_str());
            }
            return;
        }

        // ── Drive command ─────────────────────────────────────
        float speed_mps   = std::fabs(jsonFloat(json, "speed_mps", 0.0f));
        int   gear        = jsonInt  (json, "gear",  0);
        float steer_deg   = jsonFloat(json, "steer_deg", 0.0f);
        float duration    = jsonFloat(json, "duration", 0.0f);
        bool  active_hold = (json.find("\"steer_active_hold\":true") != std::string::npos);

        // ── Straight move → encoder closed-loop ───────────────
        if (std::fabs(steer_deg) <= steer_thresh_
            && active_hold && gear != 0 && speed_mps > 0.0f
            && duration > 0.01f)
        {
            // Compute target distance from speed × duration
            float target_m = speed_mps * duration;

            RCLCPP_INFO(get_logger(),
                "Straight: gear=%+d speed=%.3f dur=%.2f → target=%.3fm",
                gear, speed_mps, duration, target_m);

            // Abort previous if running
            if (drive_thread_.joinable()) {
                driver_->abort();
                drive_thread_.join();
            }

            bool forward = (gear > 0);
            float timeout = duration * 3.0f + 5.0f;  // generous safety timeout

            // Run in background thread so we don't block ROS2 spin
            drive_thread_ = std::thread([this, target_m, forward,
                                         speed_mps, timeout]() {
                StraightResult result = driver_->driveStraight(
                    target_m, forward, speed_mps, timeout);

                RCLCPP_INFO(rclcpp::get_logger("encoder_bridge"),
                    "Straight done: %s  dist=%.4fm",
                    straightResultName(result), driver_->lastDist());

                // Publish result
                auto msg = std_msgs::msg::String();
                char buf[256];
                snprintf(buf, sizeof(buf),
                    "{\"enc_result\":\"%s\",\"dist\":%.4f}",
                    straightResultName(result), driver_->lastDist());
                msg.data = buf;
                // Note: pub_ access from thread is safe for rclcpp
            });
            return;
        }

        // ── Arc move or non-straight → passthrough ─────────────
        RCLCPP_INFO(get_logger(),
            "Passthrough arc: steer=%.1f° gear=%+d speed=%.3f",
            steer_deg, gear, speed_mps);

        // Abort any running straight drive
        if (drive_thread_.joinable()) {
            driver_->abort();
            drive_thread_.join();
        }
        drive_->writeLine(json.c_str());
    }

    // ── Status publisher ────────────────────────────────────────
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
        pub_->publish(msg);
    }

    // ── Members ─────────────────────────────────────────────────
    std::string enc_port_, drive_port_;
    float       steer_thresh_;

    std::unique_ptr<DriveSerial>      drive_;
    std::unique_ptr<EncSerialReader>  enc_;
    std::unique_ptr<StraightDriver>   driver_;

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr    pub_;
    rclcpp::TimerBase::SharedPtr                           timer_;

    std::thread drive_thread_;
};


// ── main ──────────────────────────────────────────────────────
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<EncoderBridgeNode>());
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("encoder_bridge"),
                     "Fatal: %s", e.what());
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
