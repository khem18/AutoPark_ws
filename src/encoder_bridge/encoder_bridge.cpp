// ============================================================
//  encoder_bridge.cpp  —  ROS2 encoder closed-loop bridge  v7
//
//  CHANGES vs v6:
//  [v7a] drive_thread_ lambda now has try-catch.
//        Previously, if driveStraight() threw an uncaught C++ exception
//        (e.g. from a serial read/write error on /dev/ttyUSB0), set_busy(false)
//        was never called → enc_busy stayed True → serial_bridge silently
//        blocked ALL subsequent kinetic REVERSE commands (gear < 0).
//        The motor appeared dead during KINETIC-B even though commands were
//        being sent by autopark_master.
//        Fix: wrap the thread body in try/catch; call set_busy(false) and
//        publish ENC_FAIL result on any exception path.
//
//  [v7b] arc_monitor_thread_ already had try-catch and set_busy(false) in
//        all paths — no change needed there.
//
//  Existing behaviour (unchanged):
//  [v6] Arc-move stuck detection (arc_monitor_thread_).
//  [v5] Straight reverse (Move3): session_rev_speed_ boost.
//  [v5] Forward monitor (Move1): stuck → driveStraight() takeover.
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
    std::string k = std::string("\"") + key + "\":";
    auto pos = s.find(k);
    if (pos == std::string::npos) return def;
    size_t vp = pos + k.size();
    while (vp < s.size() && s[vp] == ' ') vp++;
    if (vp >= s.size() || s[vp] != '"') return def;
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

// ── Timestamp helper (ms) ─────────────────────────────────────
static inline uint64_t nowMs_br() {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<milliseconds>(
            steady_clock::now().time_since_epoch()).count());
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
        declare_parameter("enc_fwd_speed_mps",     0.06);
        declare_parameter("enc_rev_speed_mps",     0.04);

        // ── [v5] Stuck detection / session speed boost params ──────────────
        // stuck_speed_enabled: master switch for the entire stuck-boost feature.
        // Set true in the launch file to enable auto-boost for 98 kg passenger load.
        // Default false so an unloaded car is never accidentally boosted.
        declare_parameter("stuck_speed_enabled",  false);
        declare_parameter("stuck_boost_mps",      0.010);
        declare_parameter("stuck_max_speed_mps",  0.150);
        declare_parameter("stuck_check_s",        3.0);
        // [v7] When the car has NOT moved at all (zero encoder displacement),
        // multiply the first boost by this factor for a more aggressive start.
        // Helps when passengers add enough load that even the first boost
        // (stuck_boost_mps) is insufficient.  1.0 = same as before.
        declare_parameter("stuck_zero_boost_factor", 2.0);
        // [v7] Minimum encoder movement per stuck_check_s to NOT be considered stuck.
        // Old default 0.005m (5 mm/3 s = 1.7 mm/s) was too low: a car carrying
        // passengers can move at 28 mm/s (below expected speed) without triggering
        // stuck detection — it plods to target in 50+ seconds.
        // Raise to 0.050m (50 mm/3 s = 16.7 mm/s) so a car slower than ~17 mm/s
        // is boosted.  Tune down if boost fires too early on empty car.
        declare_parameter("stuck_min_move_m", 0.050);

        enc_port_         = get_parameter("enc_port").as_string();
        drive_port_       = get_parameter("drive_port").as_string();
        steer_thresh_     = (float)get_parameter("straight_steer_thresh").as_double();
        speed_scale_      = (float)get_parameter("speed_scale").as_double();
        enc_fwd_speed_    = (float)get_parameter("enc_fwd_speed_mps").as_double();
        enc_rev_speed_    = (float)get_parameter("enc_rev_speed_mps").as_double();

        stuck_speed_enabled_ = get_parameter("stuck_speed_enabled").as_bool();
        stuck_boost_mps_    = (float)get_parameter("stuck_boost_mps").as_double();
        stuck_max_speed_    = (float)get_parameter("stuck_max_speed_mps").as_double();
        stuck_check_s_      = (float)get_parameter("stuck_check_s").as_double();
        stuck_zero_boost_   = (float)get_parameter("stuck_zero_boost_factor").as_double();
        stuck_min_move_m_   = (float)get_parameter("stuck_min_move_m").as_double();

        // Session speeds — start at calibrated values, never reset within a round.
        session_fwd_speed_  = enc_fwd_speed_;
        session_rev_speed_  = enc_rev_speed_;
        session_arc_rev_speed_ = 0.0f;   // [v6] set from first arc command

        RCLCPP_INFO(get_logger(),
            "enc=%s  drive=%s  thresh=%.1f°  speed_scale=%.4f"
            "  fwd_spd=%.3f  rev_spd=%.3f"
            "  stuck_speed_enabled=%s"
            "  stuck_boost=%.3f  stuck_max=%.3f  stuck_check=%.1fs"
            "  stuck_min_move=%.3fm  stuck_zero_boost_factor=%.1f",
            enc_port_.c_str(), drive_port_.c_str(),
            steer_thresh_, speed_scale_,
            enc_fwd_speed_, enc_rev_speed_,
            stuck_speed_enabled_ ? "ON" : "OFF",
            stuck_boost_mps_, stuck_max_speed_, stuck_check_s_,
            stuck_min_move_m_, stuck_zero_boost_);

        try {
            drive_ = std::make_unique<DriveSerial>(drive_port_.c_str());
            enc_   = std::make_unique<EncSerialReader>(enc_port_.c_str());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Serial open failed: %s", e.what());
            throw;
        }

        StraightDriveConfig cfg;
        cfg.slowdown_m    = 0.15f;
        cfg.stop_thresh_m = 0.02f;
        cfg.min_speed_mps = 0.025f;
        cfg.speed_kp      = 0.4f;
        cfg.speed_ki      = 0.08f;
        cfg.enc_check_s   = stuck_check_s_;
        cfg.enc_check_m   = stuck_min_move_m_;
        cfg.arm_wait_ms   = 300.0f;
        cfg.stuck_boost_mps     = stuck_boost_mps_;
        cfg.stuck_max_speed_mps = stuck_max_speed_;
        cfg.stuck_check_s       = stuck_check_s_;
        cfg.stuck_min_move_m    = stuck_min_move_m_;

        driver_ = std::make_unique<StraightDriver>(*enc_, *drive_, cfg);

        sub_ = create_subscription<std_msgs::msg::String>(
            "/autopark/cmd_json", 10,
            [this](const std_msgs::msg::String::SharedPtr msg) {
                handle_command(msg->data);
            });

        pub_status_ = create_publisher<std_msgs::msg::String>("/enc_status", 10);
        pub_result_ = create_publisher<std_msgs::msg::String>("/enc_result", 10);
        pub_busy_   = create_publisher<std_msgs::msg::Bool>("/enc_busy", 10);

        create_subscription<std_msgs::msg::Bool>(
            "/autopark/esp32_steer_ready", 20,
            [this](const std_msgs::msg::Bool::SharedPtr msg) {
                if (msg->data) steer_ready_flag_.store(true);
            });

        // FIX: publish /enc_status at 25ms (40Hz) instead of 100ms (10Hz).
        // _wait_arc_imu checks enc every 40ms.  With 100ms updates the arc
        // overshoots by up to 6mm (0.26°) because it sees a stale reading.
        // At 25ms updates, worst-case overshoot drops to 2.4mm (0.10°).
        timer_ = create_wall_timer(
            std::chrono::milliseconds(25),
            [this]() { publish_status(); });

        {
            int fd = ::open(enc_port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
            if (fd >= 0) { ::write(fd, "r\n", 2); ::close(fd); }
        }

        RCLCPP_INFO(get_logger(), "Encoder bridge v6 ready"
            "  session_fwd=%.3f  session_rev=%.3f  session_arc=%.3f",
            session_fwd_speed_, session_rev_speed_, session_arc_rev_speed_);
    }

private:
    void set_busy(bool b) {
        auto msg = std_msgs::msg::Bool();
        msg.data = b;
        pub_busy_->publish(msg);
    }

    void handle_command(const std::string& json) {
        std::string type = jsonStr(json, "type", "");

        // ── Non-drive: abort everything ───────────────────────
        if (type != "drive") {
            if (type == "stop" || type == "disarm" || type == "manual") {
                // [v5] abort driver_ BEFORE joining threads (deadlock fix)
                driver_->abort();

                // [v6] Abort arc monitor first
                if (arc_monitor_thread_.joinable()) {
                    arc_monitor_abort_.store(true);
                    arc_monitor_thread_.join();
                    arc_monitor_abort_.store(false);
                }

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

        float dist_m_raw = jsonFloat(json, "dist_m", -1.0f);
        if (dist_m_raw < 0.001f)
            dist_m_raw = jsonFloat(json, "target_dist_m", -1.0f);

        bool  act_hold   = jsonBool (json, "steer_active_hold");

        RCLCPP_INFO(get_logger(),
            "CMD  gear=%+d speed=%.4f steer=%.1f° dur=%.2f "
            "dist_m=%.4f active_hold=%d"
            "  sess_fwd=%.4f  sess_rev=%.4f  sess_arc=%.4f",
            gear, speed_mps, steer_deg, duration, dist_m_raw, (int)act_hold,
            session_fwd_speed_, session_rev_speed_, session_arc_rev_speed_);

        // ── Straight/encoder intercept condition ───────────────
        bool speed_ok  = speed_mps > 0.0f;
        bool gear_ok   = gear != 0;
        bool dist_ok   = dist_m_raw > 0.001f;
        bool steer_ok  = std::fabs(steer_deg) <= steer_thresh_;
        bool enc_force = jsonBool(json, "use_encoder");

        bool intercept = speed_ok && gear_ok && dist_ok &&
                         (gear > 0 || steer_ok || enc_force);

        if (!intercept) {
            RCLCPP_INFO(get_logger(),
                "  → %s (gear=%+d speed_ok=%d dist_ok=%d steer_ok=%d enc_force=%d)"
                " — serial_bridge handles",
                (gear < 0 && !steer_ok && !enc_force) ? "Arc reverse — IMU handles"
                                                       : "No target dist or zero speed",
                gear, (int)speed_ok, (int)dist_ok, (int)steer_ok, (int)enc_force);

            if (!speed_ok) {
                steer_ready_flag_.store(false);
                RCLCPP_INFO(get_logger(),
                    "  → Settle CMD (gear=%+d steer=%.1f°): "
                    "steer_ready_flag reset, awaiting servo",
                    gear, steer_deg);
            }

            // ── [v6] Arc reverse stuck monitor ─────────────────
            // serial_bridge + IMU handle the arc stop, but if the car
            // stalls due to passenger weight we need to boost the speed.
            // Start arc_monitor_thread_ to watch encoder and take over
            // with driveWithSteer() at a boosted speed if stuck.
            if (speed_ok && gear_ok && gear < 0 && dist_ok) {
                // Initialise session_arc_rev_speed_ from the command the first
                // time we see an arc command (or if command is faster than
                // current session — e.g. after a reset).
                // [v7] Clamp up to enc_rev_speed_: the cmd speed_mps is the
                // serial_bridge protocol speed (= speed_scale = 0.01 m/s).
                // Using that as the arc monitor start speed means the first
                // driveWithSteer() boost fires at 0.01→0.02 m/s — far too slow.
                // Start at enc_rev_speed_ (0.04 m/s) so the arc moves immediately.
                float arc_init = std::max(speed_mps, enc_rev_speed_);
                if (arc_init > session_arc_rev_speed_)
                    session_arc_rev_speed_ = arc_init;

                startArcMonitor(gear, steer_deg);
            }

            return;
        }

        // ── Compute target distance ────────────────────────────
        float target_m;
        if (dist_m_raw > 0.001f) {
            target_m = dist_m_raw;
            RCLCPP_INFO(get_logger(),
                "  → Straight (dist from cmd): target=%.4fm", target_m);
        } else {
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

        if (!enc_ || !enc_->isValid()) {
            RCLCPP_WARN(get_logger(),
                "  → Encoder NOT valid (pkts=%u) — serial_bridge handles (time-based fallback)",
                enc_ ? enc_->packetCount() : 0u);
            return;
        }

        bool forward = (gear > 0);

        // ── FORWARD (gear=+1): Monitor mode with v5 stuck detection ──────
        if (forward) {
            driver_->abort();
            if (arc_monitor_thread_.joinable()) {   // [v6]
                arc_monitor_abort_.store(true);
                arc_monitor_thread_.join();
                arc_monitor_abort_.store(false);
            }
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
                "  → Monitor mode (serial_bridge→ESP32): target=%.4fm"
                "  safety=%.1fs  session_fwd=%.4f",
                target_m, safety_s, session_fwd_speed_);

            monitor_thread_ = std::thread(
                [this, target_m, safety_s, steer_deg]() {
                try {
                    for (int i = 0; i < 30 && !enc_->isValid(); i++)
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));

                    float start_dist = enc_->getSnapshot().rightDistM;
                    auto  t_start    = std::chrono::steady_clock::now();

                    uint64_t stuckNextMs  = static_cast<uint64_t>(stuck_check_s_ * 1000.0f);
                    float    stuckRefDist = 0.0f;
                    bool     in_drive_mode = false;

                    while (!monitor_abort_.load()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                        float dist    = std::fabs(enc_->getSnapshot().rightDistM - start_dist);
                        float elapsed = std::chrono::duration<float>(
                            std::chrono::steady_clock::now() - t_start).count();
                        auto  elapsedMs = static_cast<uint64_t>(elapsed * 1000.0f);

                        if (!in_drive_mode && elapsedMs >= stuckNextMs) {
                            float moved = dist - stuckRefDist;
                            if (stuck_speed_enabled_ && moved < stuck_min_move_m_) {
                                float oldSpeed = session_fwd_speed_;
                                // [v7] Zero-movement boost: if car hasn't moved at ALL
                                // (passengers stalling from rest), use a larger first boost
                                // to overcome starting torque.
                                float boost_mult = (dist < 0.001f) ? stuck_zero_boost_ : 1.0f;
                                session_fwd_speed_ = std::min(
                                    session_fwd_speed_ + stuck_boost_mps_ * boost_mult,
                                    stuck_max_speed_);
                                float remaining   = std::max(0.001f, target_m - dist);
                                float rem_timeout = std::max(5.0f, safety_s - elapsed + 5.0f);

                                RCLCPP_INFO(rclcpp::get_logger("encoder_bridge"),
                                    "Fwd Monitor STUCK (moved=%.5fm in %.1fs)"
                                    " → boost session_fwd %.4f→%.4f m/s"
                                    " (x%.1f zero-boost)"
                                    " → Drive mode, remaining=%.4fm",
                                    moved, elapsed,
                                    oldSpeed, session_fwd_speed_,
                                    boost_mult, remaining);

                                set_busy(true);
                                in_drive_mode = true;

                                StraightResult res = driver_->driveStraight(
                                    remaining, true,
                                    session_fwd_speed_,
                                    rem_timeout,
                                    steer_deg,
                                    /*skipArm=*/true);

                                set_busy(false);

                                float total_dist = dist + driver_->lastDist();
                                const char* res_str = straightResultName(res);

                                RCLCPP_INFO(rclcpp::get_logger("encoder_bridge"),
                                    "Monitor→Drive done: %s"
                                    "  total_dist=%.4fm  target=%.4fm"
                                    "  session_fwd now=%.4f",
                                    res_str, total_dist, target_m, session_fwd_speed_);

                                auto msg = std_msgs::msg::String();
                                char buf[256];
                                snprintf(buf, sizeof(buf),
                                    "{\"enc_result\":\"%s\","
                                    "\"dist\":%.4f,\"target\":%.4f}",
                                    res_str, total_dist, target_m);
                                msg.data = buf;
                                pub_result_->publish(msg);
                                return;
                            }
                            stuckRefDist  = dist;
                            stuckNextMs  += static_cast<uint64_t>(stuck_check_s_ * 1000.0f);
                        }

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
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(rclcpp::get_logger("encoder_bridge"),
                        "Monitor thread exception: %s — publishing TIMEOUT", e.what());
                    try {
                        set_busy(false);
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
                    set_busy(false);
                }
            });
            return;
        }

        // ── REVERSE straight (gear=-1): Drive mode ─────────────────────────
        driver_->abort();
        if (arc_monitor_thread_.joinable()) {   // [v6]
            arc_monitor_abort_.store(true);
            arc_monitor_thread_.join();
            arc_monitor_abort_.store(false);
        }
        if (monitor_thread_.joinable()) {
            monitor_abort_.store(true);
            driver_->abort();
            monitor_thread_.join();
            monitor_abort_.store(false);
        }
        if (drive_thread_.joinable()) {
            driver_->abort();
            drive_thread_.join();
        }

        float timeout = target_m / 0.025f + 30.0f;

        RCLCPP_INFO(get_logger(),
            "  Safety timeout=%.1fs  (%.3fm / 0.025 m/s + 30s buffer)",
            timeout, target_m);
        RCLCPP_INFO(get_logger(),
            "  → Encoder drive (rev): target=%.4fm  session_rev_speed=%.4f m/s"
            "  steer=%.1f°  gear=-1%s",
            target_m, session_rev_speed_, steer_deg,
            enc_force ? "  (use_encoder forced — rear-cam steer)" : "");

        set_busy(true);

        drive_thread_ = std::thread(
            [this, target_m, timeout, steer_deg]() {
            // [v7a] try-catch: set_busy(false) MUST be called on every exit path.
            // Without this, a serial exception inside driveStraight() left
            // enc_busy=True forever → serial_bridge blocked all kinetic REVERSE
            // commands silently (no log, debug_serial=false by default).
            try {
                StraightResult result = driver_->driveStraight(
                    target_m,
                    false,
                    session_rev_speed_,
                    timeout,
                    steer_deg,
                    /*skipArm=*/true);

                RCLCPP_INFO(rclcpp::get_logger("encoder_bridge"),
                    "Straight done: %s  dist=%.4fm  target=%.4fm"
                    "  session_rev now=%.4f",
                    straightResultName(result), driver_->lastDist(), target_m,
                    session_rev_speed_);

                set_busy(false);

                auto msg = std_msgs::msg::String();
                char buf[256];
                snprintf(buf, sizeof(buf),
                    "{\"enc_result\":\"%s\",\"dist\":%.4f,\"target\":%.4f}",
                    straightResultName(result), driver_->lastDist(), target_m);
                msg.data = buf;
                pub_result_->publish(msg);

            } catch (const std::exception& e) {
                RCLCPP_ERROR(rclcpp::get_logger("encoder_bridge"),
                    "Drive thread exception: %s — releasing enc_busy", e.what());
                set_busy(false);
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
                    "Drive thread unknown exception — releasing enc_busy");
                set_busy(false);
                try {
                    auto msg = std_msgs::msg::String();
                    char buf[256];
                    snprintf(buf, sizeof(buf),
                        "{\"enc_result\":\"ERROR\",\"dist\":0.0,\"target\":%.4f}",
                        target_m);
                    msg.data = buf;
                    pub_result_->publish(msg);
                } catch (...) {}
            }
        });
    }

    // ── [v6] Arc stuck monitor ─────────────────────────────────────────────
    // Called for non-intercepted reverse arc commands (Move2, gear=-1, steer>thresh).
    // serial_bridge + IMU still control arc stop.  This thread watches encoder
    // distance; if the car stalls it takes over ttyUSB2 at a boosted speed.
    // When autopark_master sends stop (IMU arc done), handle_command aborts it.
    void startArcMonitor(int gear, float steer_deg) {
        // Abort previous arc monitor if any
        if (arc_monitor_thread_.joinable()) {
            arc_monitor_abort_.store(true);
            arc_monitor_thread_.join();
            arc_monitor_abort_.store(false);
        }

        RCLCPP_INFO(get_logger(),
            "[v6] Arc monitor start: gear=%d steer=%.1f° session_arc=%.4f m/s",
            gear, steer_deg, session_arc_rev_speed_);

        arc_monitor_thread_ = std::thread([this, gear, steer_deg]() {
            try {
                // Wait for encoder to be ready
                for (int i = 0; i < 30 && !enc_->isValid(); i++)
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));

                if (!enc_->isValid()) {
                    RCLCPP_WARN(rclcpp::get_logger("encoder_bridge"),
                        "Arc monitor: encoder not valid — skipping");
                    return;
                }

                float    startRight   = enc_->getSnapshot().rightDistM;
                uint64_t stuckNextMs  = nowMs_br() +
                    static_cast<uint64_t>(stuck_check_s_ * 1000.0f);
                float    stuckRefDist = 0.0f;
                int      boostCount   = 0;
                bool     in_boost     = false;

                while (!arc_monitor_abort_.load()) {
                    uint64_t loopStart = nowMs_br();

                    float dist = std::fabs(enc_->getSnapshot().rightDistM - startRight);

                    // ── Stuck check every stuck_check_s ────────────────────
                    if (nowMs_br() >= stuckNextMs) {
                        float moved = dist - stuckRefDist;
                        if (stuck_speed_enabled_ && moved < stuck_min_move_m_) {
                            if (session_arc_rev_speed_ < stuck_max_speed_) {
                                float old = session_arc_rev_speed_;
                                // [v7] Zero-movement boost for passengers stalling arc from rest
                                float boost_mult = (dist < 0.001f) ? stuck_zero_boost_ : 1.0f;
                                session_arc_rev_speed_ = std::min(
                                    session_arc_rev_speed_ + stuck_boost_mps_ * boost_mult,
                                    stuck_max_speed_);
                                boostCount++;

                                if (!in_boost) {
                                    set_busy(true);   // block serial_bridge arc cmd
                                    in_boost = true;
                                }

                                RCLCPP_INFO(rclcpp::get_logger("encoder_bridge"),
                                    "[v7] Arc STUCK #%d (moved=%.5fm in %.1fs)"
                                    " → boost %.4f→%.4f m/s (x%.1f zero-boost)  steer=%.1f°",
                                    boostCount, moved, stuck_check_s_,
                                    old, session_arc_rev_speed_,
                                    boost_mult, steer_deg);
                            } else {
                                RCLCPP_WARN(rclcpp::get_logger("encoder_bridge"),
                                    "[v6] Arc STUCK at max speed %.4f (moved=%.5fm)"
                                    " — waiting for IMU arc stop",
                                    session_arc_rev_speed_, moved);
                            }
                        }
                        stuckRefDist = dist;
                        stuckNextMs += static_cast<uint64_t>(stuck_check_s_ * 1000.0f);
                    }

                    // ── Drive at boosted speed every loop tick ─────────────
                    if (in_boost) {
                        drive_->driveWithSteer(session_arc_rev_speed_, gear,
                                               steer_deg, 0.3f);
                    }

                    // ── Sleep remainder of 20 ms loop ──────────────────────
                    uint64_t elapsed = nowMs_br() - loopStart;
                    if (elapsed < 20)
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(20 - elapsed));
                }

                // ── Clean exit ─────────────────────────────────────────────
                if (in_boost) {
                    drive_->stop("arc_boost_done");
                    set_busy(false);
                }

                RCLCPP_INFO(rclcpp::get_logger("encoder_bridge"),
                    "[v6] Arc monitor done: boosts=%d  session_arc=%.4f m/s",
                    boostCount, session_arc_rev_speed_);

            } catch (const std::exception& e) {
                RCLCPP_ERROR(rclcpp::get_logger("encoder_bridge"),
                    "[v6] Arc monitor exception: %s", e.what());
                set_busy(false);
            } catch (...) {
                RCLCPP_ERROR(rclcpp::get_logger("encoder_bridge"),
                    "[v6] Arc monitor unknown exception");
                set_busy(false);
            }
        });
    }

    void publish_status() {
        if (!enc_ || !enc_->isValid()) return;
        auto snap = enc_->getSnapshot();
        char buf[320];
        // [v6] Publish BOTH left and right encoder data.
        // lr_spd_delta = right - left speed (positive → right faster → left slower → car drifts right).
        // Use this value to tune left_pwm_boost in esp32_drive_controller.ino:
        //   lr_spd_delta ~ 0.010 m/s  →  try left_pwm_boost = 10
        //   lr_spd_delta ~ 0.020 m/s  →  try left_pwm_boost = 15–20
        float lr_delta = snap.rightSpeedMs - snap.leftSpeedMs;
        snprintf(buf, sizeof(buf),
            "{\"rc\":%ld,\"rd\":%.4f,\"rrpm\":%.2f,\"rspd\":%.3f"
            ",\"lc\":%ld,\"ld\":%.4f,\"lrpm\":%.2f,\"lspd\":%.3f"
            ",\"lr_delta\":%.3f"
            ",\"sess_fwd\":%.4f,\"sess_rev\":%.4f,\"sess_arc\":%.4f}",
            snap.rightCount, snap.rightDistM,
            snap.rightWRpm,  snap.rightSpeedMs,
            snap.leftCount,  snap.leftDistM,
            snap.leftWRpm,   snap.leftSpeedMs,
            lr_delta,
            session_fwd_speed_, session_rev_speed_, session_arc_rev_speed_);
        auto msg = std_msgs::msg::String();
        msg.data = buf;
        pub_status_->publish(msg);
    }

    // ── Members ──────────────────────────────────────────────────────────────
    std::string enc_port_, drive_port_;
    float       steer_thresh_, speed_scale_;
    float       enc_fwd_speed_, enc_rev_speed_;

    // Session speeds — persist for the whole parking round
    float session_fwd_speed_;        // Move1 forward  (boosted by monitor thread)
    float session_rev_speed_;        // Move3 reverse straight (boosted by driveStraight ref)
    float session_arc_rev_speed_;    // [v6] Move2 arc reverse (boosted by arc monitor)

    // Stuck detection parameters
    bool  stuck_speed_enabled_;  // master switch — set true for 98 kg passenger load
    float stuck_boost_mps_;
    float stuck_max_speed_;
    float stuck_check_s_;
    float stuck_min_move_m_;
    float stuck_zero_boost_;    // [v7] multiplier when car hasn't moved at all

    std::unique_ptr<DriveSerial>      drive_;
    std::unique_ptr<EncSerialReader>  enc_;
    std::unique_ptr<StraightDriver>   driver_;

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr    pub_status_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr    pub_result_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr      pub_busy_;
    rclcpp::TimerBase::SharedPtr                           timer_;

    std::thread drive_thread_;
    std::thread monitor_thread_;
    std::atomic<bool> monitor_abort_{false};
    std::atomic<bool> steer_ready_flag_{false};

    // [v6] Arc move stuck monitor
    std::thread arc_monitor_thread_;
    std::atomic<bool> arc_monitor_abort_{false};
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
