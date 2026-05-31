#pragma once
// ============================================================
//  diff_encoder.hpp  —  Differential Drive Encoder for RDK X5
//
//  Left  Motor:  C1 = GPIO 27  (gpiochip4 line 27)
//                C2 = GPIO 17  (gpiochip4 line 17)
//  Right Motor:  C1 = GPIO 24  (gpiochip4 line 24)
//                C2 = GPIO 23  (gpiochip4 line 23)
//
//  Gear ratio: 74.25 : 1  (all output values are wheel-side)
//
//  Build:
//    g++ -std=c++17 -O2 -o diff_encoder_demo diff_encoder_demo.cpp \
//        -lgpiod -lpthread
// ============================================================

#include <gpiod.h>
#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

// ============================================================
//  Config structs defined at TOP LEVEL (outside any class).
//  This avoids the GCC "default member initializer required
//  before end of enclosing class" error.
// ============================================================

struct SingleEncoderConfig {
    const char* chip_name  = "gpiochip4";
    unsigned    pin_c1     = 0;
    unsigned    pin_c2     = 0;
    int         ppr        = 1000;
    float       gear_ratio = 74.25f;
    float       wheel_circ = 0.20f;
    const char* label      = "enc";
};

struct DiffDriveConfig {
    const char* chip_name  = "gpiochip4";
    unsigned    left_c1    = 27;
    unsigned    left_c2    = 17;
    unsigned    right_c1   = 24;
    unsigned    right_c2   = 23;
    int         ppr        = 1000;
    float       gear_ratio = 74.25f;
    float       wheel_circ_m = 0.20f;
};

// ============================================================
//  SingleEncoder
// ============================================================
class SingleEncoder {
public:
    using Config = SingleEncoderConfig;

    explicit SingleEncoder(Config cfg = Config{}) : cfg_(cfg) {
        open_gpio();
        start_thread();
    }

    ~SingleEncoder() {
        stop_thread();
        close_gpio();
    }

    SingleEncoder(const SingleEncoder&)            = delete;
    SingleEncoder& operator=(const SingleEncoder&) = delete;

    // ── Motor shaft ──────────────────────────────────────────
    long  motorCount() const { return count_.load(std::memory_order_relaxed); }
    int   direction()  const { return dir_.load(std::memory_order_relaxed); }

    float motorRpm() const {
        uint64_t iv   = interval_us_.load(std::memory_order_relaxed);
        uint64_t last = last_rise_us_.load(std::memory_order_relaxed);
        if (iv == 0 || (now_us() - last) > 200'000ULL) return 0.0f;
        return (60.0f * 1'000'000.0f) / (static_cast<float>(iv) * cfg_.ppr);
    }

    // ── Wheel side (÷ gear_ratio) ─────────────────────────────
    float wheelRpm()     const { return motorRpm() / cfg_.gear_ratio; }
    float wheelSpeedMs() const { return (wheelRpm() / 60.0f) * cfg_.wheel_circ; }
    float wheelRevs()    const {
        long c = count_.load(std::memory_order_relaxed);
        float motorRevs = static_cast<float>(c) / (static_cast<float>(cfg_.ppr) * 4.0f);
        return motorRevs / cfg_.gear_ratio;
    }
    float wheelDistM()   const { return wheelRevs() * cfg_.wheel_circ; }

    void reset() { count_.store(0, std::memory_order_relaxed); }

private:
    void open_gpio() {
        chip_ = gpiod_chip_open_by_name(cfg_.chip_name);
        if (!chip_)
            throw std::runtime_error(
                std::string("[") + cfg_.label + "] Cannot open " + cfg_.chip_name +
                " — run with sudo");

        line_c1_ = gpiod_chip_get_line(chip_, cfg_.pin_c1);
        if (!line_c1_)
            throw std::runtime_error(
                std::string("[") + cfg_.label + "] get_line C1=" +
                std::to_string(cfg_.pin_c1) + " failed");
        if (gpiod_line_request_both_edges_events(line_c1_, cfg_.label) < 0)
            throw std::runtime_error(
                std::string("[") + cfg_.label + "] request_edges on C1 failed");

        line_c2_ = gpiod_chip_get_line(chip_, cfg_.pin_c2);
        if (!line_c2_)
            throw std::runtime_error(
                std::string("[") + cfg_.label + "] get_line C2=" +
                std::to_string(cfg_.pin_c2) + " failed");
        if (gpiod_line_request_input(line_c2_, cfg_.label) < 0)
            throw std::runtime_error(
                std::string("[") + cfg_.label + "] request_input on C2 failed");
    }

    void close_gpio() {
        if (line_c1_) { gpiod_line_release(line_c1_); line_c1_ = nullptr; }
        if (line_c2_) { gpiod_line_release(line_c2_); line_c2_ = nullptr; }
        if (chip_)    { gpiod_chip_close(chip_);       chip_    = nullptr; }
    }

    void start_thread() { running_ = true;  thread_ = std::thread(&SingleEncoder::reader_loop, this); }
    void stop_thread()  { running_ = false; if (thread_.joinable()) thread_.join(); }

    void reader_loop() {
        struct timespec timeout { 0, 100'000'000L };
        while (running_) {
            if (gpiod_line_event_wait(line_c1_, &timeout) <= 0) continue;
            struct gpiod_line_event ev;
            if (gpiod_line_event_read(line_c1_, &ev) < 0) continue;

            uint64_t now  = now_us();
            uint64_t last = last_rise_us_.load(std::memory_order_relaxed);
            bool rising   = (ev.event_type == GPIOD_LINE_EVENT_RISING_EDGE);

            if (rising) {
                if (last > 0) interval_us_.store(now - last, std::memory_order_relaxed);
                last_rise_us_.store(now, std::memory_order_relaxed);
            }

            int c2   = gpiod_line_get_value(line_c2_);
            int step = 0;
            if ( rising && c2 == 0) step = +1;
            if ( rising && c2 == 1) step = -1;
            if (!rising && c2 == 1) step = +1;
            if (!rising && c2 == 0) step = -1;

            if (step != 0) {
                count_.fetch_add(step, std::memory_order_relaxed);
                dir_.store(step, std::memory_order_relaxed);
            }
        }
    }

    static uint64_t now_us() {
        using namespace std::chrono;
        return static_cast<uint64_t>(
            duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count());
    }

    Config      cfg_;
    gpiod_chip* chip_    = nullptr;
    gpiod_line* line_c1_ = nullptr;
    gpiod_line* line_c2_ = nullptr;

    std::atomic<long>     count_       { 0 };
    std::atomic<int>      dir_         { 0 };
    std::atomic<uint64_t> last_rise_us_{ 0 };
    std::atomic<uint64_t> interval_us_ { 0 };
    std::atomic<bool>     running_     { false };
    std::thread           thread_;
};

// ============================================================
//  DiffDriveEncoder
// ============================================================
class DiffDriveEncoder {
public:
    using Config = DiffDriveConfig;

    explicit DiffDriveEncoder(Config cfg = Config{}) {
        SingleEncoderConfig lc;
        lc.chip_name  = cfg.chip_name;
        lc.pin_c1     = cfg.left_c1;
        lc.pin_c2     = cfg.left_c2;
        lc.ppr        = cfg.ppr;
        lc.gear_ratio = cfg.gear_ratio;
        lc.wheel_circ = cfg.wheel_circ_m;
        lc.label      = "enc_left";

        SingleEncoderConfig rc;
        rc.chip_name  = cfg.chip_name;
        rc.pin_c1     = cfg.right_c1;
        rc.pin_c2     = cfg.right_c2;
        rc.ppr        = cfg.ppr;
        rc.gear_ratio = cfg.gear_ratio;
        rc.wheel_circ = cfg.wheel_circ_m;
        rc.label      = "enc_right";

        left_  = std::make_unique<SingleEncoder>(lc);
        right_ = std::make_unique<SingleEncoder>(rc);
    }

    // Left
    long  leftMotorCount()  const { return left_->motorCount(); }
    int   leftDir()         const { return left_->direction(); }
    float leftMotorRpm()    const { return left_->motorRpm(); }
    float leftWheelRpm()    const { return left_->wheelRpm(); }
    float leftSpeedMs()     const { return left_->wheelSpeedMs(); }
    float leftDistM()       const { return left_->wheelDistM(); }

    // Right
    long  rightMotorCount() const { return right_->motorCount(); }
    int   rightDir()        const { return right_->direction(); }
    float rightMotorRpm()   const { return right_->motorRpm(); }
    float rightWheelRpm()   const { return right_->wheelRpm(); }
    float rightSpeedMs()    const { return right_->wheelSpeedMs(); }
    float rightDistM()      const { return right_->wheelDistM(); }

    // Combined
    float avgWheelRpm()     const { return (leftWheelRpm()  + rightWheelRpm())  * 0.5f; }
    float avgSpeedMs()      const { return (leftSpeedMs()   + rightSpeedMs())   * 0.5f; }

    void  resetAll()              { left_->reset(); right_->reset(); }

private:
    std::unique_ptr<SingleEncoder> left_;
    std::unique_ptr<SingleEncoder> right_;
};
