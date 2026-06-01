#pragma once
// ============================================================
//  enc_serial_reader.hpp  —  RDK X5: Read encoder data
//                             from Encoder ESP32 over serial
//
//  Runs a background thread that continuously reads lines
//  from the encoder ESP32 serial port and parses the JSON.
//  Main thread calls getSnapshot() for the latest values.
// ============================================================

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

// ── Snapshot of latest encoder values ────────────────────────
struct EncSnapshot {
    long  leftCount   = 0;
    long  rightCount  = 0;
    float leftDistM   = 0.0f;
    float rightDistM  = 0.0f;
    float leftWRpm    = 0.0f;
    float rightWRpm   = 0.0f;
    float leftMRpm    = 0.0f;
    float rightMRpm   = 0.0f;
    float rightSpeedMs= 0.0f;
    bool  valid       = false;    // true once first packet received
    uint64_t timestampUs = 0;
};

// ── EncSerialReader ───────────────────────────────────────────
class EncSerialReader {
public:
    // port   = serial port of encoder ESP32 (e.g. /dev/ttyUSB0)
    // baud   = must match encoder sketch (115200)
    explicit EncSerialReader(const char* port, int baud_const = B115200) {
        fd_ = ::open(port, O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_ < 0)
            throw std::runtime_error(
                std::string("EncSerialReader: cannot open ") + port);

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

        running_ = true;
        thread_  = std::thread(&EncSerialReader::readLoop, this);
    }

    ~EncSerialReader() {
        running_ = false;
        if (thread_.joinable()) thread_.join();
        if (fd_ >= 0) ::close(fd_);
    }

    EncSerialReader(const EncSerialReader&)            = delete;
    EncSerialReader& operator=(const EncSerialReader&) = delete;

    // Thread-safe snapshot of latest encoder values
    EncSnapshot getSnapshot() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return snap_;
    }

    // True once at least one valid packet received
    bool isValid() const { return snap_.valid; }

    // Count of received packets (debug)
    uint32_t packetCount() const { return pktCount_.load(); }

private:
    void readLoop() {
        char  buf[512];
        std::string line;

        while (running_) {
            int n = ::read(fd_, buf, sizeof(buf) - 1);
            if (n <= 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            buf[n] = '\0';
            for (int i = 0; i < n; i++) {
                char c = buf[i];
                if (c == '\n') {
                    if (!line.empty()) parseLine(line);
                    line.clear();
                } else if (c != '\r') {
                    line += c;
                    if (line.size() > 400) line.clear();
                }
            }
        }
    }

    // Parse: {"type":"enc","lc":N,"rc":N,"ld":F,"rd":F,"lrpm":F,"rrpm":F,"lmrpm":F,"rmrpm":F}
    void parseLine(const std::string& line) {
        if (line.find("\"type\":\"enc\"") == std::string::npos) return;

        EncSnapshot s;
        s.timestampUs = nowUs();
        s.valid = true;

        // Simple field extraction — no external JSON library needed
        auto getFloat = [&](const char* key) -> float {
            std::string k = std::string("\"") + key + "\":";
            auto pos = line.find(k);
            if (pos == std::string::npos) return 0.0f;
            return std::stof(line.substr(pos + k.size()));
        };
        auto getLong = [&](const char* key) -> long {
            std::string k = std::string("\"") + key + "\":";
            auto pos = line.find(k);
            if (pos == std::string::npos) return 0L;
            return std::stol(line.substr(pos + k.size()));
        };

        s.leftCount  = getLong("lc");
        s.rightCount = getLong("rc");
        s.leftDistM  = getFloat("ld");
        s.rightDistM = getFloat("rd");
        s.leftWRpm   = getFloat("lrpm");
        s.rightWRpm  = getFloat("rrpm");
        s.leftMRpm   = getFloat("lmrpm");
        s.rightMRpm  = getFloat("rmrpm");
        s.rightSpeedMs= s.rightWRpm / 60.0f * 0.5586f;

        {
            std::lock_guard<std::mutex> lock(mtx_);
            snap_ = s;
        }
        pktCount_++;
    }

    static uint64_t nowUs() {
        using namespace std::chrono;
        return static_cast<uint64_t>(
            duration_cast<microseconds>(
                steady_clock::now().time_since_epoch()).count());
    }

    int   fd_ = -1;
    std::atomic<bool>    running_  { false };
    std::thread          thread_;
    mutable std::mutex   mtx_;
    EncSnapshot          snap_;
    std::atomic<uint32_t> pktCount_{ 0 };
};
