#include <chrono>
#include <memory>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"

using namespace std::chrono_literals;

#define MPU6050_ADDR 0x68
#define PWR_MGMT_1 0x6B
#define ACCEL_XOUT_H 0x3B

class Mpu6050NodeCpp : public rclcpp::Node
{
public:
  Mpu6050NodeCpp() : Node("mpu6050_cpp_node")
  {
    publisher_ = this->create_publisher<sensor_msgs::msg::Imu>("/imu/data_raw", 10);
    
    // เปิดการเชื่อมต่อ I2C Bus 5
    if ((i2c_file_ = open("/dev/i2c-5", O_RDWR)) < 0) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open the i2c bus");
      return;
    }
    if (ioctl(i2c_file_, I2C_SLAVE, MPU6050_ADDR) < 0) {
      RCLCPP_ERROR(this->get_logger(), "Failed to acquire bus access and/or talk to slave");
      return;
    }

    // ปลุก MPU6050 (เขียน 0 ไปที่ PWR_MGMT_1)
    uint8_t buffer[2] = {PWR_MGMT_1, 0x00};
    if (write(i2c_file_, buffer, 2) != 2) {
      RCLCPP_ERROR(this->get_logger(), "Failed to wake up MPU6050");
    } else {
      RCLCPP_INFO(this->get_logger(), "MPU6050 Woken up successfully! Running at 200Hz.");
    }

    // ตั้ง Timer ความถี่ 200 Hz (5ms) เพื่อป้อนข้อมูลให้ VINS-Mono อย่างจุใจ
    timer_ = this->create_wall_timer(5ms, std::bind(&Mpu6050NodeCpp::timer_callback, this));
  }

  ~Mpu6050NodeCpp() {
    close(i2c_file_);
  }

private:
  void timer_callback()
  {
    // สั่งให้อ่านเริ่มตั้งแต่ Address ACCEL_XOUT_H
    uint8_t reg[1] = {ACCEL_XOUT_H};
    if (write(i2c_file_, reg, 1) != 1) return;

    // อ่านข้อมูลรวดเดียว 14 ไบต์ (AccX, AccY, AccZ, Temp, GyroX, GyroY, GyroZ)
    uint8_t data[14];
    if (read(i2c_file_, data, 14) != 14) return;

    auto msg = sensor_msgs::msg::Imu();
    
    // 1. Timestamp สำคัญที่สุด!
    msg.header.stamp = this->get_clock()->now();
    msg.header.frame_id = "imu_link";

    // 2. แปลงค่า Accelerometer (หน่วย m/s^2)
    int16_t acc_x = (data[0] << 8) | data[1];
    int16_t acc_y = (data[2] << 8) | data[3];
    int16_t acc_z = (data[4] << 8) | data[5];
    msg.linear_acceleration.x = (acc_x / 16384.0) * 9.80665;
    msg.linear_acceleration.y = (acc_y / 16384.0) * 9.80665;
    msg.linear_acceleration.z = (acc_z / 16384.0) * 9.80665;

    // 3. แปลงค่า Gyroscope (หน่วย rad/s)
    int16_t gyro_x = (data[8] << 8) | data[9];
    int16_t gyro_y = (data[10] << 8) | data[11];
    int16_t gyro_z = (data[12] << 8) | data[13];
    msg.angular_velocity.x = (gyro_x / 131.0) * (M_PI / 180.0);
    msg.angular_velocity.y = (gyro_y / 131.0) * (M_PI / 180.0);
    msg.angular_velocity.z = (gyro_z / 131.0) * (M_PI / 180.0);

    // 4. Covariance (บอกให้ VINS รู้ว่าเซนเซอร์เรามี Noise)
    msg.linear_acceleration_covariance[0] = 0.04;
    msg.linear_acceleration_covariance[4] = 0.04;
    msg.linear_acceleration_covariance[8] = 0.04;
    msg.angular_velocity_covariance[0] = 0.002;
    msg.angular_velocity_covariance[4] = 0.002;
    msg.angular_velocity_covariance[8] = 0.002;

    publisher_->publish(msg);
  }

  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  int i2c_file_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Mpu6050NodeCpp>());
  rclcpp::shutdown();
  return 0;
}
