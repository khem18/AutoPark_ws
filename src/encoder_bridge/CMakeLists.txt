cmake_minimum_required(VERSION 3.16)
project(encoder_bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(encoder_bridge encoder_bridge.cpp)

ament_target_dependencies(encoder_bridge rclcpp std_msgs)

target_include_directories(encoder_bridge PRIVATE
  ${CMAKE_SOURCE_DIR}
)

target_link_libraries(encoder_bridge pthread)

install(TARGETS encoder_bridge
  DESTINATION lib/${PROJECT_NAME})

ament_package()
