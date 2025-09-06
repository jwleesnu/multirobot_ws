cmake_minimum_required(VERSION 3.8)
project(ground_station)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)



# 라이브러리
add_library(${PROJECT_NAME} SHARED src/global_mapper.cpp src/optitrack_calib.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/ground_station>
  $<INSTALL_INTERFACE:include/ground_station>)
target_link_libraries(${PROJECT_NAME} PRIVATE yaml-cpp)
ament_target_dependencies(${PROJECT_NAME} PUBLIC
  rclcpp OpenCV sensor_msgs nav_msgs geometry_msgs std_msgs yaml-cpp Eigen3 tf2 tf2_ros tf2_geometry_msgs
)

# 실행파일
add_executable(global_mapper exe/global_mapper.cpp)
target_link_libraries(global_mapper PRIVATE ${PROJECT_NAME} Eigen3::Eigen yaml-cpp)
ament_target_dependencies(global_mapper PUBLIC
  rclcpp OpenCV sensor_msgs nav_msgs geometry_msgs std_msgs yaml-cpp Eigen3 tf2 tf2_ros tf2_geometry_msgs
)

add_executable(optitrack_calib exe/optitrack_calib.cpp)
target_link_libraries(optitrack_calib PRIVATE ${PROJECT_NAME} Eigen3::Eigen yaml-cpp)
ament_target_dependencies(optitrack_calib PUBLIC
  rclcpp OpenCV sensor_msgs nav_msgs geometry_msgs std_msgs yaml-cpp Eigen3 tf2 tf2_ros tf2_geometry_msgs
)

# 3) 설치
install(TARGETS ${PROJECT_NAME} global_mapper optitrack_calib
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include/)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY param DESTINATION share/${PROJECT_NAME}/)

ament_export_include_directories(include/ground_station)
ament_export_dependencies(
  rclcpp OpenCV sensor_msgs nav_msgs geometry_msgs std_msgs yaml-cpp Eigen3 tf2 tf2_ros tf2_geometry_msgs
)
ament_package()


