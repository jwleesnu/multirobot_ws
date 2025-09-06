#pragma once

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float64.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>


namespace global_mapper {

class GlobalMapperNode : public rclcpp::Node {
public:
  GlobalMapperNode();
  
private:

  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr global_map_publisher_;
  // Callback functions for subscriptions
  
  // Timer for periodic tasks
  rclcpp::TimerBase::SharedPtr global_map_timer_;
  void global_map_timer();
  
  nav_msgs::msg::OccupancyGrid::SharedPtr global_map_;

  bool use_prior_map_;
  double object_size_x_;
  double object_size_y_;
  double global_map_timer_period_;
  std::string map_path_; // Path to the prior map defined in YAML file
  void load_map(const std::string & map_path);

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr tmp_coord_pub_;

  bool reload_from_yaml(const std::string & yaml_path) {
    try {
      YAML::Node config = YAML::LoadFile(yaml_path);

      // Get directory of the yaml file
      std::string yaml_dir = yaml_path.substr(0, yaml_path.find_last_of("/\\") + 1);

      // 예시: map_path, resolution, origin_x, origin_y, png_path 등 읽기
      if (config["resolution"]) {
        this->declare_parameter("resolution", config["resolution"].as<double>());
      }
      if (config["origin_x"]) {
        this->declare_parameter("origin_x", config["origin_x"].as<double>());
      }
      if (config["origin_y"]) {
        this->declare_parameter("origin_y", config["origin_y"].as<double>());
      }
      if (config["png_name"]) {
        this->declare_parameter("png_path", yaml_dir + config["png_name"].as<std::string>());
      }
      return true;
    } catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "YAML parse failed: %s", e.what());
      return false;
    }
  }
};

} // namespace global_mapper