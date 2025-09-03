#include <chrono>
#include <cmath>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

using namespace std::chrono_literals;

class SimulationReference : public rclcpp::Node {
public:
  SimulationReference()
  : rclcpp::Node("simulation_reference"),
    start_time_(std::chrono::steady_clock::now()) {
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/reference/trajectory", 10);

    // Publish initial path for [0, 3] sec
    publish_path_horizon(0.0, 3.0);

    // Timer to publish at fixed rate (100 Hz)
    timer_ = this->create_wall_timer(10ms, std::bind(&SimulationReference::on_timer, this));
  }

private:
  void on_timer() {
    const auto now = std::chrono::steady_clock::now();
    const double t = std::chrono::duration<double>(now - start_time_).count();
    publish_path_horizon(t, 3.0);
  }

  void publish_path_horizon(double t0, double horizon_sec) {
    constexpr double pi = 3.14159265358979323846;
    const double dt = 0.10;  // 10 Hz sampling for path
    const int steps = static_cast<int>(horizon_sec / dt) + 1;

    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = this->now();
    path_msg.header.frame_id = "map";  // adjust frame as needed

    // Circle parameters: radius 10 m, constant speed 0.5 m/s
    const double R = 1.0;
    const double v = 0.2;
    const double omega = v / R;               // rad/s
    const double cx = 0.0;                    // center x
    const double cy = -1.0;                  // center y
    const double theta0 = pi / 2.0;           // start at (0,0)

    path_msg.poses.reserve(steps);
    for (int k = 0; k < steps; ++k) {
      const double tk = t0 + k * dt;
      const double theta = theta0 - omega * tk;   // CW rotation
      const double x = cx + R * std::cos(theta);
      const double y = cy + R * std::sin(theta);
      const double yaw = theta - pi / 2.0; 

      geometry_msgs::msg::PoseStamped ps;
      ps.header.stamp = path_msg.header.stamp;
      ps.header.frame_id = path_msg.header.frame_id;
      ps.pose.position.x = x;
      ps.pose.position.y = y;
      ps.pose.position.z = 0.0;
      // Yaw -> quaternion (z, w for yaw-only rotation)
      const double half = 0.5 * yaw;
      ps.pose.orientation.x = 0.0;
      ps.pose.orientation.y = 0.0;
      ps.pose.orientation.z = std::sin(half);
      ps.pose.orientation.w = std::cos(half);

      path_msg.poses.push_back(std::move(ps));
    }

    path_pub_->publish(path_msg);
  }

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::chrono::steady_clock::time_point start_time_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimulationReference>());
  rclcpp::shutdown();
  return 0;
}
