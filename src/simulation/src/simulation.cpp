#include <chrono>
#include <cmath>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"

using namespace std::chrono_literals;

class KinematicsSimulation : public rclcpp::Node {
public:
  KinematicsSimulation()
  : rclcpp::Node("kinematics_simulation") {
    // Parameters (align with controller_node)
    this->declare_parameter<double>("dt", 0.0025);               // 400 Hz
    this->declare_parameter<double>("sys.robo_rdi", 0.10);
    this->declare_parameter<double>("sys.robo_dst", 0.50);
    this->declare_parameter<int>("sys.n_rbt", 4);
    this->declare_parameter<std::vector<double>>("sys.r_BtoR",
      {1.02, -0.30, 1.02, 0.20, 0.20, -0.515, -0.60, -0.515});

    this->declare_parameter<double>("init.xB", 0.0);
    this->declare_parameter<double>("init.yB", 0.0);
    this->declare_parameter<double>("init.thB", 0.0);
    this->declare_parameter<std::vector<double>>("init.thR",
      {0.0, 0.0, 0.0, 0.0});

    // Frame config
    this->declare_parameter<std::string>("frame.cart_parent", "world");
    this->declare_parameter<std::string>("frame.robot_parent", "odom");
    this->declare_parameter<std::string>("frame.cart", "base_link");
    this->declare_parameter<std::string>("frame.robot_prefix", "robot");
    this->declare_parameter<std::string>("frame.robot_suffix", "_base_2");


    dt_ = this->get_parameter("dt").as_double();
    robo_rdi_ = this->get_parameter("sys.robo_rdi").as_double();
    robo_dst_ = this->get_parameter("sys.robo_dst").as_double();
    n_rbt_ = this->get_parameter("sys.n_rbt").as_int();
    auto rBtoR_flat = this->get_parameter("sys.r_BtoR").as_double_array();
    r_BtoR_.resize(n_rbt_);
    for (int i = 0; i < n_rbt_; ++i) {
      r_BtoR_[i].first = rBtoR_flat[2*i];
      r_BtoR_[i].second = rBtoR_flat[2*i+1];
    }

    xB_ = this->get_parameter("init.xB").as_double();
    yB_ = this->get_parameter("init.yB").as_double();
    thB_ = this->get_parameter("init.thB").as_double();
    auto thR_init = this->get_parameter("init.thR").as_double_array();
    thR_.assign(thR_init.begin(), thR_init.end());
    if ((int)thR_.size() < n_rbt_) thR_.resize(n_rbt_, 0.0);

    frame_cart_parent_ = this->get_parameter("frame.cart_parent").as_string();
    frame_robot_parent_ = this->get_parameter("frame.robot_parent").as_string();
    frame_cart_ = this->get_parameter("frame.cart").as_string();
    frame_robot_prefix_ = this->get_parameter("frame.robot_prefix").as_string();
    frame_robot_suffix_ = this->get_parameter("frame.robot_suffix").as_string();

    // Subscriber for pseudo input Up = [xM,yM,omM,omR1..omR4]
    sub_up_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/mpc/pseudo_input", 10,
      [this](std_msgs::msg::Float64MultiArray::SharedPtr msg){ up_ = *msg; has_up_ = true; }
    );

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    timer_ = this->create_wall_timer(std::chrono::duration<double>(dt_), std::bind(&KinematicsSimulation::on_tick, this));
  }

private:
  void on_tick() {
    if (!has_up_) {
      // publish current TFs so that controller can start
      publish_cart_tf();
      publish_robot_tfs();
      return;
    }
    // Parse Up
    if (up_.data.size() < 7) return;
    const double xM = up_.data[0];
    const double yM = up_.data[1];
    const double omM = up_.data[2];
    std::vector<double> omR(n_rbt_, 0.0);
    for (int i = 0; i < std::min(n_rbt_, (int)up_.data.size() - 3); ++i) {
      omR[i] = up_.data[3 + i];
    }

    // 2) Choose pseudo center M relative to cart (body x-axis offset)
    const double c = std::cos(thB_);
    const double s = std::sin(thB_);

    // omM, xM, yM are from Up directly (MATLAB-consistent)

    // 4) Dynamics (same as controller model)
    const double rMx = xB_ - xM;
    const double rMy = yB_ - yM;
    const double vBx = -omM * rMy;
    const double vBy =  omM * rMx;
    const double thB_dot = omM;

    // 5) Integrate (Euler)
    xB_ += dt_ * vBx;
    yB_ += dt_ * vBy;
    thB_ += dt_ * thB_dot;
    for (int i = 0; i < n_rbt_; ++i) thR_[i] += dt_ * omR[i];

    // 6) Publish TFs
    publish_cart_tf();
    publish_robot_tfs();
  }

  void publish_cart_tf() {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = this->now();
    t.header.frame_id = frame_cart_parent_;
    t.child_frame_id = frame_cart_;
    t.transform.translation.x = xB_;
    t.transform.translation.y = yB_;
    t.transform.translation.z = 0.0;
    const double half = 0.5 * thB_;
    t.transform.rotation.x = 0.0;
    t.transform.rotation.y = 0.0;
    t.transform.rotation.z = std::sin(half);
    t.transform.rotation.w = std::cos(half);
    tf_broadcaster_->sendTransform(t);
  }

  void publish_robot_tfs() {
    const double c = std::cos(thB_);
    const double s = std::sin(thB_);
    for (int i = 0; i < n_rbt_; ++i) {
      const double rx = c * r_BtoR_[i].first - s * r_BtoR_[i].second;
      const double ry = s * r_BtoR_[i].first + c * r_BtoR_[i].second;
      geometry_msgs::msg::TransformStamped t;
      t.header.stamp = this->now();
      t.header.frame_id = frame_robot_parent_;
      t.child_frame_id = frame_robot_prefix_ + std::to_string(i+1) + frame_robot_suffix_;
      t.transform.translation.x = xB_ + rx;
      t.transform.translation.y = yB_ + ry;
      t.transform.translation.z = 0.0;
      const double half = 0.5 * thR_[i];
      t.transform.rotation.x = 0.0;
      t.transform.rotation.y = 0.0;
      t.transform.rotation.z = std::sin(half);
      t.transform.rotation.w = std::cos(half);
      tf_broadcaster_->sendTransform(t);
    }
  }

  // Params
  double dt_{};
  double robo_rdi_{};
  double robo_dst_{};
  int n_rbt_{};
  std::vector<std::pair<double,double>> r_BtoR_;

  // State
  double xB_{};
  double yB_{};
  double thB_{};
  std::vector<double> thR_;

  // Frames
  std::string frame_cart_parent_;
  std::string frame_robot_parent_;
  std::string frame_cart_;
  std::string frame_robot_prefix_;
  std::string frame_robot_suffix_;
  // removed: pseudo center offset parameter

  // I/O
  std_msgs::msg::Float64MultiArray up_;
  bool has_up_ = false;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_up_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<KinematicsSimulation>());
  rclcpp::shutdown();
  return 0;
}


