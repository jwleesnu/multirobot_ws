#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <stdexcept>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

#include <geometry_msgs/msg/pose2_d.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>


#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/time.h>

#include <casadi/casadi.hpp>

// Namespace shortcuts
using std::placeholders::_1;
using casadi::MX;
using SX = casadi::MX;
using casadi::DM;
using casadi::Function;
using casadi::Slice;
namespace ca = casadi;


class CasadiMPCControllerNode : public rclcpp::Node {
public:
  CasadiMPCControllerNode() : rclcpp::Node("casadi_mpc_controller_node") {
    // Parameters (mirror Python)
    this->declare_parameter<double>("control_rate_hz", 10.0);
    this->declare_parameter<double>("con.t_delta", 0.1);
    this->declare_parameter<int>("con.n_hor", 20);
    this->declare_parameter<double>("con.arg_bnd", 1e3);

    this->declare_parameter<double>("con.Q_err_trn_x", 100.0);
    this->declare_parameter<double>("con.Q_err_trn_y", 100.0);
    this->declare_parameter<double>("con.Q_err_ang", 1000.0);
    this->declare_parameter<double>("con.Q_hdg", 2000.0);
    this->declare_parameter<double>("con.Q_con", 1e-6);
    this->declare_parameter<double>("con.Q_chg", 1e-1);

    this->declare_parameter<int>("sys.n_rbt", 4);
    this->declare_parameter<double>("sys.cart_hgt", 1.03);
    this->declare_parameter<double>("sys.cart_wdt", 2.04);
    this->declare_parameter<double>("sys.robo_sze", 0.30);
    this->declare_parameter<double>("sys.robo_dst", 0.50);
    this->declare_parameter<double>("sys.robo_rdi", 0.10);

    std::vector<double> default_r_BtoR = {
      1.02, -0.30,
      1.02,  0.20,
      0.20, -0.515,
     -0.60, -0.515,
    };
    this->declare_parameter<std::vector<double>>("sys.r_BtoR", default_r_BtoR);
    this->declare_parameter<double>("sys.u_lower", -20.0);
    this->declare_parameter<double>("sys.u_upper",  20.0);

    this->declare_parameter<std::string>("frame.map", "odom");
    this->declare_parameter<std::string>("frame.cart", "base_link");
    this->declare_parameter<std::string>("frame.cart_parent", "odom");
    this->declare_parameter<std::string>("frame.robot_parent", "odom");
    this->declare_parameter<std::string>("frame.robot_prefix", "robot");
    this->declare_parameter<std::string>("frame.robot_suffix", "_base_2");

    control_rate_hz_ = this->get_parameter("control_rate_hz").as_double();

    params_con_t_delta_ = this->get_parameter("con.t_delta").as_double();
    params_con_n_hor_ = this->get_parameter("con.n_hor").as_int();
    params_con_arg_bnd_ = this->get_parameter("con.arg_bnd").as_double();
    params_con_Q_err_trn_x_ = this->get_parameter("con.Q_err_trn_x").as_double();
    params_con_Q_err_trn_y_ = this->get_parameter("con.Q_err_trn_y").as_double();
    params_con_Q_err_ang_ = this->get_parameter("con.Q_err_ang").as_double();
    params_con_Q_hdg_ = this->get_parameter("con.Q_hdg").as_double();
    params_con_Q_con_ = this->get_parameter("con.Q_con").as_double();
    params_con_Q_chg_ = this->get_parameter("con.Q_chg").as_double();

    params_sys_n_rbt_ = this->get_parameter("sys.n_rbt").as_int();
    params_sys_cart_hgt_ = this->get_parameter("sys.cart_hgt").as_double();
    params_sys_cart_wdt_ = this->get_parameter("sys.cart_wdt").as_double();
    params_sys_robo_sze_ = this->get_parameter("sys.robo_sze").as_double();
    params_sys_robo_dst_ = this->get_parameter("sys.robo_dst").as_double();
    params_sys_robo_rdi_ = this->get_parameter("sys.robo_rdi").as_double();
    params_sys_r_BtoR_pairs_ = param_list_to_pairs(
      this->get_parameter("sys.r_BtoR").as_double_array(), params_sys_n_rbt_
    );
    params_sys_u_lower_ = this->get_parameter("sys.u_lower").as_double();
    params_sys_u_upper_ = this->get_parameter("sys.u_upper").as_double();

    frame_map_ = this->get_parameter("frame.map").as_string();
    frame_cart_ = this->get_parameter("frame.cart").as_string();
    frame_cart_parent_ = this->get_parameter("frame.cart_parent").as_string();
    frame_robot_parent_ = this->get_parameter("frame.robot_parent").as_string();
    frame_robot_prefix_ = this->get_parameter("frame.robot_prefix").as_string();
    frame_robot_suffix_ = this->get_parameter("frame.robot_suffix").as_string();

    // ROS I/O
    sub_path_ = this->create_subscription<nav_msgs::msg::Path>(
      "/reference/trajectory", rclcpp::QoS(10), std::bind(&CasadiMPCControllerNode::on_path, this, _1)
    );

    // Subscribe towing angles: /robot_i/towing_angle (degrees 0..360)
    towing_deg_.assign(params_sys_n_rbt_, 0.0);
    has_towing_angle_.assign(params_sys_n_rbt_, false);
    for (int i = 0; i < params_sys_n_rbt_; ++i) {
      std::string topic = "/robot_" + std::to_string(i+1) + "/towing_angle";
      sub_towing_.push_back(
        this->create_subscription<std_msgs::msg::Float64>(
          topic, 10,
          [this, i](const std_msgs::msg::Float64::SharedPtr msg) {
            towing_deg_[i] = msg->data;
            has_towing_angle_[i] = true;
          }
        )
      );
    }


    for (int i = 0; i < params_sys_n_rbt_; ++i) {
      std::string topic = "/multirobot/robot" + std::to_string(i+1) + "/cmd_vel";
      pub_cmd_vel_.push_back(this->create_publisher<geometry_msgs::msg::Twist>(topic, 10));
    }
    pub_cart_pose_ = this->create_publisher<geometry_msgs::msg::Pose2D>("/cart/pose", 10);
    pub_ref_x_ = this->create_publisher<std_msgs::msg::Float64>("/mpc/xref", 10);
    pub_ref_y_ = this->create_publisher<std_msgs::msg::Float64>("/mpc/yref", 10);
    pub_ref_th_ = this->create_publisher<std_msgs::msg::Float64>("/mpc/thref", 10);
    for (int i = 0; i < params_sys_n_rbt_; ++i) {
      pub_ul_dbg_.push_back(this->create_publisher<std_msgs::msg::Float64>("/debug/robot" + std::to_string(i+1) + "/ul", 10));
      pub_ur_dbg_.push_back(this->create_publisher<std_msgs::msg::Float64>("/debug/robot" + std::to_string(i+1) + "/ur", 10));
      pub_om_dbg_.push_back(this->create_publisher<std_msgs::msg::Float64>("/debug/robot" + std::to_string(i+1) + "/omega", 10));
    }
    pub_up_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/mpc/pseudo_input", 10);

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    RCLCPP_INFO(this->get_logger(), "Building CasADi NLP (IPOPT)...");
    build_nlp();

    x_prev_valid_ = false;
    last_U_.assign(nlp_NU_, 0.0);
    thR_.assign(params_sys_n_rbt_, 0.0);

    warn_ref_printed_ = false;
    warn_pose_printed_ = false;

    using namespace std::chrono_literals;
    timer_ = this->create_wall_timer(std::chrono::duration<double>(1.0 / control_rate_hz_), std::bind(&CasadiMPCControllerNode::control_tick, this));
    RCLCPP_INFO(this->get_logger(), "CasadiMPCControllerNode started.");
  }

private:
  // Parameters
  double control_rate_hz_{};
  double params_con_t_delta_{};
  int params_con_n_hor_{};
  double params_con_arg_bnd_{};
  double params_con_Q_err_trn_x_{};
  double params_con_Q_err_trn_y_{};
  double params_con_Q_err_ang_{};
  double params_con_Q_hdg_{};
  double params_con_Q_con_{};
  double params_con_Q_chg_{};

  int params_sys_n_rbt_{};
  double params_sys_cart_hgt_{};
  double params_sys_cart_wdt_{};
  double params_sys_robo_sze_{};
  double params_sys_robo_dst_{};
  double params_sys_robo_rdi_{};
  std::vector<std::pair<double,double>> params_sys_r_BtoR_pairs_;
  double params_sys_u_lower_{};
  double params_sys_u_upper_{};

  std::string frame_map_;
  std::string frame_cart_;
  std::string frame_cart_parent_;
  std::string frame_robot_parent_; // unused
  std::string frame_robot_prefix_; // unused
  std::string frame_robot_suffix_; // unused

  // ROS
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_path_;
  std::vector<rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr> pub_cmd_vel_;
  rclcpp::Publisher<geometry_msgs::msg::Pose2D>::SharedPtr pub_cart_pose_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_ref_x_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_ref_y_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_ref_th_;
  std::vector<rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr> pub_ul_dbg_;
  std::vector<rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr> pub_ur_dbg_;
  std::vector<rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr> pub_om_dbg_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_up_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::TimerBase::SharedPtr timer_;

  // CasADi NLP
  Function solver_init_;
  Function solver_warm_;
  int nlp_N_{};
  int nlp_M_{};
  int nlp_NX_{};
  int nlp_NUP_{};
  int nlp_NU_{};
  double nlp_dt_{};
  // column lengths
  int len_X_col_{};
  int len_Up_col_{};
  int len_U_col_{};
  int len_P_X_curr_col_{};
  int len_P_Xb_desr_col_{};
  int len_P_U_last_col_{};
  int len_G_Init_col_{};
  int len_G_Dyna_col_{};
  int len_G_Cont_col_{};

  // runtime buffers
  nav_msgs::msg::Path::SharedPtr path_cache_;
  double xB_{0.0}, yB_{0.0}, thB_{0.0};
  std::vector<double> thR_;
  std::vector<double> last_U_;

  // Towing angle inputs (degrees) and subscriptions
  std::vector<double> towing_deg_;
  std::vector<bool> has_towing_angle_;
  std::vector<rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr> sub_towing_;

  bool x_prev_valid_{};
  DM x_prev_;
  DM lam_x_prev_;
  DM lam_g_prev_;

  bool warn_ref_printed_{};
  bool warn_pose_printed_{};

private:
  static std::vector<std::pair<double,double>> param_list_to_pairs(const std::vector<double>& lst, int expected_pairs) {
    if ((int)lst.size() != 2*expected_pairs) {
      throw std::runtime_error("sys.r_BtoR must have length " + std::to_string(2*expected_pairs));
    }
    std::vector<std::pair<double,double>> out;
    out.reserve(expected_pairs);
    for (int i = 0; i < expected_pairs; ++i) {
      out.emplace_back(lst[2*i], lst[2*i+1]);
    }
    return out;
  }

  void on_path(const nav_msgs::msg::Path::SharedPtr msg) {
    path_cache_ = msg;
  }


  std::tuple<double,double,double> lookup_pose_yaw(const std::string& target_frame, const std::string& parent_frame) {
    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_->lookupTransform(parent_frame, target_frame, tf2::TimePointZero, tf2::durationFromSec(0.05));
    } catch (const std::exception& e) {
      throw;
    }
    const auto& t = tf.transform.translation;
    const auto& q = tf.transform.rotation;
    double yaw = std::atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z));
    return {t.x, t.y, yaw};
  }

  void build_nlp() {
    const int N = params_con_n_hor_;
    const int M = params_sys_n_rbt_;
    const int NX = 3 + M;
    const int NUP = 3 + M;
    const int NU = 2 * M;
    const double dt = params_con_t_delta_;

    // Decision variables
    SX X = MX::sym("X", NX, N+1);
    SX Up = MX::sym("Up", NUP, N);
    SX U = MX::sym("U", NU, N);

    // Parameters
    SX P_X_curr = MX::sym("P_X_curr", NX, 1);
    SX P_Xb_desr = MX::sym("P_Xb_desr", 3, N);
    SX P_U_last = MX::sym("P_U_last", NU, 1);

    auto rot2_mul_vec = [](const SX& th, const SX& v3) {
      SX c = MX::cos(th);
      SX s = MX::sin(th);
      SX x = v3(0);
      SX y = v3(1);
      SX z = v3(2);
      SX rx = c * x - s * y;
      SX ry = s * x + c * y;
      std::vector<SX> parts = {rx, ry, z};
      return MX::vertcat(parts);
    };

    auto current_to_next = [&](const SX& x, const SX& up) {
      SX posiB = MX::vertcat(std::vector<SX>{x(0), x(1), SX(0)});
      SX thetaB = x(2);
      SX thetaR = x(Slice(3, 3+M));
      SX posiM = MX::vertcat(std::vector<SX>{up(0), up(1), SX(0)});
      SX omegaM = MX::vertcat(std::vector<SX>{SX(0), SX(0), up(2)});
      SX r_MtoB = posiB - posiM;
      SX v = MX::vertcat(std::vector<SX>{
        omegaM(1)*r_MtoB(2) - omegaM(2)*r_MtoB(1),
        omegaM(2)*r_MtoB(0) - omegaM(0)*r_MtoB(2),
        omegaM(0)*r_MtoB(1) - omegaM(1)*r_MtoB(0)
      });
      SX posiB_next = posiB + v*dt;
      SX thetaB_next = thetaB + omegaM(2)*dt;
      SX thetaR_next = thetaR + up(Slice(3, 3+M))*dt;
      return MX::vertcat(std::vector<SX>{posiB_next(0), posiB_next(1), thetaB_next, thetaR_next});
    };

    auto pseudo_to_actual_inputs = [&](const SX& x, const SX& up) {
      SX posiB = MX::vertcat(std::vector<SX>{x(0), x(1), SX(0)});
      SX thetaB = x(2);
      SX posiM = MX::vertcat(std::vector<SX>{up(0), up(1), SX(0)});
      SX omegaM = MX::vertcat(std::vector<SX>{SX(0), SX(0), up(2)});
      SX r_MtoB = posiB - posiM;
      std::vector<SX> u_list;
      u_list.reserve(2*M);
      for (int i = 0; i < M; ++i) {
        SX rBtoR = MX::vertcat(std::vector<SX>{SX(params_sys_r_BtoR_pairs_[i].first), SX(params_sys_r_BtoR_pairs_[i].second), SX(0)});
        SX r_MtoR = r_MtoB + rot2_mul_vec(thetaB, rBtoR);
        SX v = MX::vertcat(std::vector<SX>{
          omegaM(1)*r_MtoR(2) - omegaM(2)*r_MtoR(1),
          omegaM(2)*r_MtoR(0) - omegaM(0)*r_MtoR(2),
          omegaM(0)*r_MtoR(1) - omegaM(1)*r_MtoR(0)
        });
        SX vnorm = MX::sqrt(v(0)*v(0) + v(1)*v(1) + v(2)*v(2));
        SX omegaR_i = up(3 + i);
        SX uL = (vnorm - 0.5*omegaR_i*params_sys_robo_dst_) / params_sys_robo_rdi_;
        SX uR = (vnorm + 0.5*omegaR_i*params_sys_robo_dst_) / params_sys_robo_rdi_;
        u_list.push_back(uL);
        u_list.push_back(uR);
      }
      return MX::vertcat(u_list);
    };

    auto heading_cost = [&](const SX& x, const SX& up) {
      SX posiB = MX::vertcat(std::vector<SX>{x(0), x(1), SX(0)});
      SX thetaB = x(2);
      SX posiM = MX::vertcat(std::vector<SX>{up(0), up(1), SX(0)});
      SX omegaM = MX::vertcat(std::vector<SX>{SX(0), SX(0), up(2)});
      SX r_MtoB = posiB - posiM;
      SX total = SX(0);
      for (int i = 0; i < M; ++i) {
        SX rBtoR = MX::vertcat(std::vector<SX>{SX(params_sys_r_BtoR_pairs_[i].first), SX(params_sys_r_BtoR_pairs_[i].second), SX(0)});
        SX r_MtoR = r_MtoB + rot2_mul_vec(thetaB, rBtoR);
        SX v_pseudo = MX::vertcat(std::vector<SX>{
          omegaM(1)*r_MtoR(2) - omegaM(2)*r_MtoR(1),
          omegaM(2)*r_MtoR(0) - omegaM(0)*r_MtoR(2),
          omegaM(0)*r_MtoR(1) - omegaM(1)*r_MtoR(0)
        });
        SX vnorm = MX::sqrt(v_pseudo(0)*v_pseudo(0) + v_pseudo(1)*v_pseudo(1) + v_pseudo(2)*v_pseudo(2));
        SX v_actual = MX::vertcat(std::vector<SX>{vnorm*MX::cos(x(3+i)), vnorm*MX::sin(x(3+i)), SX(0)});
        SX dvx = v_actual(0) - v_pseudo(0);
        SX dvy = v_actual(1) - v_pseudo(1);
        total = total + dvx*dvx + dvy*dvy;
      }
      return total;
    };

    // Cost
    SX F_cost = MX(0);
    for (int i = 0; i < N; ++i) {
      SX e_pos = X(Slice(0,2+1), i+1) - P_Xb_desr(Slice(0,2+1), i);
      SX ex = e_pos(0);
      SX ey = e_pos(1);
      F_cost = F_cost + (params_con_Q_err_trn_x_ * ex * ex + params_con_Q_err_trn_y_ * ey * ey);
      SX c1 = MX::cos(X(2, i+1)) - MX::cos(P_Xb_desr(2, i));
      SX s1 = MX::sin(X(2, i+1)) - MX::sin(P_Xb_desr(2, i));
      F_cost = F_cost + params_con_Q_err_ang_ * (c1*c1 + s1*s1);
      F_cost = F_cost + params_con_Q_hdg_ * heading_cost(X(Slice(), i), Up(Slice(), i));
      SX dot_ui0 = 0;
      for (int rr = 0; rr < NU; ++rr) dot_ui0 = dot_ui0 + U(rr,0) * U(rr,i);
      F_cost = F_cost + params_con_Q_con_ * dot_ui0;
      SX du;
      if (i == 0) {
        du = (U(Slice(), i) - P_U_last) / dt;
      } else {
        du = (U(Slice(), i) - U(Slice(), i-1)) / dt;
      }
      SX du_norm2 = 0;
      for (int rr = 0; rr < NU; ++rr) du_norm2 = du_norm2 + du(rr,0) * du(rr,0);
      F_cost = F_cost + params_con_Q_chg_ * du_norm2;
    }

    // Constraints
    std::vector<SX> G_list;
    G_list.push_back(X(Slice(), 0) - P_X_curr);
    for (int i = 0; i < N; ++i) {
      G_list.push_back(current_to_next(X(Slice(), i), Up(Slice(), i)) - X(Slice(), i+1));
      G_list.push_back(pseudo_to_actual_inputs(X(Slice(), i), Up(Slice(), i)) - U(Slice(), i));
    }
    SX g = MX::vertcat(G_list);

    // Decision and parameter vectors
    std::vector<SX> x_parts = {MX::reshape(X, -1, 1), MX::reshape(Up, -1, 1), MX::reshape(U, -1, 1)};
    SX x_vec = MX::vertcat(x_parts);
    std::vector<SX> p_parts = {MX::reshape(P_X_curr, -1, 1), MX::reshape(P_Xb_desr, -1, 1), MX::reshape(P_U_last, -1, 1)};
    SX p_vec = MX::vertcat(p_parts);

    casadi::MXDict nlp;
    nlp["x"] = x_vec;
    nlp["f"] = F_cost;
    nlp["g"] = g;
    nlp["p"] = p_vec;

    casadi::Dict ipopt_init;
    ipopt_init["print_time"] = false;
    ipopt_init["ipopt.max_iter"] = 3000;
    ipopt_init["ipopt.tol"] = 1e-6;
    ipopt_init["ipopt.print_level"] = 3;
    ipopt_init["ipopt.warm_start_init_point"] = std::string("no");

    casadi::Dict ipopt_warm;
    ipopt_warm["print_time"] = false;
    ipopt_warm["ipopt.max_iter"] = 3000;
    ipopt_warm["ipopt.tol"] = 1e-6;
    ipopt_warm["ipopt.print_level"] = 3;
    ipopt_warm["ipopt.warm_start_init_point"] = std::string("yes");
    ipopt_warm["ipopt.warm_start_bound_frac"] = 1e-16;
    ipopt_warm["ipopt.warm_start_bound_push"] = 1e-16;
    ipopt_warm["ipopt.warm_start_mult_bound_push"] = 1e-16;
    ipopt_warm["ipopt.warm_start_slack_bound_frac"] = 1e-16;
    ipopt_warm["ipopt.warm_start_slack_bound_push"] = 1e-16;
    ipopt_warm["ipopt.max_cpu_time"] = 0.5 * dt;

    solver_init_ = casadi::nlpsol("npp_solver_init", "ipopt", nlp, ipopt_init);
    solver_warm_ = casadi::nlpsol("npp_solver_warm", "ipopt", nlp, ipopt_warm);

    // sizes
    nlp_N_ = N;
    nlp_M_ = M;
    nlp_NX_ = NX;
    nlp_NUP_ = NUP;
    nlp_NU_ = NU;
    nlp_dt_ = dt;
    len_X_col_ = (N + 1) * NX;
    len_Up_col_ = N * NUP;
    len_U_col_ = N * NU;
    len_P_X_curr_col_ = NX;
    len_P_Xb_desr_col_ = 3 * N;
    len_P_U_last_col_ = NU;
    len_G_Init_col_ = NX;
    len_G_Dyna_col_ = N * NX;
    len_G_Cont_col_ = N * NU;
  }

  static double quat_to_yaw(double x, double y, double z, double w) {
    return std::atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z));
  }

  void control_tick() {
    // Reference readiness
    if (!path_cache_ || path_cache_->poses.empty()) {
      if (!warn_ref_printed_) {
        RCLCPP_INFO(this->get_logger(), "Waiting for /reference/trajectory...");
        warn_ref_printed_ = true;
      }
      return;
    }

    // 1) TF lookup for cart pose
    try {
      auto [xB, yB, thB] = lookup_pose_yaw(frame_cart_, frame_cart_parent_);
      xB_ = xB; yB_ = yB; thB_ = thB;
    } catch (...) {
      if (!warn_pose_printed_) {
        RCLCPP_INFO(this->get_logger(), "Waiting for TF (cart pose %s->%s)...", frame_cart_parent_.c_str(), frame_cart_.c_str());
        warn_pose_printed_ = true;
      }
      return;
    }

    // 2) Robot headings from TF (world -> robot_i)
    try {
      std::vector<double> thR_list(params_sys_n_rbt_, 0.0);
      for (int i = 0; i < params_sys_n_rbt_; ++i) {
        std::string robot_frame = "robot_" + std::to_string(i+1);
        auto [rx, ry, yaw_i] = lookup_pose_yaw(robot_frame, "world");
        (void)rx; (void)ry;
        thR_list[i] = yaw_i;
      }
      thR_ = thR_list;
    } catch (...) {
      RCLCPP_INFO_THROTTLE(this->get_logger(), *(this->get_clock()), 1000,
        "Waiting for TFs (world->robot_i)...");
      return;
    }

    const int N = nlp_N_;
    const int NX = nlp_NX_;
    const int NU = nlp_NU_;

    // Build reference arrays
    std::vector<double> xref(N, 0.0), yref(N, 0.0), thref(N, 0.0);
    const auto& poses = path_cache_->poses;
    for (int k = 0; k < N; ++k) {
      int idx = k < (int)poses.size() ? k : (int)poses.size() - 1;
      const auto& pose = poses[idx].pose;
      xref[k] = pose.position.x;
      yref[k] = pose.position.y;
      double qw = pose.orientation.w;
      double qz = pose.orientation.z;
      thref[k] = std::atan2(2.0*qz*qw, 1.0 - 2.0*(qz*qz));
    }

    // Parameter vector P = [X_curr; Xb_desr(3N); U_last]
    std::vector<double> P(NX + 3*N + NU, 0.0);
    // X_curr
    P[0] = xB_; P[1] = yB_; P[2] = thB_;
    for (int i = 0; i < params_sys_n_rbt_; ++i) P[3+i] = thR_[i];
    // Xb_desr stacked
    int base = NX;
    for (int i = 0; i < N; ++i) {
      P[base + 3*i + 0] = xref[i];
      P[base + 3*i + 1] = yref[i];
      P[base + 3*i + 2] = thref[i];
    }
    base = NX + 3*N;
    for (int i = 0; i < NU; ++i) P[base + i] = last_U_[i];

    // Bounds on variables
    double arg_bnd = params_con_arg_bnd_;
    std::vector<double> lbx_x(len_X_col_, -arg_bnd), ubx_x(len_X_col_, arg_bnd);
    std::vector<double> lbx_up(len_Up_col_, -arg_bnd), ubx_up(len_Up_col_, arg_bnd);
    std::vector<double> lbx_u(len_U_col_, params_sys_u_lower_), ubx_u(len_U_col_, params_sys_u_upper_);
    std::vector<double> lbx; lbx.reserve(len_X_col_ + len_Up_col_ + len_U_col_);
    std::vector<double> ubx; ubx.reserve(lbx.capacity());
    lbx.insert(lbx.end(), lbx_x.begin(), lbx_x.end());
    lbx.insert(lbx.end(), lbx_up.begin(), lbx_up.end());
    lbx.insert(lbx.end(), lbx_u.begin(), lbx_u.end());
    ubx.insert(ubx.end(), ubx_x.begin(), ubx_x.end());
    ubx.insert(ubx.end(), ubx_up.begin(), ubx_up.end());
    ubx.insert(ubx.end(), ubx_u.begin(), ubx_u.end());

    // Bounds on constraints (equalities)
    int n_g = len_G_Init_col_ + len_G_Dyna_col_ + len_G_Cont_col_;
    std::vector<double> lbg(n_g, 0.0), ubg(n_g, 0.0);

    // Initial guess
    casadi::DMDict arg;
    if (!x_prev_valid_) {
      // X0 tiled with current state, Up0 small omegas, U0 zeros
      std::vector<double> x0;
      x0.reserve(len_X_col_ + len_Up_col_ + len_U_col_);
      {
        // X0 (column-major: NX x (N+1))
        std::vector<double> state0;
        state0.reserve(3 + params_sys_n_rbt_);
        state0.push_back(xB_);
        state0.push_back(yB_);
        state0.push_back(thB_);
        for (int i = 0; i < params_sys_n_rbt_; ++i) state0.push_back(thR_[i]);
        for (int col = 0; col < N+1; ++col) x0.insert(x0.end(), state0.begin(), state0.end());
      }
      {
        // Up0 (NUP x N), seed with [xB,yB,1e-3, zeros(M)]
        std::vector<double> up0;
        up0.reserve(3 + params_sys_n_rbt_);
        up0.push_back(xB_);
        up0.push_back(yB_);
        up0.push_back(1e-3);
        for (int i = 0; i < params_sys_n_rbt_; ++i) up0.push_back(0.0);
        for (int col = 0; col < N; ++col) x0.insert(x0.end(), up0.begin(), up0.end());
      }
      x0.insert(x0.end(), len_U_col_, 0.0);
      arg["x0"] = DM(x0);
      arg["p"] = DM(P);
      arg["lbx"] = DM(lbx);
      arg["ubx"] = DM(ubx);
      arg["lbg"] = DM(lbg);
      arg["ubg"] = DM(ubg);
    } else {
      arg["x0"] = x_prev_;
      arg["p"] = DM(P);
      arg["lbx"] = DM(lbx);
      arg["ubx"] = DM(ubx);
      arg["lbg"] = DM(lbg);
      arg["ubg"] = DM(ubg);
      arg["lam_x0"] = lam_x_prev_.is_empty() ? DM::zeros(x_prev_.size1()*x_prev_.size2()) : lam_x_prev_;
      arg["lam_g0"] = lam_g_prev_.is_empty() ? DM::zeros(n_g) : lam_g_prev_;
    }

    casadi::DMDict res;
    try {
      if (!x_prev_valid_) res = solver_init_(arg);
      else res = solver_warm_(arg);
    } catch (const std::exception& e) {
      RCLCPP_WARN(this->get_logger(), "IPOPT failed: %s", e.what());
      return;
    }

    DM x_opt_dm = res.at("x");
    std::vector<double> x_opt = x_opt_dm.nonzeros();
    if (res.find("lam_x") != res.end()) lam_x_prev_ = res.at("lam_x");
    if (res.find("lam_g") != res.end()) lam_g_prev_ = res.at("lam_g");
    x_prev_ = x_opt_dm;
    x_prev_valid_ = true;

    // Unstack Up(:,0) and U(:,0)
    int nX = len_X_col_;
    int nUp = len_Up_col_;
    int NUP = nlp_NUP_;
    std::vector<double> vec_Up(x_opt.begin()+nX, x_opt.begin()+nX+nUp);
    std::vector<double> vec_U(x_opt.begin()+nX+nUp, x_opt.end());
    // reshape column-major
    std::vector<double> up0(NUP, 0.0);
    for (int r = 0; r < NUP; ++r) up0[r] = vec_Up[r + 0*NUP];
    std::vector<double> u0(NU, 0.0);
    for (int r = 0; r < NU; ++r) u0[r] = vec_U[r + 0*NU];

    // Publish cmd_vel from wheel speeds
    publish_cmd_vel_from_wheels(u0);

    // Debug/pseudo publishes
    std_msgs::msg::Float64MultiArray up_msg;
    up_msg.data.resize(up0.size());
    for (size_t i = 0; i < up0.size(); ++i) up_msg.data[i] = up0[i];
    pub_up_->publish(up_msg);

    geometry_msgs::msg::Pose2D pose_msg;
    pose_msg.x = xB_;
    pose_msg.y = yB_;
    pose_msg.theta = thB_;
    pub_cart_pose_->publish(pose_msg);

    std_msgs::msg::Float64 fx; fx.data = xref[0]; pub_ref_x_->publish(fx);
    std_msgs::msg::Float64 fy; fy.data = yref[0]; pub_ref_y_->publish(fy);
    std_msgs::msg::Float64 ft; ft.data = thref[0]; pub_ref_th_->publish(ft);

    for (int i = 0; i < params_sys_n_rbt_; ++i) {
      std_msgs::msg::Float64 ul; ul.data = u0[2*i+0]; pub_ul_dbg_[i]->publish(ul);
      std_msgs::msg::Float64 ur; ur.data = u0[2*i+1]; pub_ur_dbg_[i]->publish(ur);
      std_msgs::msg::Float64 om; om.data = up0[3+i]; pub_om_dbg_[i]->publish(om);
    }

    last_U_ = u0;
  }

  void publish_cmd_vel_from_wheels(const std::vector<double>& U) {
    double r = params_sys_robo_rdi_;
    double d = params_sys_robo_dst_;
    for (int i = 0; i < params_sys_n_rbt_; ++i) {
      double ul = U[2*i+0];
      double ur = U[2*i+1];
      double v = 0.5 * r * (ur + ul);
      double wz = (r / d) * (ur - ul);
      geometry_msgs::msg::Twist msg;
      msg.linear.x = v;
      msg.linear.y = 0.0;
      msg.linear.z = 0.0;
      msg.angular.x = 0.0;
      msg.angular.y = 0.0;
      msg.angular.z = wz;
      pub_cmd_vel_[i]->publish(msg);
    }
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CasadiMPCControllerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}


