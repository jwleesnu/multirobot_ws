#include "optitrack_calib.hpp"

using namespace std::chrono_literals;
namespace optitrack_calib {

OptiTrackCalibNode::OptiTrackCalibNode() : rclcpp::Node("optitrack_calib_node") {
    // Get number of robots from parameter server
    this->declare_parameter("num_robot", 2); // Default to 2 robots if not set
    this->get_parameter("num_robot", num_robots_);
    RCLCPP_INFO(this->get_logger(), "Number of robots: %d", num_robots_);
    
    this->declare_parameter("world_frame_id", "world");
    this->get_parameter("world_frame_id", world_frame_id_);
    RCLCPP_INFO(this->get_logger(), "World frame ID: %s", world_frame_id_.c_str());

    // Resize vectors based on number of robots
    offset_towing_angle_.resize(num_robots_, 0.0);
    T_ri_th_r.resize(num_robots_, Eigen::Matrix4d::Identity());
    T_o_ri.resize(num_robots_, Eigen::Matrix4d::Identity());
    robot_pose_receive_flag_.resize(num_robots_, false);
    encoder_receive_flag_.resize(num_robots_, false);
    
    
    
    for (int i = 0; i < num_robots_; ++i) {
        // Initialize subscriptions for each robot
        std::string robot_frame_param = "robot_" + std::to_string(i+1) + "_frame_id";
        this->declare_parameter(robot_frame_param, "robot_" + std::to_string(i));
        std::string frame_id;
        this->get_parameter(robot_frame_param, frame_id);
        robot_frame_id_.push_back(frame_id);

        std::string robot_fix_param = "robot_" + std::to_string(i+1) + "_fix_status";
        this->declare_parameter(robot_fix_param, "out"); // Default to "out"
        std::string fix_status;
        this->get_parameter(robot_fix_param, fix_status);
        robot_fix_status_.push_back(fix_status);

        std::string optitrack_robot_topic;
        std::string robot_calib_param = "robot_" + std::to_string(i+1) + "_T_r" + std::to_string(i+1);
        optitrack_robot_topic = "/optitrackrobot_" + std::to_string(i+1);
        if (fix_status == "in") {
            optitrack_robot_topic += "_tail";
            robot_calib_param += "t_r" + std::to_string(i+1);
            RCLCPP_INFO(this->get_logger(), "Robot %d is inner fixed. Subscribing to %s", i+1, optitrack_robot_topic.c_str());
        } else {
            optitrack_robot_topic += "_head";
            robot_calib_param += "h_r" + std::to_string(i+1);
            RCLCPP_INFO(this->get_logger(), "Robot %d is outter fixed. Subscribing to %s", i+1, optitrack_robot_topic.c_str());
        }
        optitrack_robot_pose_subscribers_.push_back(
            this->create_subscription<geometry_msgs::msg::PoseStamped>(optitrack_robot_topic, 10,
                [this, i](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {this->optitrackRobotPoseCallback(i, msg);}));
        RCLCPP_INFO(this->get_logger(), "Subscribed to %s", optitrack_robot_topic.c_str());

        robot_encoders_subscribers_.push_back(
            this->create_subscription<std_msgs::msg::Float64>(
                "/robot_" + std::to_string(i+1) + "/" + "towing_angle", 10,
                [this, i](const std_msgs::msg::Float64::SharedPtr msg) { this->towingAngleCallback(i, msg); }
            )
        );
        RCLCPP_INFO(this->get_logger(), "Subscribed to /robot_%d/towing_angle", i+1);

        // Store calibration results from parameter server T_ri_th_r -> saved as 12 elements vector
        std::vector<double> calib_vec(12, 0.0);
        this->declare_parameter(robot_calib_param, calib_vec);
        this->get_parameter(robot_calib_param, calib_vec);
        double sum = std::accumulate(calib_vec.begin(), calib_vec.end(), 0.0);

        // check nan value
        if (std::any_of(calib_vec.begin(), calib_vec.end(), [](double v) { return !std::isfinite(v); })) {
            RCLCPP_ERROR(this->get_logger(), "NaN detected in calibration vector for robot %d! Skipping TF broadcast.", i+1);
            return;
        }

        if (calib_vec.size() == 12 && sum != 0.0) {
            // save as 4x4 matrix in T_ri_th_r
            T_ri_th_r[i] << calib_vec[0], calib_vec[1], calib_vec[2], calib_vec[3],
                           calib_vec[4], calib_vec[5], calib_vec[6], calib_vec[7],
                           calib_vec[8], calib_vec[9], calib_vec[10], calib_vec[11],
                           0, 0, 0, 1;
            RCLCPP_INFO(this->get_logger(), "Loaded calibration for robot %d from parameters.", i+1);
        } else {
            RCLCPP_WARN(this->get_logger(), "Calibration vector for robot %d is not of size 12!", i+1);
        }
    }

    // Subscription for object pose
    this->declare_parameter("object_frame_id", "object");
    this->get_parameter("object_frame_id", object_frame_id_);
    std::string optitrack_object_topic = "/optitrack" + object_frame_id_;
    RCLCPP_INFO(this->get_logger(), "Subscribing to object topic: %s", optitrack_object_topic.c_str());
    optitrack_object_pose_subscriber_ = 
        this->create_subscription<geometry_msgs::msg::PoseStamped>(optitrack_object_topic, 10, 
            std::bind(&OptiTrackCalibNode::optitrackObjectPoseCallback, this, std::placeholders::_1));

    // Store object calibration results from parameter server T_VO -> saved as 12 elements vector
    std::vector<double> calib_vec(12, 0.0);
    this->declare_parameter("object_T_vo", calib_vec);
    this->get_parameter("object_T_vo", calib_vec);
    double sum = std::accumulate(calib_vec.begin(), calib_vec.end(), 0.0);
    if (calib_vec.size() == 12 && sum != 0.0) {
        // save as 4x4 matrix in T_vo
        T_vo << calib_vec[0], calib_vec[1], calib_vec[2], calib_vec[3],
               calib_vec[4], calib_vec[5], calib_vec[6], calib_vec[7],
               calib_vec[8], calib_vec[9], calib_vec[10], calib_vec[11],
               0, 0, 0, 1;
        RCLCPP_INFO(this->get_logger(), "Loaded object calibration from parameters.");
    } else {
        RCLCPP_WARN(this->get_logger(), "Object calibration vector is not of size 12 or is all zeros!");
    }

    // Publisher for static transforms
    static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this); // <-- 반드시 필요
    publishStaticTransforms();
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    RCLCPP_INFO(this->get_logger(), "OptiTrack Calibration Node Initialized.");
}

void OptiTrackCalibNode::publishStaticTransforms() {
    std::vector<geometry_msgs::msg::TransformStamped> static_transforms;

    // Publish T_vo
    geometry_msgs::msg::TransformStamped t_vo_msg;
    t_vo_msg.header.stamp = this->now();
    t_vo_msg.header.frame_id = "optitrack/" + object_frame_id_;
    t_vo_msg.child_frame_id = object_frame_id_;
    Eigen::Vector3d translation = T_vo.block<3,1>(0,3);
    Eigen::Matrix3d rotation = T_vo.block<3,3>(0,0);
    Eigen::Quaterniond quat(rotation);
    t_vo_msg.transform.translation.x = translation.x();
    t_vo_msg.transform.translation.y = translation.y();
    t_vo_msg.transform.translation.z = translation.z();
    t_vo_msg.transform.rotation.x = quat.x();
    t_vo_msg.transform.rotation.y = quat.y();
    t_vo_msg.transform.rotation.z = quat.z();
    t_vo_msg.transform.rotation.w = quat.w();
    static_transforms.push_back(t_vo_msg);
    RCLCPP_INFO(this->get_logger(), "Published static transform from optitrack to %s", object_frame_id_.c_str());

    // Publish T_ri_th_r for each robot
    for (int i = 0; i < num_robots_; ++i) {
        geometry_msgs::msg::TransformStamped t_ri_msg;
        t_ri_msg.header.stamp = this->now();
        t_ri_msg.header.frame_id = "optitrack/" + robot_frame_id_[i];
        t_ri_msg.child_frame_id = robot_frame_id_[i];
        Eigen::Vector3d translation = T_ri_th_r[i].block<3,1>(0,3);
        Eigen::Matrix3d rotation = T_ri_th_r[i].block<3,3>(0,0);
        Eigen::Quaterniond quat(rotation);
        t_ri_msg.transform.translation.x = translation.x();
        t_ri_msg.transform.translation.y = translation.y();
        t_ri_msg.transform.translation.z = translation.z();
        t_ri_msg.transform.rotation.x = quat.x();
        t_ri_msg.transform.rotation.y = quat.y();
        t_ri_msg.transform.rotation.z = quat.z();
        t_ri_msg.transform.rotation.w = quat.w();
        static_transforms.push_back(t_ri_msg);
        RCLCPP_INFO(this->get_logger(), "Published static transform from optitrack to %s", robot_frame_id_[i].c_str());
    }
    static_tf_broadcaster_->sendTransform(static_transforms);
}
void OptiTrackCalibNode::optitrackRobotPoseCallback(int robot_index, const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    if(all_receive_flag_) return; // If all initial poses are received, ignore further callbacks

    RCLCPP_INFO(this->get_logger(), "Received pose for robot %d from OptiTrack.", robot_index+1);
    // Send the received pose after transforming with T_ri_th_r
    Eigen::Matrix4d T_optitrack;
    // quaternion to rotation matrix
    Eigen::Quaterniond q(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z);
    Eigen::Matrix3d R = q.toRotationMatrix();
    T_optitrack.block<3,3>(0,0) = R;
    T_optitrack(0,3) = msg->pose.position.x;
    T_optitrack(1,3) = msg->pose.position.y;
    T_optitrack(2,3) = msg->pose.position.z;
    T_optitrack(3,0) = 0; T_optitrack(3,1) = 0; T_optitrack(3,2) = 0; T_optitrack(3,3) = 1; 
    Eigen::Matrix4d T_robot = T_optitrack * T_ri_th_r[robot_index];

    // Broadcast the transform
    geometry_msgs::msg::TransformStamped robot_tf;
    robot_tf.header.stamp = msg->header.stamp;
    robot_tf.header.frame_id = world_frame_id_;
    robot_tf.child_frame_id = robot_frame_id_[robot_index];
    robot_tf.transform.translation.x = T_robot(0,3);
    robot_tf.transform.translation.y = T_robot(1,3);
    robot_tf.transform.translation.z = T_robot(2,3);
    Eigen::Matrix3d R_robot = T_robot.block<3,3>(0,0);
    Eigen::Quaterniond q_robot(R_robot);
    // NaN 체크
    if (!std::isfinite(T_robot(0,3)) || !std::isfinite(T_robot(1,3)) || !std::isfinite(T_robot(2,3)) ||
        !std::isfinite(q_robot.x()) || !std::isfinite(q_robot.y()) || !std::isfinite(q_robot.z()) || !std::isfinite(q_robot.w())) {
        RCLCPP_ERROR(this->get_logger(), "NaN detected in transform for robot %d! Skipping TF broadcast.", robot_index+1);
        return;
    }
    robot_tf.transform.rotation.x = q_robot.x();
    robot_tf.transform.rotation.y = q_robot.y();
    robot_tf.transform.rotation.z = q_robot.z();
    robot_tf.transform.rotation.w = q_robot.w();
    tf_broadcaster_->sendTransform(robot_tf);
    RCLCPP_INFO(this->get_logger(), "Broadcasted transform for robot %d", robot_index+1);
    
    // turn on the receive flag
    if (!robot_pose_receive_flag_[robot_index]) {
        robot_pose_receive_flag_[robot_index] = true;
        RCLCPP_INFO(this->get_logger(), "Received first pose for robot %d from OptiTrack.", robot_index+1);
    }
}

void OptiTrackCalibNode::optitrackObjectPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    // Send the received pose after transforming with T_vo
    Eigen::Matrix4d T_optitrack;
    // quaternion to rotation matrix
    Eigen::Quaterniond q(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z);
    Eigen::Matrix3d R = q.toRotationMatrix();
    T_optitrack.block<3,3>(0,0) = R;
    T_optitrack(0,3) = msg->pose.position.x;
    T_optitrack(1,3) = msg->pose.position.y;
    T_optitrack(2,3) = msg->pose.position.z;
    T_optitrack(3,0) = 0; T_optitrack(3,1) = 0; T_optitrack(3,2) = 0; T_optitrack(3,3) = 1; 
    Eigen::Matrix4d T_object = T_optitrack * T_vo;

    // Broadcast the transform
    geometry_msgs::msg::TransformStamped object_tf;
    object_tf.header.stamp = msg->header.stamp;
    object_tf.header.frame_id = world_frame_id_;
    object_tf.child_frame_id = object_frame_id_;
    Eigen::Vector3d translation = T_object.block<3,1>(0,3);
    Eigen::Matrix3d rotation = T_object.block<3,3>(0,0);
    Eigen::Quaterniond quat(rotation);
    object_tf.transform.translation.x = translation.x();
    object_tf.transform.translation.y = translation.y();
    object_tf.transform.translation.z = translation.z();
    object_tf.transform.rotation.x = quat.x();
    object_tf.transform.rotation.y = quat.y();
    object_tf.transform.rotation.z = quat.z();
    object_tf.transform.rotation.w = quat.w();
    tf_broadcaster_->sendTransform(object_tf);
    RCLCPP_INFO(this->get_logger(), "Broadcasted transform for object");
    // Turn on the receive flag
    if (!all_receive_flag_) {
        if (std::all_of(robot_pose_receive_flag_.begin(), robot_pose_receive_flag_.end(), [](bool v) { return v; })
            && std::all_of(encoder_receive_flag_.begin(), encoder_receive_flag_.end(), [](bool v) { return v; })) {
            all_receive_flag_ = true;
            RCLCPP_INFO(this->get_logger(), "All initial poses received. System is ready.");
            // Store T_o_ri, relative transform between object and each robot
            for (int i = 0; i < num_robots_; ++i) {
                // get T_ri from tf
                std::string robot_tf_name = robot_frame_id_[i];
                geometry_msgs::msg::TransformStamped robot_tf;
                try {
                    robot_tf = tf_buffer_->lookupTransform(world_frame_id_, robot_tf_name, tf2::TimePointZero);
                    RCLCPP_INFO(this->get_logger(), "Got transform for robot %d", i+1);
                } catch (tf2::TransformException & ex) {
                    RCLCPP_WARN(this->get_logger(), "Could not get transform for robot %d: %s", i+1, ex.what());
                    continue;
                }

                // Compute T_o_ri
                Eigen::Matrix4d T_ri = Eigen::Matrix4d::Identity();
                // Properly convert quaternion to rotation matrix
                Eigen::Quaterniond q(
                    robot_tf.transform.rotation.w,
                    robot_tf.transform.rotation.x,
                    robot_tf.transform.rotation.y,
                    robot_tf.transform.rotation.z
                );
                T_ri.block<3,3>(0,0) = q.toRotationMatrix();
                T_ri(0,3) = robot_tf.transform.translation.x;
                T_ri(1,3) = robot_tf.transform.translation.y;
                T_ri(2,3) = robot_tf.transform.translation.z;
                T_o_ri[i] = T_object.inverse() * T_ri;
                RCLCPP_INFO(this->get_logger(), "Computed T_o_r%d", i+1);
            }
        }
    }
}

void OptiTrackCalibNode::towingAngleCallback(int robot_index, const std_msgs::msg::Float64::SharedPtr msg) {
    if (!encoder_receive_flag_[robot_index]) {
        encoder_receive_flag_[robot_index] = true;
        offset_towing_angle_[robot_index] = msg->data;
        RCLCPP_INFO(this->get_logger(), "Received first towing angle for robot %d.", robot_index+1);
    }
    if (all_receive_flag_) {
        // publish robot tf with towing angle
        Eigen::Matrix4d T_ri_ri_current = Eigen::Matrix4d::Identity();
        double towing_angle = msg->data - offset_towing_angle_[robot_index];
        T_ri_ri_current.block<3,3>(0,0) = Eigen::AngleAxisd(towing_angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        // get current T_o from tf
        geometry_msgs::msg::TransformStamped object_tf;
        try {
            object_tf = tf_buffer_->lookupTransform(world_frame_id_, object_frame_id_, tf2::TimePointZero);
            RCLCPP_INFO(this->get_logger(), "Got transform for object");
        } catch (tf2::TransformException & ex) {
            RCLCPP_WARN(this->get_logger(), "Could not get transform for object: %s", ex.what());
            return;
        }
        Eigen::Matrix4d T_o = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d T_ri_cur = Eigen::Matrix4d::Identity();
        // Convert object_tf to Eigen matrix
        TransformStampedToEigenMatrix4d(object_tf, T_o);
        // Compute T_ri
        T_ri_cur = T_o * T_o_ri[robot_index] * T_ri_ri_current;
        // Publish the transform
        geometry_msgs::msg::TransformStamped tf_msg;
        EigenMatrix4dToTransformStamped(T_ri_cur, tf_msg, this->now(), world_frame_id_, robot_frame_id_[robot_index]);
        tf_broadcaster_->sendTransform(tf_msg);
    }
    
}
}