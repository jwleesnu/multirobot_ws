# include <eigen3/Eigen/Geometry>
# include <rclcpp/rclcpp.hpp>
# include <rclcpp/parameter.hpp>
# include <nav_msgs/msg/odometry.hpp>
# include <geometry_msgs/msg/pose_stamped.h>
# include <std_msgs/msg/float64.hpp>

// with tf2
# include <tf2_ros/transform_broadcaster.h>
# include <tf2_ros/static_transform_broadcaster.h>
# include <tf2_ros/transform_listener.h>
# include <tf2_ros/buffer.h>
# include <tf2/LinearMath/Transform.h>
# include <tf2/LinearMath/Quaternion.h>
# include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>


namespace optitrack_calib {
class OptiTrackCalibNode : public rclcpp::Node {
public:
    OptiTrackCalibNode();
private:
    std::vector<rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr> optitrack_robot_pose_subscribers_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr optitrack_object_pose_subscriber_;
    std::vector<rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr> robot_encoders_subscribers_;
    
    // Store calibration results from parameter server
    std::vector<Eigen::Matrix4d> T_ri_th_r; // (r : robot, t : tail, h : head)
    Eigen::Matrix4d T_vo; // T_vo (o : object, v : Vicon/OptiTrack)
    std::vector<double> offset_towing_angle_; // initial towing angle for each robot

    // Store the extrinsic parameters b/w robot and object
    std::vector<Eigen::Matrix4d> T_o_ri; // static transform from robot to object
    std::vector<bool> robot_pose_receive_flag_{false};
    std::vector<bool> encoder_receive_flag_{false};
    bool all_receive_flag_{false};

    int num_robots_;
    std::string world_frame_id_;
    std::string object_frame_id_;
    std::vector<std::string> robot_frame_id_;
    std::vector<std::string> robot_fix_status_; // "in" or "out" 침대 고정 위치 여부
    // tf2 broadcasters
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

    void publishStaticTransforms();
    void optitrackObjectPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void optitrackRobotPoseCallback(int robot_index, const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void towingAngleCallback(int robot_index, const std_msgs::msg::Float64::SharedPtr msg);

    void TransformStampedToEigenMatrix4d(const geometry_msgs::msg::TransformStamped& tf_msg, Eigen::Matrix4d& mat){
        tf2::Transform tf_transform;
        tf2::fromMsg(tf_msg.transform, tf_transform);
        Eigen::Matrix4d eigen_mat = Eigen::Matrix4d::Identity();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                eigen_mat(i, j) = tf_transform.getBasis()[i][j];
            }
            eigen_mat(i, 3) = tf_transform.getOrigin()[i];
        }
        mat = eigen_mat;
    };
    void EigenMatrix4dToTransformStamped(const Eigen::Matrix4d& mat, geometry_msgs::msg::TransformStamped& tf_msg, const rclcpp::Time& time_stamp, const std::string& parent_frame, const std::string& child_frame){
        tf2::Matrix3x3 basis;
        basis.setValue(
            mat(0,0), mat(0,1), mat(0,2),
            mat(1,0), mat(1,1), mat(1,2),
            mat(2,0), mat(2,1), mat(2,2)
        );
        tf2::Vector3 origin(mat(0,3), mat(1,3), mat(2,3));
        tf2::Transform tf_transform(basis, origin);
        tf_msg.header.stamp = time_stamp;
        tf_msg.header.frame_id = parent_frame;
        tf_msg.child_frame_id = child_frame;
        tf_msg.transform = tf2::toMsg(tf_transform);
    };
    
};
} // namespace optitrack_calib