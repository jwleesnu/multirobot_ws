#include "optitrack_calib.hpp"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<optitrack_calib::OptiTrackCalibNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

