#include "global_mapper.hpp"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<global_mapper::GlobalMapperNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

