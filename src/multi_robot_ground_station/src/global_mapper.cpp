#include "global_mapper.hpp"
#include <opencv2/opencv.hpp>

using namespace std::chrono_literals;

namespace global_mapper {

GlobalMapperNode::GlobalMapperNode() : rclcpp::Node("global_mapper_node") {

  // Get other parameters
  this->declare_parameter("use_prior_map", true);
  this->get_parameter("use_prior_map", use_prior_map_);
  RCLCPP_INFO(this->get_logger(), "use_prior_map: %s", use_prior_map_ ? "true" : "false");
  


  // Initialize object odom and global map
  global_map_ = std::make_shared<nav_msgs::msg::OccupancyGrid>();

  if (use_prior_map_) {
    this->declare_parameter("map_path", "map/hospital_map.yaml");
    this->get_parameter("map_path", map_path_);
    load_map(map_path_);
    RCLCPP_INFO(this->get_logger(), "map_path: %s", map_path_.c_str());
  }
  else{
    RCLCPP_WARN(this->get_logger(), "Not using prior map. Starting with empty map.");
  }
  
  // Get timer periods from parameters or use defaults
  this->declare_parameter("global_map_timer_period", 0.5);
  this->get_parameter("global_map_timer_period", global_map_timer_period_);
  RCLCPP_INFO(this->get_logger(), "Global map timer period: %.2f seconds", global_map_timer_period_);
  
  // Publisher for global map
  global_map_publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/global_map", 10);
  RCLCPP_INFO(this->get_logger(), "Publisher created for /global_map");  
  // Timer for periodic tasks
  global_map_timer_ = this->create_wall_timer(std::chrono::duration<double>(global_map_timer_period_), std::bind(&GlobalMapperNode::global_map_timer, this));

  RCLCPP_INFO(this->get_logger(), "GlobalMapperNode started.");

}

void GlobalMapperNode::load_map(const std::string & map_path) {
  // 방어적 초기화
  if (!global_map_) {
    global_map_ = std::make_shared<nav_msgs::msg::OccupancyGrid>();
  }
  
  // map_path example : "map/hospital_map.yaml"
  // load from map_path yaml file

  if (!reload_from_yaml(map_path)){
    RCLCPP_ERROR(this->get_logger(), "Failed to load map from %s", map_path.c_str());
    return;
  }

  global_map_->header.frame_id = "map";
  global_map_->info.resolution = this->get_parameter("resolution").as_double(); // meters per cell

  // Get width, height from png file
  std::string png_path = this->get_parameter("png_path").as_string();
  // Load png file to get width and height
  cv::Mat img = cv::imread(png_path, cv::IMREAD_GRAYSCALE);
  if (img.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load PNG image from %s", png_path.c_str());
    return;
  }
  global_map_->info.width = img.cols;
  global_map_->info.height = img.rows;
  global_map_->info.origin.position.x = this->get_parameter("origin_x").as_double();
  global_map_->info.origin.position.y = this->get_parameter("origin_y").as_double();


  global_map_->data.resize(global_map_->info.width * global_map_->info.height, -1); // Unknown cells
  for (int y = 0; y < img.rows; ++y) {
    for (int x = 0; x < img.cols; ++x) {
      uint8_t pixel = img.at<uint8_t>(y, x);
      int idx = (img.rows - 1 - y) * img.cols + x; // y축 반전
      // Typical convention: 0=free, 100=occupied, -1=unknown
      // but, this case, we use the inverse
      // if (pixel == 0) {
      //   global_map_->data[idx] = 100; // occupied
      // } else if (pixel == 255) {
      //   global_map_->data[idx] = 0;   // free
      // } else {
      //   global_map_->data[idx] = -1;  // unknown
      // }
      if (pixel == 255) {
        global_map_->data[idx] = 0;   // free
      } else {
        global_map_->data[idx] = 100;  // occupied
      }
    }
  }
  RCLCPP_INFO(this->get_logger(), "Loaded occupancy grid from PNG: %s (%dx%d)", png_path.c_str(), img.cols, img.rows);
}

void GlobalMapperNode::global_map_timer() {
  // TODO : Update global map based on latest odometries but currently just republishes prior map
  // Publish the global map
  if (global_map_) {
    global_map_->header.stamp = this->now();

    global_map_publisher_->publish(*global_map_);
    RCLCPP_INFO(this->get_logger(), "Published global map.");
  }
  else {
    RCLCPP_WARN(this->get_logger(), "Global map is not initialized.");
  }

  // Tmp coord publish, publish map's corner coordinate
  if (global_map_) {
    nav_msgs::msg::Odometry tmp_odom;
    tmp_odom.header.stamp = this->now();
    tmp_odom.header.frame_id = "map";
    tmp_odom.pose.pose.position.x = global_map_->info.origin.position.x;
    tmp_odom.pose.pose.position.y = global_map_->info.origin.position.y;
    tmp_odom.pose.pose.position.z = 0.0;
    tmp_odom.pose.pose.orientation.x = 0.0;
    tmp_odom.pose.pose.orientation.y = 0.0;
    tmp_odom.pose.pose.orientation.z = 0.0;
    tmp_odom.pose.pose.orientation.w = 1.0;
    tmp_coord_pub_->publish(tmp_odom);
    
    //map size = 1532x839
    tmp_odom.pose.pose.position.x = global_map_->info.origin.position.x + 1532*0.05; // origin_x + width*res
    tmp_odom.pose.pose.position.y = global_map_->info.origin.position.y; // origin_y
    tmp_coord_pub_->publish(tmp_odom);

    tmp_odom.pose.pose.position.x = global_map_->info.origin.position.x; // origin_x
    tmp_odom.pose.pose.position.y = global_map_->info.origin.position.y + 839*0.05; // origin_y + height*res
    tmp_coord_pub_->publish(tmp_odom);

    tmp_odom.pose.pose.position.x = global_map_->info.origin.position.x + 1532*0.05; // origin_x + width*res
    tmp_odom.pose.pose.position.y = global_map_->info.origin.position.y + 839*0.05; // origin_y + height*res
    tmp_coord_pub_->publish(tmp_odom);

        tmp_odom.pose.pose.position.x = 0;
    tmp_odom.pose.pose.position.y = 0;
    tmp_odom.pose.pose.position.z = 10;
    
    tmp_coord_pub_->publish(tmp_odom);
  }
    
}

} // namespace global_mapper