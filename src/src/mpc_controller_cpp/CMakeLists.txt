cmake_minimum_required(VERSION 3.8)
project(mpc_controller_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
# find_package(OpenCV REQUIRED)

# CasADi via pkg-config
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(CASADI QUIET casadi)
endif()

# Fallback: manual search if pkg-config failed
if(NOT CASADI_FOUND)
  message(WARNING "pkg-config could not find casadi; trying manual search via CASADI_PREFIX/CASADI_ROOT/CONDA_PREFIX...")
  set(CASADI_HINTS $ENV{CASADI_PREFIX} $ENV{CASADI_ROOT} $ENV{CONDA_PREFIX})
  find_path(CASADI_INCLUDE_DIR casadi/casadi.hpp
    HINTS ${CASADI_HINTS}
    PATHS /usr /usr/local
    PATH_SUFFIXES include include/casadi
  )
  find_library(CASADI_LIBRARY NAMES casadi
    HINTS ${CASADI_HINTS}
    PATHS /usr /usr/local
    PATH_SUFFIXES lib lib64
  )
  if(NOT CASADI_INCLUDE_DIR OR NOT CASADI_LIBRARY)
    message(FATAL_ERROR "CasADi not found. Install libcasadi-dev (apt) or set CASADI_PREFIX or CASADI_INCLUDE_DIR/CASADI_LIBRARY.")
  endif()
  set(CASADI_INCLUDE_DIRS ${CASADI_INCLUDE_DIR})
  set(CASADI_LIBRARIES ${CASADI_LIBRARY})
endif()

include_directories(${CASADI_INCLUDE_DIRS} # ${OpenCV_INCLUDE_DIRS}
)

add_executable(casadi_controller_node
  src/casadi_controller_node.cpp
)

ament_target_dependencies(casadi_controller_node
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

target_link_libraries(casadi_controller_node ${CASADI_LIBRARIES} # ${OpenCV_LIBS}
)

if(CASADI_CFLAGS_OTHER)
  target_compile_options(casadi_controller_node PRIVATE ${CASADI_CFLAGS_OTHER})
endif()
if(CASADI_LDFLAGS_OTHER)
  target_link_options(casadi_controller_node PRIVATE ${CASADI_LDFLAGS_OTHER})
endif()

install(TARGETS casadi_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()


