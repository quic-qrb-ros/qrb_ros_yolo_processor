cmake_minimum_required(VERSION 3.8)
project(qrb_ros_yolo_processor)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# set(CMAKE_BUILD_TYPE Debug)

# find dependencies
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# specify header search path
include_directories(
  ${PROJECT_SOURCE_DIR}/include
)

ament_auto_add_library(qrb_ros_yolo_process_component SHARED
  src/yolo_pre_node.cpp
  src/yolo_det_post_node.cpp
  src/yolo_det_overlay_node.cpp
  src/yolo_seg_post_node.cpp
  src/yolo_seg_overlay_node.cpp
)
# target_link_libraries(qrb_ros_yolo_process_component qrb_yolo_processor)

rclcpp_components_register_nodes(
  qrb_ros_yolo_process_component
  PLUGIN "qrb_ros::yolo_processor::YoloPreProcessNode"
)

rclcpp_components_register_nodes(
  qrb_ros_yolo_process_component
  PLUGIN "qrb_ros::yolo_processor::YoloDetPostProcessNode"
)

rclcpp_components_register_nodes(
  qrb_ros_yolo_process_component
  PLUGIN "qrb_ros::yolo_processor::YoloDetOverlayNode"
)

rclcpp_components_register_nodes(
  qrb_ros_yolo_process_component
  PLUGIN "qrb_ros::yolo_processor::YoloSegPostProcessNode"
)

rclcpp_components_register_nodes(
  qrb_ros_yolo_process_component
  PLUGIN "qrb_ros::yolo_processor::YoloSegOverlayNode"
)

## install launch file
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
