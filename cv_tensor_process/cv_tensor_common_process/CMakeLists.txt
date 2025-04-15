cmake_minimum_required(VERSION 3.8)
project(qrb_ros_cv_tensor_common_process)

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
  ${PROJECT_SOURCE_DIR}/src/include
)

ament_auto_add_library(qrb_ros_cv_tensor_common_process_component SHARED
  src/cv_tensor_common_process.cpp
)

rclcpp_components_register_nodes(
  qrb_ros_cv_tensor_common_process_component
  PLUGIN "qrb_ros::cv_tensor_common_process::CvTensorCommonProcessNode"
)

## install launch file
## install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)

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
