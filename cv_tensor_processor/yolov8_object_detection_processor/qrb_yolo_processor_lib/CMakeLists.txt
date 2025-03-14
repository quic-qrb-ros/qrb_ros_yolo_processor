cmake_minimum_required(VERSION 3.8)
project(qrb_yolo_processor VERSION 1.1.0)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Find dependencies (for ros package only)
find_package(ament_cmake REQUIRED)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_BUILD_TYPE Debug)

# dependencies
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

# add targets
add_library(${PROJECT_NAME} SHARED
  src/yolo_preprocess.cpp
  src/yolo_det_postprocess.cpp
  src/yolo_seg_postprocess.cpp
  src/yolo_det_overlay.cpp
  src/yolo_seg_overlay.cpp
  src/bounding_box.cpp
  src/common.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  yaml-cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 1)
# target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(TARGETS ${PROJECT_NAME}
  EXPORT qrb_yolo_processor
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include
)

# Install include files
install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)


ament_export_include_directories(include)
ament_export_libraries(qrb_yolo_processor)
ament_package()
