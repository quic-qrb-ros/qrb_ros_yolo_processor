// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "qrb_ros_yolo_processor/yolo_pre_node.hpp"

#include "builtin_interfaces/msg/time.hpp"
#include "cv_bridge/cv_bridge.h"

namespace qrb_ros::yolo_processor
{
// Tensor Dtype
enum class TensorDtype
{
  UINT8,
  INT8,
  FLOAT32,
  FLOAT64,
};

YoloPreProcessNode::YoloPreProcessNode(const rclcpp::NodeOptions & options)
  : Node("yolo_preprocess_node", options)
{
  sub_ = this->create_subscription<sensor_msgs::msg::Image>("yolo_input_img", 10,
      std::bind(&YoloPreProcessNode::msg_callback, this, std::placeholders::_1));

  pub_ = this->create_publisher<custom_msg::TensorList>("yolo_raw_img", 10);

  // fixed params reuqired by model
  std::array<int, 4> shape = { 1, 640, 640, 3 };
  processor_ = std::make_unique<qrb::yolo_processor::YoloPreProcessor>(
      shape, qrb::yolo_processor::DataType::FLOAT32);

  // fill msg body, also alloc buf in advance
  msg_ts_.name = "image";
  msg_ts_.data_type = static_cast<int>(TensorDtype::FLOAT32);
  msg_ts_.shape = { 1, 640, 640, 3 };

  msg_ts_.data.resize(1 * 640 * 640 * 3 * sizeof(float));
  RCLCPP_INFO(this->get_logger(), "init done~");
}

void YoloPreProcessNode::msg_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg_img)
{
  cv_bridge::CvImageConstPtr cv_ptr =
      cv_bridge::toCvShare(msg_img, sensor_msgs::image_encodings::BGR8);

  void * msg_buf = msg_ts_.data.data();
  size_t size = msg_ts_.data.size() * sizeof(msg_ts_.data[0]);

  // process data, save processed data into msg_buf
  bool res = processor_->process(cv_ptr->image, msg_buf, size);
  if (res != true) {
    RCLCPP_ERROR(this->get_logger(), "pre process failed.");
    return;
  }

  // construct msg and publish
  custom_msg::TensorList msg_out;
  msg_out.header.stamp = msg_img->header.stamp;
  msg_out.tensor_list.push_back(msg_ts_);

  pub_->publish(msg_out);
}

}  // namespace qrb_ros::yolo_processor

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(qrb_ros::yolo_processor::YoloPreProcessNode)
