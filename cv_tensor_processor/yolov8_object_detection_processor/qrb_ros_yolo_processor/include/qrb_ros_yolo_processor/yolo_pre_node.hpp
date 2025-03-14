// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#ifndef _QRB_ROS_YOLO_PROCESSOR_PREPROCESS_HPP_
#define _QRB_ROS_YOLO_PROCESSOR_PREPROCESS_HPP_

#include "qrb_ros_tensor_list_msgs/msg/tensor.hpp"
#include "qrb_ros_tensor_list_msgs/msg/tensor_list.hpp"
#include "qrb_yolo_processor/yolo_preprocess.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace qrb_ros::yolo_processor
{
namespace custom_msg = qrb_ros_tensor_list_msgs::msg;

class YoloPreProcessNode : public rclcpp::Node
{
public:
  YoloPreProcessNode(const rclcpp::NodeOptions & options);
  ~YoloPreProcessNode() = default;

private:
  std::unique_ptr<qrb::yolo_processor::YoloPreProcessor> processor_{ nullptr };
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_{ nullptr };
  rclcpp::Publisher<custom_msg::TensorList>::SharedPtr pub_{ nullptr };
  custom_msg::Tensor msg_ts_;

  void msg_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg_img);
};
}  // namespace qrb_ros::yolo_processor

#endif  // _QRB_ROS_YOLO_PROCESSOR_PREPROCESS_HPP_
