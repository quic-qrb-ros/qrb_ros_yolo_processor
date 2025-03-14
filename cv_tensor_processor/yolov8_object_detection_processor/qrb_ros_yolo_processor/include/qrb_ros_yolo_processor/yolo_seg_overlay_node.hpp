// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#ifndef _QRB_ROS_YOLO_PROCESSOR_SEGMENTATION_OVERLAY_NODE_HPP_
#define _QRB_ROS_YOLO_PROCESSOR_SEGMENTATION_OVERLAY_NODE_HPP_

#include <memory>

#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/sync_policies/exact_time.h"
#include "message_filters/synchronizer.h"
#include "qrb_ros_vision_msgs/msg/detection2_d_with_mask.hpp"
#include "qrb_ros_vision_msgs/msg/detection2_d_with_mask_array.hpp"
#include "qrb_yolo_processor/yolo_seg_overlay.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace qrb_ros::yolo_processor
{
using ExactPolicy = message_filters::sync_policies::ExactTime<sensor_msgs::msg::Image,
    qrb_ros_vision_msgs::msg::Detection2DWithMaskArray>;
using ExactSync = message_filters::Synchronizer<ExactPolicy>;
using namespace std::placeholders;

class YoloSegOverlayNode : public rclcpp::Node
{
public:
  YoloSegOverlayNode(const rclcpp::NodeOptions & options);
  ~YoloSegOverlayNode() = default;

private:
  std::unique_ptr<qrb::yolo_processor::YoloSegOverlay> overlay_{ nullptr };
  std::unique_ptr<ExactSync> exact_sync_{ nullptr };
  std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> pub_{ nullptr };
  message_filters::Subscriber<qrb_ros_vision_msgs::msg::Detection2DWithMaskArray> yolo_seg_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> img_sub_;

  void msg_callback(sensor_msgs::msg::Image::ConstSharedPtr img_msg,
      qrb_ros_vision_msgs::msg::Detection2DWithMaskArray::ConstSharedPtr yolo_msg);
};

}  // namespace qrb_ros::yolo_processor
#endif  // _QRB_ROS_YOLO_PROCESSOR_SEGMENTATION_OVERLAY_NODE_HPP_