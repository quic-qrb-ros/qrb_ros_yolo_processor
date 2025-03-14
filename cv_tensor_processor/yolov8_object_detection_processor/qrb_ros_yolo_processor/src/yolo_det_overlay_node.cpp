// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "qrb_ros_yolo_processor/yolo_det_overlay_node.hpp"

#include "cv_bridge/cv_bridge.h"

namespace qrb_ros::yolo_processor
{
YoloDetOverlayNode::YoloDetOverlayNode(const rclcpp::NodeOptions & options)
  : Node("yolo_det_overlay_node", options)
{
  img_sub_.subscribe(this, "yolo_input_img");
  yolo_det_sub_.subscribe(this, "yolo_detect_result");
  pub_ = create_publisher<sensor_msgs::msg::Image>("yolo_detect_overlay", 10);

  exact_sync_.reset(new ExactSync(ExactPolicy(10), img_sub_, yolo_det_sub_));
  exact_sync_->registerCallback(std::bind(&YoloDetOverlayNode::msg_callback, this, _1, _2));

  overlay_ = std::make_unique<qrb::yolo_processor::YoloDetOverlay>();
  RCLCPP_INFO(this->get_logger(), "init done~");
}

void YoloDetOverlayNode::msg_callback(sensor_msgs::msg::Image::ConstSharedPtr img_msg,
    vision_msgs::msg::Detection2DArray::ConstSharedPtr yolo_msg)
{
  // publish image as is when no detections info
  if (yolo_msg->detections.empty()) {
    RCLCPP_INFO(this->get_logger(), "empty detection.");
    pub_->publish(*img_msg);
    return;
  }

  // RCLCPP_INFO(this->get_logger(), "Received image~~");

  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  std::vector<qrb::yolo_processor::YoloInstance> instances;

  for (const auto & it : yolo_msg->detections) {
    using QrbBbox = qrb::yolo_processor::BoundingBox;
    QrbBbox::Format bbox_fmt = QrbBbox::Format::CXYWH;
    float x = it.bbox.center.position.x;
    float y = it.bbox.center.position.y;
    float w = it.bbox.size_x;
    float h = it.bbox.size_y;
    float score = it.results[0].hypothesis.score;

    std::string label = it.results[0].hypothesis.class_id;
    qrb::yolo_processor::YoloInstance instance({ x, y, w, h }, bbox_fmt, score, label);
    instances.push_back(instance);
  }

  overlay_->draw_inplace(instances, cv_ptr->image);
  pub_->publish(*(cv_ptr->toImageMsg()));
}

}  // namespace qrb_ros::yolo_processor

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(qrb_ros::yolo_processor::YoloDetOverlayNode)