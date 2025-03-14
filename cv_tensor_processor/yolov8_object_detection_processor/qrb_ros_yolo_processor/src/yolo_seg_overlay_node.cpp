// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "qrb_ros_yolo_processor/yolo_seg_overlay_node.hpp"

#include "cv_bridge/cv_bridge.h"

namespace qrb_ros::yolo_processor
{
YoloSegOverlayNode::YoloSegOverlayNode(const rclcpp::NodeOptions & options)
  : Node("yolo_seg_overlay_node", options)
{
  img_sub_.subscribe(this, "yolo_input_img");
  yolo_seg_sub_.subscribe(this, "yolo_segment_result");
  pub_ = create_publisher<sensor_msgs::msg::Image>("yolo_segment_overlay", 10);

  exact_sync_.reset(new ExactSync(ExactPolicy(10), img_sub_, yolo_seg_sub_));
  exact_sync_->registerCallback(std::bind(&YoloSegOverlayNode::msg_callback, this, _1, _2));

  overlay_ = std::make_unique<qrb::yolo_processor::YoloSegOverlay>();
  RCLCPP_INFO(this->get_logger(), "init done~");
}

void YoloSegOverlayNode::msg_callback(sensor_msgs::msg::Image::ConstSharedPtr img_msg,
    qrb_ros_vision_msgs::msg::Detection2DWithMaskArray::ConstSharedPtr yolo_msg)
{
  // publish image as is when no detections info
  if (yolo_msg->array.empty()) {
    RCLCPP_INFO(this->get_logger(), "empty image~~");
    pub_->publish(*img_msg);
    return;
  }

  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  std::vector<qrb::yolo_processor::YoloInstance> instances;

  for (const auto & it : yolo_msg->array) {
    float x = it.detection.bbox.center.position.x;
    float y = it.detection.bbox.center.position.y;
    float w = it.detection.bbox.size_x;
    float h = it.detection.bbox.size_y;
    float score = it.detection.results[0].hypothesis.score;
    std::string label = it.detection.results[0].hypothesis.class_id;

    using QrbBbox = qrb::yolo_processor::BoundingBox;
    QrbBbox::Format box_fmt = QrbBbox::Format::CXYWH;
    qrb::yolo_processor::YoloInstance instance(
        x, y, w, h, box_fmt, score, label, it.instance_mask.data);
    instances.push_back(instance);
  }

  overlay_->draw_inplace(instances, cv_ptr->image);
  pub_->publish(*(cv_ptr->toImageMsg()));
}

}  // namespace qrb_ros::yolo_processor

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(qrb_ros::yolo_processor::YoloSegOverlayNode)