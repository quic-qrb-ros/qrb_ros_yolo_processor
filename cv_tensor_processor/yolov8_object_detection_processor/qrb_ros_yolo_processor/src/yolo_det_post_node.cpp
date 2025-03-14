// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "qrb_ros_yolo_processor/yolo_det_post_node.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qrb_ros::yolo_processor
{
YoloDetPostProcessNode::YoloDetPostProcessNode(const rclcpp::NodeOptions & options)
  : Node("yolo_postprocess_node", options)
{
  // params handling
  this->declare_parameter<std::string>("label_file", "");
  this->declare_parameter<double>("score_thres", 0.3);
  this->declare_parameter<double>("iou_thres", 0.5);

  std::string label_file = this->get_parameter("label_file").as_string();
  double iou_thres = this->get_parameter("iou_thres").as_double();
  double score_thres = this->get_parameter("score_thres").as_double();

  RCLCPP_INFO(this->get_logger(), "label file path: %s", label_file.c_str());
  RCLCPP_INFO(this->get_logger(), "iou_thres: %f", iou_thres);
  RCLCPP_INFO(this->get_logger(), "score_thres: %f", score_thres);

  if (label_file.empty()) {
    throw std::invalid_argument("label_file not specified.");
  }

  // topic publisher & subscriber
  sub_ = this->create_subscription<custom_msg::TensorList>("yolo_detect_tensor_output", 10,
      std::bind(&YoloDetPostProcessNode::msg_callback, this, std::placeholders::_1));

  pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("yolo_detect_result", 10);

  // instantiate post processor provided by hal-lib
  processor_ = std::make_unique<qrb::yolo_processor::YoloDetPostProcessor>(
      label_file, score_thres, iou_thres);

  RCLCPP_INFO(this->get_logger(), "init done~");
}

void YoloDetPostProcessNode::populate_tensor_from_msg(const custom_msg::TensorList::SharedPtr & msg,
    std::vector<qrb::yolo_processor::Tensor> & tensors)
{
  tensors.reserve(msg->tensor_list.size());
  for (uint32_t i = 0; i < msg->tensor_list.size(); i++) {
    qrb::yolo_processor::Tensor tensor;

    tensor.name = msg->tensor_list[i].name;
    tensor.shape = msg->tensor_list[i].shape;
    tensor.p_vec = &msg->tensor_list[i].data;
    tensor.dtype = qrb::yolo_processor::DataType::FLOAT32;

    tensors.push_back(tensor);
  }
}

void YoloDetPostProcessNode::populate_pub_msg(vision_msgs::msg::Detection2DArray & det_2d_arr,
    std::vector<qrb::yolo_processor::YoloInstance> & instances)
{
  for (auto & it : instances) {
    vision_msgs::msg::Detection2D detect;

    // Detection2D need bbox with CXYWH format
    using QrbBbox = qrb::yolo_processor::BoundingBox;
    QrbBbox box_cxywh = it.bbox.to_fmt(QrbBbox::Format::CXYWH);

    detect.bbox.center.position.x = box_cxywh[0];
    detect.bbox.center.position.y = box_cxywh[1];
    detect.bbox.size_x = box_cxywh[2];
    detect.bbox.size_y = box_cxywh[3];

    vision_msgs::msg::ObjectHypothesisWithPose hypo;
    hypo.hypothesis.score = it.score;
    hypo.hypothesis.class_id = it.label;

    detect.results.push_back(hypo);
    det_2d_arr.detections.push_back(detect);
  }
}

void YoloDetPostProcessNode::msg_callback(const custom_msg::TensorList::SharedPtr msg)
{
  uint32_t tensor_size = msg->tensor_list.size();

  // Yolo TFlite model output 3 tensors
  if (tensor_size != 3) {
    RCLCPP_ERROR(this->get_logger(), "got invalid tensor size %d", tensor_size);
    return;
  }

  // create tensor structure recognized by hal-lib
  std::vector<qrb::yolo_processor::Tensor> tensors;
  populate_tensor_from_msg(msg, tensors);

  // call hal-lib to process tensor
  std::vector<qrb::yolo_processor::YoloInstance> instances;
  processor_->process(tensors, instances);

  // construct ros msg and publish
  vision_msgs::msg::Detection2DArray det_2d_arr;
  det_2d_arr.header.stamp = msg->header.stamp;
  populate_pub_msg(det_2d_arr, instances);

  pub_->publish(det_2d_arr);
}

}  // namespace qrb_ros::yolo_processor

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(qrb_ros::yolo_processor::YoloDetPostProcessNode)
