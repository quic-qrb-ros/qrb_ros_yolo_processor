// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#ifndef _QRB_ROS_YOLO_PROCESSOR_POSTPROCESS_HPP_
#define _QRB_ROS_YOLO_PROCESSOR_POSTPROCESS_HPP_

#include "qrb_ros_tensor_list_msgs/msg/tensor.hpp"
#include "qrb_ros_tensor_list_msgs/msg/tensor_list.hpp"
#include "qrb_yolo_processor/yolo_det_postprocess.hpp"
#include "rclcpp/rclcpp.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

namespace qrb_ros::yolo_processor
{
namespace custom_msg = qrb_ros_tensor_list_msgs::msg;

class YoloDetPostProcessNode : public rclcpp::Node
{
public:
  YoloDetPostProcessNode(const rclcpp::NodeOptions & options);
  ~YoloDetPostProcessNode() = default;

private:
  std::unique_ptr<qrb::yolo_processor::YoloDetPostProcessor> processor_{ nullptr };
  rclcpp::Subscription<custom_msg::TensorList>::SharedPtr sub_{ nullptr };
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pub_{ nullptr };

  void msg_callback(const custom_msg::TensorList::SharedPtr msg);

  /**
   * \brief Extract tensor info from msg "TensorList" and wrap it into qrb::yolo_processor::Tensor
   * \param msg Ros msg that carries tensors output from YOLO detection model.
   * \param tensor (Output) tensor structure required by underlaying lib
   */
  void populate_tensor_from_msg(const custom_msg::TensorList::SharedPtr & msg,
      std::vector<qrb::yolo_processor::Tensor> & tensors);

  /**
   * \brief Populate vision_msgs::msg::Detection2DArray from Yolo detection result
   * \param instances: Contains yolo instance info
   * \param det_2d_arr: (Output) target msg to be filled with instance info
   */
  void populate_pub_msg(vision_msgs::msg::Detection2DArray & det_2d_arr,
      std::vector<qrb::yolo_processor::YoloInstance> & instances);
};
}  // namespace qrb_ros::yolo_processor

#endif  // _QRB_ROS_YOLO_PROCESSOR_POSTPROCESS_HPP_