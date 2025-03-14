// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#ifndef _QRB_ROS_YOLO_PROCESSOR_SEGMENTATION_POSTPROCESS_NODE_HPP_
#define _QRB_ROS_YOLO_PROCESSOR_SEGMENTATION_POSTPROCESS_NODE_HPP_

#include "qrb_ros_tensor_list_msgs/msg/tensor.hpp"
#include "qrb_ros_tensor_list_msgs/msg/tensor_list.hpp"
#include "qrb_ros_vision_msgs/msg/detection2_d_with_mask.hpp"
#include "qrb_ros_vision_msgs/msg/detection2_d_with_mask_array.hpp"
#include "qrb_yolo_processor/yolo_seg_postprocess.hpp"
#include "rclcpp/rclcpp.hpp"

namespace qrb_ros::yolo_processor
{
namespace custom_msg = qrb_ros_tensor_list_msgs::msg;

class YoloSegPostProcessNode : public rclcpp::Node
{
public:
  YoloSegPostProcessNode(const rclcpp::NodeOptions & options);
  ~YoloSegPostProcessNode() = default;

private:
  void msg_callback(const custom_msg::TensorList::SharedPtr msg);
  /**
   * \brief Extract tensor info from msg "TensorList" and wrap it into qrb::yolo_processor::Tensor
   * \param msg Ros msg that carries tensors output from YOLO detection model.
   * \param tensor (Output) tensor structure required by underlaying lib
   */
  void populate_tensor_from_msg(const custom_msg::TensorList::SharedPtr & msg,
      std::vector<qrb::yolo_processor::Tensor> & tensor);

  /**
   * \brief Populate ros msg as per info of detected instances.
   * \param instances Contains detected instance info
   * \param msg (Output) msg to be filled with instance info
   */
  void populate_pub_msg(qrb_ros_vision_msgs::msg::Detection2DWithMaskArray & msg,
      std::vector<qrb::yolo_processor::YoloInstance> & instances);
  std::unique_ptr<qrb::yolo_processor::YoloSegPostProcessor> processor_{ nullptr };
  rclcpp::Subscription<custom_msg::TensorList>::SharedPtr sub_{ nullptr };
  rclcpp::Publisher<qrb_ros_vision_msgs::msg::Detection2DWithMaskArray>::SharedPtr pub_{ nullptr };
};

}  // namespace qrb_ros::yolo_processor

#endif  //_QRB_ROS_YOLO_PROCESSOR_SEGMENTATION_POSTPROCESS_NODE_HPP_
