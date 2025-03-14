// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#ifndef _QRB_YOLO_DETECTION_POST_PROCESS_CPP_
#define _QRB_YOLO_DETECTION_POST_PROCESS_CPP_

#include <iostream>
#include <map>

#include "common.hpp"

namespace qrb::yolo_processor
{
class YoloDetPostProcessor
{
public:
  /**
   * \brief YoloDetPostProcessor Constructor
   * \param label_file Yaml file path that contains the "class_id <--> string" mapping
   * \param score_thres Score threshold (0~1) for NMS(non-maximum-suppression)
   * \param iou_thres Iou threshold (0~1) for NMS(non-maximum-suppression)
   */
  YoloDetPostProcessor(const std::string & label_file,
      float score_thres = 0.5f,
      float iou_thres = 0.3f,
      float eta = 1.f,
      int top_k = 0);
  ~YoloDetPostProcessor() = default;

  /**
   * \brief Handle the output tensors of YOLO detection model and return instance info,
   * including bounding box / score / label
   * \param tensors Tensors output from YOLO segmentation model.
   * \param instances (Output) Detected instances info will be stored here
   */
  void process(const std::vector<Tensor> & tensors, std::vector<YoloInstance> & instances);

private:
  std::map<int, std::string> label_map_;

  // member for NMS
  float score_thres_;
  float iou_thres_;
  float eta_;
  int top_k_;
};

}  // namespace qrb::yolo_processor
#endif  // _QRB_YOLO_DETECTION_POST_PROCESS_CPP_