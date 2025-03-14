// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#ifndef _QRB_YOLO_DETECTION_OVERLAY_HPP_
#define _QRB_YOLO_DETECTION_OVERLAY_HPP_

#include <vector>

#include "common.hpp"

namespace qrb::yolo_processor
{
class YoloDetOverlay
{
public:
  YoloDetOverlay() = default;
  ~YoloDetOverlay() = default;

  /**
   * \brief draw bounding boxes on the given cv::Mat object in an in-place manner.
   * \param instances Contains the detected instances info.
   * \param img The instance info will be overlaid onto this image.
   */
  void draw_inplace(std::vector<YoloInstance> & instances, cv::Mat & img);
};

}  // namespace qrb::yolo_processor
#endif  // _QRB_YOLO_DETECTION_OVERLAY_HPP_