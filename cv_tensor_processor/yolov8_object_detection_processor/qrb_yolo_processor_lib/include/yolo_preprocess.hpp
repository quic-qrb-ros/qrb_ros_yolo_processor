// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#ifndef _QRB_YOLO_PRE_PROCESS_CPP_
#define _QRB_YOLO_PRE_PROCESS_CPP_

#include <array>
#include <iostream>

#include "common.hpp"
#include "opencv2/opencv.hpp"

namespace qrb::yolo_processor
{
class YoloPreProcessor
{
public:
  /**
   * \brief YoloPreProcessor constructor
   * \param shape: The input tensor shape required by YOLO model
   *    - shape[0] - batch size, supported batch=1 only
   *    - shape[1] - col (image width)
   *    - shape[2] - row (image height)
   *    - shape[3] - channel, supported RGB(channel=3) only
   * \param dtype: the data type required by model, supported float32 only currently
   */
  YoloPreProcessor(std::array<int, 4> & shape, DataType dtype);
  ~YoloPreProcessor() = default;

  /**
   * \brief Convert a cv::Mat image to raw data and store in given buffer
   * \param img: Image in cv::Mat format
   * \param buf: Pointer to buffer to stored processed image data
   * \param size: Buffer size
   * \return
   *    On success: Return pointer pointing to buf
   *    On failure: Return nullptr
   */
  bool process(const cv::Mat & img, void * const buf, const int size);

private:
  std::array<int, 4> shape_;
  DataType dtype_;
};

}  // namespace qrb::yolo_processor

#endif  // _QRB_YOLO_PRE_PROCESS_CPP_