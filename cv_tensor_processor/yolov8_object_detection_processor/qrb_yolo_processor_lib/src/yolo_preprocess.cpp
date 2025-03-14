// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "yolo_preprocess.hpp"

namespace qrb::yolo_processor
{
YoloPreProcessor::YoloPreProcessor(std::array<int, 4> & shape, DataType dtype)
  : shape_(shape), dtype_(dtype)
{
  for (const int & element : shape) {
    if (element <= 0) {
      throw std::invalid_argument("invalid input shape");
    }
  }
  if (shape[0] > 1) {
    throw std::invalid_argument("support batch=1 only.");
  }
  if (shape[3] != 3) {
    throw std::invalid_argument("support channel=3 only.");
  }
}

bool YoloPreProcessor::process(const cv::Mat & img, void * const buf, const int size)
{
  if (buf == nullptr) {
    std::cerr << "buf nullptr" << std::endl;
    return false;
  }

  if (img.empty()) {
    std::cerr << "error, got empty image" << std::endl;
    return false;
  }

  const cv::Mat * mat_ptr = nullptr;
  cv::Mat resized_img;

  int model_w = shape_[1];
  int model_h = shape_[2];
  int model_ch = shape_[3];

  if (img.cols != model_w || img.rows != model_h) {
    cv::resize(img, resized_img, cv::Size(model_w, model_h));
    mat_ptr = &resized_img;
  } else {
    mat_ptr = &img;
  }

  int cvType = make_cvtype(dtype_, model_ch);

  // specify buffer for cv::Mat
  cv::Mat final_data(model_h, model_w, cvType, buf);

  // check if buf size is sufficient to store coverted data
  int data_size = final_data.total() * final_data.elemSize();
  if (data_size > size) {
    std::cerr << "err, insufficient buf size: " << size << ", requires: " << data_size << std::endl;
    return false;
  }

  mat_ptr->convertTo(final_data, cvType);
  final_data /= 255.0;

  return true;
}

}  // namespace qrb::yolo_processor