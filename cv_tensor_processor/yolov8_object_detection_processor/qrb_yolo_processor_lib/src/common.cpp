// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "common.hpp"

#include "opencv2/opencv.hpp"

namespace qrb::yolo_processor
{
std::size_t get_size_of_type(DataType dataType)
{
  switch (dataType) {
    case DataType::INT8:
      return sizeof(int8_t);
    case DataType::UINT8:
      return sizeof(uint8_t);
    case DataType::INT16:
      return sizeof(int16_t);
    case DataType::UINT16:
      return sizeof(uint16_t);
    case DataType::INT32:
      return sizeof(int32_t);
    case DataType::UINT32:
      return sizeof(uint32_t);
    case DataType::INT64:
      return sizeof(int64_t);
    case DataType::UINT64:
      return sizeof(uint64_t);
    case DataType::FLOAT32:
      return sizeof(float);
    case DataType::FLOAT64:
      return sizeof(double);
    default:
      throw std::invalid_argument(
          "Unknown DataType: " + std::to_string(static_cast<int>(dataType)));
  }
}

int make_cvtype(DataType dtype, int channel)
{
  int cvType;
  switch (dtype) {
    case DataType::INT8:
      cvType = CV_MAKETYPE(CV_8S, channel);
      break;
    case DataType::UINT8:
      cvType = CV_MAKETYPE(CV_8U, channel);
      break;
    case DataType::INT16:
      cvType = CV_MAKETYPE(CV_16S, channel);
      break;
    case DataType::UINT16:
      cvType = CV_MAKETYPE(CV_16U, channel);
      break;
    case DataType::INT32:
      cvType = CV_MAKETYPE(CV_32S, channel);
      break;
    case DataType::FLOAT16:
      cvType = CV_MAKETYPE(CV_16F, channel);
      break;
    case DataType::FLOAT32:
      cvType = CV_MAKETYPE(CV_32F, channel);
      break;
    case DataType::FLOAT64:
      cvType = CV_MAKETYPE(CV_64F, channel);
      break;
    default:
      std::cerr << "data type " << static_cast<int>(dtype) << "not supported." << std::endl;
      return -1;
  }

  return cvType;
}

std::string get_tensor_shape_str(const Tensor & tensor)
{
  std::ostringstream oss;
  oss << tensor.name << ": dtype=" << static_cast<int>(tensor.dtype) << ", shape=[";

  for (size_t i = 0; i < tensor.shape.size(); ++i) {
    oss << tensor.shape[i];
    oss << ",";
  }
  oss << "]";
  std::cout << oss.str() << std::endl;
  return oss.str();
}

}  // namespace qrb::yolo_processor
