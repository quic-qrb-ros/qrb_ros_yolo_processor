// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#ifndef _QRB_YOLO_PROCESSOR_COMMON_HPP_
#define _QRB_YOLO_PROCESSOR_COMMON_HPP_

#include <string>
#include <vector>

#include "bounding_box.hpp"

namespace qrb::yolo_processor
{
/**
 * \brief Tensor data type enumeration
 */
enum class DataType
{
  INT8,
  UINT8,
  INT16,
  UINT16,
  INT32,
  UINT32,
  INT64,
  UINT64,
  FLOAT16,
  FLOAT32,
  FLOAT64,
};

/**
 * \brief get size of basic type represented by given datatype enumeration
 * \param dataType: enum value of specific data type
 */
std::size_t get_size_of_type(DataType dataType);

/**
 * \brief struct to describe a tensor
 */
struct Tensor
{
  std::vector<uint8_t> * p_vec;  // pointer to vector that stores tensor byte stream
  std::vector<uint32_t> shape;   // shape of tensor
  std::string name;              // name of tensor
  DataType dtype;                // data type of tensor
};
std::string get_tensor_shape_str(const Tensor & tensor);

/**
 * \brief YOLO instance info structure
 */
struct YoloInstance
{
  BoundingBox bbox;
  float score;
  std::string label;
  std::vector<uint8_t> mask;

  // constructor
  YoloInstance(float x,
      float y,
      float w,
      float h,
      BoundingBox::Format box_fmt,
      float score,
      const std::string & label)
    : bbox({ x, y, w, h }, box_fmt), score(score), label(label)
  {
  }

  YoloInstance(float x,
      float y,
      float w,
      float h,
      BoundingBox::Format box_fmt,
      float score,
      const std::string & label,
      const std::vector<uint8_t> & mask)
    : bbox({ x, y, w, h }, box_fmt), score(score), label(label), mask(mask)
  {
  }

  YoloInstance(std::initializer_list<float> array,
      BoundingBox::Format box_fmt,
      float score,
      const std::string & label)
    : bbox(array, box_fmt), score(score), label(label)
  {
  }
};

/**
 * \brief make CV_TYPE as per DataType::dtype and channel number
 * \param dtype data type
 * \param channel channel number
 */
int make_cvtype(DataType dtype, int channel);

}  // namespace qrb::yolo_processor

#endif  // _QRB_YOLO_PROCESSOR_COMMON_HPP_