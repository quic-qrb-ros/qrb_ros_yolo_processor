// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "bounding_box.hpp"

namespace qrb::yolo_processor
{
BoundingBox::BoundingBox(const std::initializer_list<float> & box, Format fmt)
{
  if (box.size() != 4) {
    throw std::invalid_argument("Bounding box expected 4 elements.");
  }
  format_ = fmt;
  std::copy(box.begin(), box.end(), arr_.begin());
}

float & BoundingBox::operator[](size_t index)
{
  if (index >= 4) {
    throw std::out_of_range("Index out of range");
  }
  return arr_.at(index);
}

BoundingBox BoundingBox::to_fmt(Format fmt)
{
  ConvFnPtr ptr = fn_conv_matrix_[static_cast<int>(format_)][static_cast<int>(fmt)];
  return (this->*ptr)();
}

BoundingBox::Format BoundingBox::get_fmt()
{
  return format_;
}

// Here follow the mechods to execute bounding box format convertion

BoundingBox BoundingBox::ToSelf()
{
  return *this;
}

BoundingBox BoundingBox::TLWHToCXYWH()
{
  float c_x = arr_[0] + 0.5 * arr_[2];
  float c_y = arr_[1] + 0.5 * arr_[3];
  float w = arr_[2];
  float h = arr_[3];
  return BoundingBox({ c_x, c_y, w, h }, Format::CXYWH);
}

BoundingBox BoundingBox::TLWHToTLBR()
{
  float tl_x = arr_[0];
  float tl_y = arr_[1];
  float br_x = arr_[2] + arr_[0];
  float br_y = arr_[3] + arr_[1];
  return BoundingBox({ tl_x, tl_y, br_x, br_y }, Format::TLBR);
}

BoundingBox BoundingBox::CXYWHToTLWH()
{
  float tl_x = arr_[0] - 0.5 * arr_[2];
  float tl_y = arr_[1] - 0.5 * arr_[3];
  float w = arr_[2];
  float h = arr_[3];
  return BoundingBox({ tl_x, tl_y, w, h }, Format::TLWH);
}

BoundingBox BoundingBox::CXYWHToTLBR()
{
  float tl_x = arr_[0] - 0.5 * arr_[2];
  float tl_y = arr_[1] - 0.5 * arr_[3];
  float br_x = tl_x + arr_[2];
  float br_y = tl_y + arr_[3];
  return BoundingBox({ tl_x, tl_x, br_x, br_y }, Format::TLBR);
}

BoundingBox BoundingBox::TLBRToTLWH()
{
  float tl_x = arr_[0];
  float tl_y = arr_[1];
  float w = arr_[2] - arr_[0];
  float h = arr_[3] - arr_[1];
  return BoundingBox({ tl_x, tl_y, w, h }, Format::TLWH);
}

BoundingBox BoundingBox::TLBRToCXYWH()
{
  float w = arr_[2] - arr_[0];
  float h = arr_[3] - arr_[1];
  float c_x = arr_[0] + 0.5 * w;
  float c_y = arr_[1] + 0.5 * h;
  return BoundingBox({ c_x, c_y, w, h }, Format::CXYWH);
}

}  // namespace qrb::yolo_processor
