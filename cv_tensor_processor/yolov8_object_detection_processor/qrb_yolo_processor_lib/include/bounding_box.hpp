// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#ifndef _QRB_YOLO_PROCESSOR_BOUNDING_BOX_CPP_
#define _QRB_YOLO_PROCESSOR_BOUNDING_BOX_CPP_

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace qrb::yolo_processor
{
/**
 * \brief Class to describe bounding box, and provide methods to operate box.
 */
class BoundingBox
{
public:
  enum class Format
  {
    TLWH,
    CXYWH,
    TLBR,
    MAX
  };

  BoundingBox(const std::initializer_list<float> & box, Format fmt);
  ~BoundingBox() = default;

  /**
   * \brief Create a new bounding box with specific box format
   * \param fmt desired format of the new bounding box
   */
  BoundingBox to_fmt(Format fmt);
  Format get_fmt();
  float & operator[](size_t index);

private:
  std::array<float, 4> arr_;
  Format format_;

  using ConvFnPtr = BoundingBox (BoundingBox::*)();
  /**
   * arrary of pointers pointing to bounding-box format transfer functions
   * fn_conv_matrix_[from_fmt][to_fmt] stores the pointer of method which is
   * used to convert box from <from_fmt> to <to_fmt>
   */
  ConvFnPtr fn_conv_matrix_[static_cast<int>(Format::MAX)][static_cast<int>(Format::MAX)] = {
    { &BoundingBox::ToSelf, &BoundingBox::TLWHToCXYWH, &BoundingBox::TLWHToTLBR },
    { &BoundingBox::CXYWHToTLWH, &BoundingBox::ToSelf, &BoundingBox::CXYWHToTLBR },
    { &BoundingBox::TLBRToTLWH, &BoundingBox::TLBRToCXYWH, &BoundingBox::ToSelf },
  };

  BoundingBox ToSelf();

  // TLWH(0) <--> TLWH(0)
  // ToSelf

  // TLWH(0) <--> CXYWH(1)
  BoundingBox TLWHToCXYWH();

  // TLWH(0) <--> TLBR(2)
  BoundingBox TLWHToTLBR();

  // CXYWH(1) <--> TLWH(0)
  BoundingBox CXYWHToTLWH();

  // CXYWH(1) <--> CXYWH(1)
  // ToSelf

  // CXYWH(1) <--> TLBR(2)
  BoundingBox CXYWHToTLBR();

  // TLBR(2) <--> TLWH(0)
  BoundingBox TLBRToTLWH();

  // TLBR(2) <--> CXYWH(1)
  BoundingBox TLBRToCXYWH();

  // TLBR(2) <--> TLBR(2)
  // ToSelf
};

}  // namespace qrb::yolo_processor

#endif  // _QRB_YOLO_PROCESSOR_BOUNDING_BOX_CPP_