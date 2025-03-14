// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "yolo_det_overlay.hpp"

#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

namespace qrb::yolo_processor
{
void YoloDetOverlay::draw_inplace(std::vector<YoloInstance> & instances, cv::Mat & img)
{
  for (auto & instance : instances) {
    // TLWH format
    BoundingBox::Format box_fmt = BoundingBox::Format::TLWH;
    BoundingBox tlwh_box = instance.bbox.to_fmt(box_fmt);
    float x = tlwh_box[0];
    float y = tlwh_box[1];
    float w = tlwh_box[2];
    float h = tlwh_box[3];

    cv::rectangle(img, cv::Rect(x, y, w, h), cv::Scalar(0, 0, 255), 1);

    std::stringstream text;
    text << instance.label << ": " << std::fixed << std::setprecision(2) << instance.score;
    int baseline = 0;
    double fontScale = 0.5;
    int thickness = 1;

    cv::Size text_size =
        cv::getTextSize(text.str(), cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
    cv::Rect background(
        x, y - text_size.height + baseline, text_size.width, text_size.height + baseline);
    cv::rectangle(img, background, cv::Scalar(0, 0, 255), cv::FILLED);
    cv::putText(img, text.str(), cv::Point(x, y + baseline), cv::FONT_HERSHEY_SIMPLEX, fontScale,
        cv::Scalar(255, 255, 255), thickness);
  }
}

}  // namespace qrb::yolo_processor