// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "yolo_seg_overlay.hpp"

#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

namespace qrb::yolo_processor
{
void YoloSegOverlay::draw_inplace(std::vector<YoloInstance> & detections, cv::Mat & img)
{
  // overlay bounding box info
  for (auto & det : detections) {
    // TLWH format
    BoundingBox::Format box_fmt = BoundingBox::Format::TLWH;
    BoundingBox tlwh_box = det.bbox.to_fmt(box_fmt);
    float x = tlwh_box[0];
    float y = tlwh_box[1];
    float w = tlwh_box[2];
    float h = tlwh_box[3];

    cv::rectangle(img, cv::Rect(x, y, w, h), cv::Scalar(0, 0, 255), 1);

    std::stringstream text;
    text << det.label << ": " << std::fixed << std::setprecision(2) << det.score;
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

  // overlay mask

  // merge all mask together
  cv::Mat merged_mask(160, 160, CV_8UC1, cv::Scalar(0));
  for (auto & det : detections) {
    cv::Mat mask(160, 160, CV_8UC1, det.mask.data());
    cv::add(mask, merged_mask, merged_mask);
  }

  // scale mask to 640x640
  cv::Mat resized_image;
  cv::resize(merged_mask, resized_image, cv::Size(), 4, 4, cv::INTER_LINEAR);

  // create a color mask
  cv::Mat colorMask = cv::Mat::zeros(resized_image.size(), CV_8UC3);
  cv::Vec3b color(0, 0, 255);  // red
  for (int i = 0; i < resized_image.rows; ++i) {
    for (int j = 0; j < resized_image.cols; ++j) {
      if (resized_image.at<uchar>(i, j) != 0) {
        colorMask.at<cv::Vec3b>(i, j) = color;
      }
    }
  }
  double beta = 0.5;
  cv::addWeighted(img, 1.0, colorMask, beta, 0.0, img);
}

}  // namespace qrb::yolo_processor