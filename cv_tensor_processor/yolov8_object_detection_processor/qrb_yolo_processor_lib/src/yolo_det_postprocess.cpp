// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "yolo_det_postprocess.hpp"

#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

namespace qrb::yolo_processor
{
YoloDetPostProcessor::YoloDetPostProcessor(const std::string & label_file,
    float score_thres,
    float iou_thres,
    float eta,
    int top_k)
  : score_thres_(score_thres), iou_thres_(iou_thres), eta_(eta), top_k_(top_k)
{
  // args range check
  auto out_of_range = [](float value) { return value <= 0.0f || value >= 1.0f; };

  if (out_of_range(score_thres_) || out_of_range(iou_thres_)) {
    throw std::invalid_argument("Err: thres out of range..");
  }

  // init label map from yaml
  try {
    YAML::Node label_yml = YAML::LoadFile(label_file);
    const YAML::Node & nd_name = label_yml["names"];

    for (YAML::const_iterator it = nd_name.begin(); it != nd_name.end(); ++it) {
      label_map_[it->first.as<int>()] = it->second.as<std::string>();
    }

  } catch (const YAML::Exception & e) {
    std::cerr << "YAML Exception: " << e.what() << std::endl;
    label_map_.clear();
  }
}

void YoloDetPostProcessor::process(const std::vector<Tensor> & tensors,
    std::vector<YoloInstance> & instances)
{
  std::ostringstream oss;

  /**
   * Input params check
   *
   * Yolo TFLite model output 3 tensors shown below, N means the detected object count in one
   * inference cycle, and is a fixed value up to model
   *   - tensors[0] --> bbox tensor with shape float32[1,N,4], store the coordinate info of bouding
   *     box for N objects
   *   - tensors[1] --> score tensor with shape float32[1,N], store confidence of N objeces
   *   - tensors[2] --> label tensor with shape float32[1,N], store label index of N objeces
   */
  if (tensors.size() != 3) {
    oss << "Expect 3 tensors but got " << tensors.size();
    throw std::invalid_argument(oss.str());
  }

  const Tensor & tensor_bbox = tensors[0];
  const Tensor & tensor_score = tensors[1];
  const Tensor & tensor_label = tensors[2];

  bool err = true;
  do {
    // tensor_bbox shape check
    if (tensor_bbox.shape[0] != 1 || tensor_bbox.shape[2] != 4 ||
        tensor_bbox.dtype != DataType::FLOAT32) {
      oss << "bad tensor_bbox: (" << tensor_bbox.shape[0] << "," << tensor_bbox.shape[1] << ","
          << tensor_bbox.shape[2] << "), dtype=" << static_cast<int>(tensor_bbox.dtype);
      break;
    }

    // tensor_score shape check
    if (tensor_score.shape[0] != 1 || tensor_score.dtype != DataType::FLOAT32) {
      oss << "bad tensor_score: (" << tensor_score.shape[0] << "," << tensor_score.shape[1] << "), "
          << "dtype: " << static_cast<int>(tensor_score.dtype);
      break;
    }

    // tensor_label shape check
    if (tensor_label.shape[0] != 1 || tensor_label.dtype != DataType::FLOAT32) {
      oss << "bad tensor_label: (" << tensor_label.shape[0] << "," << tensor_label.shape[1] << "), "
          << "dtype: " << static_cast<int>(tensor_label.dtype);
      break;
    }

    // obj count check
    if (tensor_bbox.shape[1] != tensor_score.shape[1] ||
        tensor_bbox.shape[1] != tensor_label.shape[1]) {
      oss << "Inconsistent obj count, "
          << " tensor_bbox=" << tensor_bbox.shape[1] << " tensor_score=" << tensor_score.shape[1]
          << " tensor_label=" << tensor_label.shape[1];
      break;
    }
    err = false;
  } while (0);

  if (err) {
    throw std::invalid_argument(oss.str());
  }

  // tensor data parse
  std::vector<std::string> vec_class;
  std::vector<float> vec_score;
  std::vector<cv::Rect> vec_bbox;

  float(*ptr_bbox)[4] = reinterpret_cast<float(*)[4]>(tensor_bbox.p_vec->data());
  float * ptr_score = reinterpret_cast<float *>(tensor_score.p_vec->data());
  float * ptr_label = reinterpret_cast<float *>(tensor_label.p_vec->data());

  uint32_t obj_cnt = tensor_bbox.shape[1];
  for (uint32_t i = 0; i < obj_cnt; ++i) {
    float score = ptr_score[i];
    // filter data to reduce computing in NMS
    if (score < score_thres_) {
      continue;
    }

    // get label tring
    std::string label;
    int label_idx = ptr_label[i];
    try {
      label = label_map_.at(label_idx);
    } catch (const std::out_of_range & e) {
      label = "unknown";
    }

    // model returns TLBR bbox, convert to TLWH
    BoundingBox box =
        BoundingBox({ ptr_bbox[i][0], ptr_bbox[i][1], ptr_bbox[i][2], ptr_bbox[i][3] },
            BoundingBox::Format::TLBR)
            .to_fmt(BoundingBox::Format::TLWH);
    vec_bbox.push_back(cv::Rect(box[0], box[1], box[2], box[3]));
    vec_class.push_back(label);
    vec_score.push_back(score);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(vec_bbox, vec_score, score_thres_, iou_thres_, indices);

  for (std::vector<int>::iterator it = indices.begin(); it != indices.end(); ++it) {
    YoloInstance instance(vec_bbox[*it].x, vec_bbox[*it].y, vec_bbox[*it].width,
        vec_bbox[*it].height, BoundingBox::Format::TLWH, vec_score[*it], vec_class[*it]);

    instances.push_back(instance);
  }
}

}  // namespace qrb::yolo_processor