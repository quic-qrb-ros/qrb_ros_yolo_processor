#include <cv_bridge/cv_bridge.hpp>

#include <cv_tensor_common_process.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <vector>

namespace qrb_ros::cv_tensor_common_process
{
CvTensorCommonProcessNode::CvTensorCommonProcessNode(const rclcpp::NodeOptions & options)
  : Node("cv_tensor_common_processor", options)
{
  // ros parameter handling
  this->declare_parameter<int>("resize_width", 0);
  this->declare_parameter<int>("resize_height", 0);
  this->declare_parameter<bool>("normalize", true);
  this->declare_parameter<std::string>("tensor_fmt", "nhwc");
  this->declare_parameter<std::string>("data_type", "float32");

  this->get_parameter("resize_width", resize_width_);
  this->get_parameter("resize_height", resize_height_);
  this->get_parameter("normalize", normalize_);
  this->get_parameter("tensor_fmt", tensor_fmt_);
  this->get_parameter("data_type", data_type_);

  // map: string <--> CV data type
  data_type_map_["uint8"] = CV_8UC3;
  data_type_map_["float32"] = CV_32FC3;
  data_type_map_["float64"] = CV_64FC3;

  // map: string <--> Enum
  tensor_fmt_map_["nhwc"] = EnumTensorFmt::FMT_NHWC;
  tensor_fmt_map_["nchw"] = EnumTensorFmt::FMT_NCHW;

  // params check
  if (resize_width_ <= 0 || resize_height_ <= 0) {
    throw std::invalid_argument(
        "CvTensorCommonProcessNode: Invalid resize value: " + std::to_string(resize_width_) + "," +
        std::to_string(resize_height_));
  }

  if (tensor_fmt_map_.find(tensor_fmt_) == tensor_fmt_map_.end()) {
    std::ostringstream oss;
    oss << "CvTensorCommonProcessNode: Invalid value for tensor_fmt: " << data_type_ << ", ";
    oss << "support fmt: nhwc, nchw" << std::endl;
    throw std::invalid_argument(oss.str());
  }

  if (data_type_map_.find(data_type_) == data_type_map_.end()) {
    std::ostringstream oss;
    oss << "CvTensorCommonProcessNode: Invalid value for data_type: " << data_type_ << ", ";
    oss << "support type: uint8, float32, float64" << std::endl;
    throw std::invalid_argument(oss.str());
  }

  switch (data_type_map_[data_type_]) {
    case CV_8UC3:
      data_type_val_ = 0;
      break;
    case CV_32FC3:
      data_type_val_ = 2;
      break;
    case CV_64FC3:
      data_type_val_ = 3;
      break;
  }

  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>("input_image", 10,
      std::bind(&CvTensorCommonProcessNode::msg_callback, this, std::placeholders::_1));

  tensor_pub_ =
      this->create_publisher<qrb_ros_tensor_list_msgs::msg::TensorList>("encoded_image", 10);
}

void CvTensorCommonProcessNode::msg_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  RCLCPP_DEBUG(this->get_logger(), "msg_callback enter");

  try {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    cv::Mat & image = cv_ptr->image;

    cv::Mat processed_image = process_image(image);

    qrb_ros_tensor_list_msgs::msg::TensorList tensor_list = createTensorList(processed_image);
    tensor_list.header = msg->header;  // keep timestamp as is
    tensor_pub_->publish(tensor_list);
  } catch (std::invalid_argument & e) {
    RCLCPP_ERROR(this->get_logger(), "%s", e.what());
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  } catch (...) {
    RCLCPP_ERROR(this->get_logger(), "Unexpected error in msg_callback");
  }

  RCLCPP_DEBUG(this->get_logger(), "msg_callback exit");
}

cv::Mat CvTensorCommonProcessNode::process_image(const cv::Mat & image)
{
  cv::Mat processed_image = image.clone();

  if (image.channels() != 3) {
    throw std::invalid_argument("Invalid input image");
  }

  // resize
  if (image.cols != resize_width_ && image.rows != resize_height_) {
    cv::resize(image, processed_image, cv::Size(resize_width_, resize_height_));
  }

  // data type convert
  if (data_type_map_[data_type_] != processed_image.type()) {
    processed_image.convertTo(processed_image, data_type_map_[data_type_]);
  }

  // normalize
  if (normalize_) {
    double scale_factor = 255.0;
    processed_image /= scale_factor;
  }

  return processed_image;
}

qrb_ros_tensor_list_msgs::msg::TensorList CvTensorCommonProcessNode::createTensorList(
    const cv::Mat & image)
{
  qrb_ros_tensor_list_msgs::msg::Tensor tensor;
  size_t payload_size = image.total() * image.elemSize();
  tensor.name = "image_tensor";
  tensor.data_type = data_type_val_;
  tensor.shape.push_back(1);         // batch size
  tensor.data.resize(payload_size);  // data size in byte

  switch (tensor_fmt_map_[tensor_fmt_]) {
    case EnumTensorFmt::FMT_NHWC: {
      // [batch_size, height, width, channels]
      tensor.shape.push_back(image.rows);
      tensor.shape.push_back(image.cols);
      tensor.shape.push_back(image.channels());
      std::memcpy(tensor.data.data(), image.data, payload_size);
      break;
    }

    case EnumTensorFmt::FMT_NCHW: {
      // [batch_size, channels, height, width]
      tensor.shape.push_back(image.channels());
      tensor.shape.push_back(image.rows);
      tensor.shape.push_back(image.cols);

      // split rgb channel to single
      std::vector<cv::Mat> split_channels;
      cv::split(image, split_channels);

      size_t total_size = 0;
      for (const auto & channel : split_channels) {
        total_size += channel.total() * channel.elemSize();
      }

      if (total_size != payload_size) {
        throw std::runtime_error("invalid copy size");
      }

      size_t copied_size = 0;
      for (const auto & channel : split_channels) {
        size_t channal_size = channel.total() * channel.elemSize();
        std::memcpy(tensor.data.data() + copied_size, channel.data, channal_size);
        copied_size += channal_size;
      }
      break;
    }
    default:
      break;
  }

  qrb_ros_tensor_list_msgs::msg::TensorList msg;
  msg.tensor_list.push_back(tensor);
  return msg;
}

}  // namespace qrb_ros::cv_tensor_common_process

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(qrb_ros::cv_tensor_common_process::CvTensorCommonProcessNode)