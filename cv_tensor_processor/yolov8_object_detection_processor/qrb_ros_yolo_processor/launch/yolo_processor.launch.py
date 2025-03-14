# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer

def generate_launch_description():
    '''
    Generate launch desscription with multi components.
    '''
    label_file_arg = DeclareLaunchArgument(
        'label_file',
        default_value ='/opt/yolov8/coco8.yaml',
        description ='label files for yolov8 model'
    )

    score_thres = DeclareLaunchArgument(
        'score_thres',
        default_value = '0.5',
        description ='score(confidence) threshold value, between 0.0 ~ 1.0'
    )

    iou_thres = DeclareLaunchArgument(
        'iou_thres',
        default_value = '0.3',
        description='iou threshold value, between 0.0 ~ 1.0'
    )

    preprocess_node = ComposableNode(
        package='qrb_ros_yolo_processor',
        plugin='qrb_ros::yolo_processor::YoloPreProcessNode',
        name='yolo_preprocess_node'
    )

    postprocess_node = ComposableNode(
        package='qrb_ros_yolo_processor',
        plugin='qrb_ros::yolo_processor::YoloDetPostProcessNode',
        name='yolo_detection_postprocess_node',
        parameters=[
            { 'label_file': LaunchConfiguration('label_file')},
            { 'score_thres': LaunchConfiguration('score_thres')},
            { 'iou_thres': LaunchConfiguration('iou_thres')},
        ]
    )

    overlay_node = ComposableNode(
        package='qrb_ros_yolo_processor',
        plugin='qrb_ros::yolo_processor::YoloDetOverlayNode',
        name='yolo_detection_overlay_node'
    )

    container = ComposableNodeContainer(
        name='yolo_node_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions = [
            preprocess_node,
            postprocess_node,
            overlay_node
        ],
        output='screen'
    )

    return LaunchDescription([
        label_file_arg,
        score_thres,
        iou_thres,
        container,
    ])
