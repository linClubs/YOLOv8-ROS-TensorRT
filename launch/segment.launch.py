from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    seg_node = Node(
            package="yolov8_trt",
            executable="segment",
            name='segment',
            output='screen',
            parameters=[{'topic_img': '/camera/color/image_raw'},
                        {'topic_res_img': '/segment/image_raw'},
                        {'weight_name': 'yolov8n-seg.engine'}
                        ]
            )
    return LaunchDescription([
        seg_node
    ])
