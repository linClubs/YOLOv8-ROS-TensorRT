from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="yolov8_trt",
            executable="pose",
            name='pose',
            output='screen',
            parameters=[{'topic_img': '/camera/color/image_raw'},
                        {'topic_res_img': '/pose/image_raw'},
                        {'weight_name': 'yolov8n-pose.engine'}
                        ]
            )
    ])

