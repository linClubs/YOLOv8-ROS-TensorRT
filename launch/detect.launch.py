from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    action_robot_01 = Node(
        package="yolov8_trt",
        executable="detect",
        name='detect',
        output='screen',
        parameters=[{'topic_img': '/camera/color/image_raw'},
                    {'topic_res_img': '/detect/image_raw'},
                    {'weight_name': 'yolov8n.engine'}
                    ]
    )
    launch_description = LaunchDescription(
        [action_robot_01])
    
    return launch_description