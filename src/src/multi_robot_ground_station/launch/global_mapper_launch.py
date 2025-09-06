from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os

def generate_launch_description():
    # Path to the parameter file
    # find package diredtory
    param_file = os.path.join(
        os.getcwd(), 'src', 'ground_station', 'param', 'common.yaml'
    )
    print("get param_file:", param_file)
    return LaunchDescription([
        Node(
            package='ground_station',
            executable='global_mapper',
            name='global_mapper_node',
            output='screen',
            parameters=[param_file],
            # prefix=["gdbserver localhost:3000"]
        )
    ])