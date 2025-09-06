from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os

def generate_launch_description():
    # Path to the parameter file
    # find package share directory (installed path)
    param_file = os.path.join(
        get_package_share_directory('ground_station'), 'param', 'optitrack_calib.yaml'
    )
    print("get param_file:", param_file)
    return LaunchDescription([
        Node(
            package='ground_station',
            executable='optitrack_calib',
            name='optitrack_calib_node',
            # prefix=["gdbserver localhost:3000"],
            output='screen',
            parameters=[param_file],
            # prefix=["gdbserver localhost:3000"]
        )
    ])