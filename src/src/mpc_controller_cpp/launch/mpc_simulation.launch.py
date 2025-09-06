from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory('mpc_controller_cpp'), 'config', 'controller_params.yaml'
    )
    return LaunchDescription([
        Node(
            package='mpc_controller_cpp',
            executable='casadi_controller_node',
            name='casadi_mpc_controller_node',
            output='screen',
            emulate_tty=True,
            parameters=[params_file],
        ),
    ])


