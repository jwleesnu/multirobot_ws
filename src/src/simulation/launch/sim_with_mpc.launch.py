from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Parameter-only launch for simulation node
    sim_param_file = os.path.join(
        get_package_share_directory('simulation'), 'config', 'simulation_params.yaml'
    )
    return LaunchDescription([
        Node(
            package='simulation',
            executable='simulation',
            name='kinematics_simulation',
            output='screen',
            emulate_tty=True,
            parameters=[sim_param_file],
        ),
    ])