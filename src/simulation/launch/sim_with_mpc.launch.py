from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    sim_params = '$(find-pkg-share simulation)/config/simulation_params.yaml'
    mpc_params = '$(find-pkg-share mpc_controller)/config/controller_params.yaml'
    return LaunchDescription([
        Node(
            package='simulation',
            executable='simulation',
            name='kinematics_simulation',
            parameters=[sim_params]
        ),
        Node(
            package='mpc_controller',
            executable='controller_node',
            name='mpc_controller_node',
            parameters=[mpc_params]
        ),
    ])