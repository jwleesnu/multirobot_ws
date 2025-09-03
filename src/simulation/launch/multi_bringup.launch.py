#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='simulation',
            executable='simulation',
            name='simulation',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='mpc_controller',
            executable='casadi_controller',
            name='casadi_controller',
            output='screen',
            emulate_tty=True,
            parameters=['/home/jaewoo/ros2_workspace/multirobot_ws/src/mpc_controller/config/controller_params.yaml'],
        ),
        Node(
            package='reference_generator',
            executable='simulation_reference',
            name='simulation_reference',
            output='screen',
            emulate_tty=True,
        ),
    ])


