#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Deprecated multi bringup removed in favor of parameter-only sim launch.
    return LaunchDescription([])


