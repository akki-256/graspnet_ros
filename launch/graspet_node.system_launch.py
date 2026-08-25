from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="graspnet_ros",
                executable="graspnet_node",
                name="graspnet_node",
                output="screen",
            )
        ]
    )
