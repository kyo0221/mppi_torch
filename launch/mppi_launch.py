from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('mppi_torch')
    config_file = os.path.join(pkg_share, 'config', 'mppi_params.yaml')

    mppi_node = Node(
        package='mppi_torch',
        executable='mppi_node',
        name='mppi_node',
        output='screen',
        parameters=[config_file]
    )

    return LaunchDescription([
        mppi_node
    ])
