import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    venv_root = os.path.join('/root/llm_arm_ws/src/graspnet_ros/.venv')
    print(f"Virtual environment root: {venv_root}")
    venv_bin = os.path.join(venv_root, 'bin')
    
    node_env = os.environ.copy()
    
    # 2. venv用の環境変数を上書き・追加する
    node_env['VIRTUAL_ENV'] = venv_root
    node_env['PATH'] = f"{venv_bin}:{node_env.get('PATH', '')}"
    
    return LaunchDescription([
        Node(
            package='graspnet_ros',
            executable='graspnet_node',
            name='graspnet_node',
            output='screen',
            env=node_env
        )
    ])