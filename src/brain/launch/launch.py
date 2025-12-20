# coding: utf8

import launch
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

# Python 함수 안에서 context를 사용해 동적으로 설정
def handle_configuration(context, *args, **kwargs):

    # launch 인자 vision_config_path에서 vision 설정 디렉토리를 가져옴
    vision_config_dir = context.perform_substitution(LaunchConfiguration('vision_config_path'))
    vision_config_file = os.path.join(vision_config_dir, 'vision.yaml')
    vision_config_local_file = os.path.join(vision_config_dir, 'vision_local.yaml')

    config_path = os.path.join(os.path.dirname(__file__), '../config')
    config_file = os.path.join(config_path, 'config.yaml') 
    config_local_file = os.path.join(config_path, 'config_local.yaml') 

    behavior_trees_dir = os.path.join(os.path.dirname(__file__), '../behavior_trees')
    def make_tree_path(name):
        if not name.endswith('.xml'):
            name += '.xml'
        return os.path.join(behavior_trees_dir, name)
    tree = context.perform_substitution(LaunchConfiguration('tree'))
    tree_path = make_tree_path(tree)

    # 여기서 정의한 config는 config_file에 같은 항목이 존재하면 그 값을 덮어써서,
    # launch 실행 시 파라미터를 신속하게 변경할 수 있게 해주며,
    # 매번 config.yaml 파일을 수정하지 않아도 되도록 설계
    
    config = {
            # 로드할 행동 트리 파일 지정
            "tree_file_path": tree_path,
            "vision_config_path": vision_config_file,
            "vision_config_local_path": vision_config_local_file,
    }

    role = context.perform_substitution(LaunchConfiguration('role'))
    if not role == '':
        config['game.player_role'] = role

    sim = context.perform_substitution(LaunchConfiguration('sim'))
    if sim in ['true', 'True', '1']:
        config['use_sim_time'] = True

    disableLog = context.perform_substitution(LaunchConfiguration('disable_log'))
    if disableLog in ['true', 'True', '1']:
        config['rerunLog.enable_file'] = False
        config['rerunLog.enable_tcp'] = False

    disableCom = context.perform_substitution(LaunchConfiguration('disable_com'))
    if disableCom in ['true', 'True', '1']:
        config['enable_com'] = False

    return [
        Node(
            package ='brain',
            executable='brain_node',
            output='screen',
            parameters=[
                config_file,
                config_local_file,
                config
            ]
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'vision_config_path',
            default_value=os.path.join(os.path.dirname(__file__), '../../../../vision/share/vision/config'),
            description='Directory containing vision.yaml and vision_local.yaml'
        ),
        # ros2 launch brain launch.py param:=value 형식으로 전달해야 하는 파라미터들은
        # 여기에서 DeclareLaunchArgument로 선언한 뒤,
        # handle_configuration 함수에서 처리
        
        DeclareLaunchArgument(
            'tree', 
            default_value='game.xml',
            description='Specify behavior tree file name. DO NOT include full path, file should be in src/brain/config/behavior_trees'
        ),
        DeclareLaunchArgument(
            'role', 
            default_value='',
            description='// config.yaml의 game.player_role을 덮어쓰려면 launch 시 role:=striker로 지정'
        ),
        DeclareLaunchArgument(
            'sim', 
            default_value='false',
            description='시뮬레이션 환경에서 실행 중인지 여부'
        ),
        DeclareLaunchArgument(
            'disable_log', 
            default_value='false',
            description='파일 로그 기록을 강제로 비활성화'
        ),
        DeclareLaunchArgument(
            'disable_com', 
            default_value='false',
            description='통신 기능을 강제로 비활성화'
        ),
        OpaqueFunction(function=handle_configuration) # handle_configuration 함수로 전달하여 추가 설정을 처리
    ])
