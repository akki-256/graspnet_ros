# GraspNet_ros

## discrption
GraspNetAPIおよびgraspnet-baselineのros2 wrapper

## 環境
### OS
- docker_container: nvidia/cuda:12.8.0-runtime-ubuntu22.04
- ros: ros2_humble
- python: 3.10
- python_package_manager: uv

### ros_package
- realsense2_ros
- cv_bridge

### python_package
// - torch(TODO version)

## セットアップ
1. graspnet-baselineのREADME.mdに則って必要なパッケージをインストールする
2. rosdepを利用して必要なros packageをインストールする(現時点では未対応)
