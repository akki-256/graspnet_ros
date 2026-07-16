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
torch torchvision --index-url https://download.pytorch.org/whl/cu126

## セットアップ
ビルドスクリプトの実行
```bash 
# graspnet-baselineのREADME.mdに則って必要なパッケージをインストールする
cd <path_to_this_package>
uv venv
source .venv/bin/activate

cd graspnet-baseline
uv pip install -r requirements.txt

# pytorchのインストール
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

cd pointnet2
python setup.py install

cd ../knn
python setup.py install

cd ../graspnetAPI
uv pip insatall .

# rosdepを利用して必要なros packageをインストールする(現時点では未対応)
# 必要に応じてrosdep install -r -y -i --from-paths .
# 
cd ~/llm-arm
colcon build --packages-select graspnet_ros
```
