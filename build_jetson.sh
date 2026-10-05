#!/bin/bash
set -e

echo "======================================"
echo " Building graspnet_ros on Jetson"
echo "======================================"

# ROS2の環境変数を読み込む
source /opt/ros/humble/install/setup.bash

cd /workspace/src/graspnet_ros

# 1. 依存Pythonパッケージのインストール
# ※ PyTorchはL4T環境に最適化されたものがインストール済みのため、requirements.txtからのみインストール
echo "Installing python dependencies..."
cd graspnet-baseline
uv pip install --system -r requirements.txt

# 2. カスタムCUDA拡張モジュールのビルド (pointnet2, knn)
echo "Building pointnet2..."
cd pointnet2
uv pip install --system --no-build-isolation .

echo "Building knn..."
cd ../knn
uv pip install --system --no-build-isolation .

# 3. graspnetAPIのインストール
echo "Installing graspnetAPI..."
cd ../graspnetAPI
uv pip install --system .
uv pip install --system ipython

# 4. ROS2パッケージのビルド
echo "Building ROS2 package (graspnet_ros)..."
cd /workspace
colcon build --packages-select graspnet_ros --symlink-install

echo "======================================"
echo " Build Completed Successfully!"
echo " Please run: source /workspace/install/setup.bash"
echo "======================================"
