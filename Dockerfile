FROM dustynv/ros:humble-desktop-pytorch-l4t-r35.4.1

# ROSのGPGキーが期限切れの場合の対策
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# uvとその他のシステム依存パッケージのインストール
RUN apt-get update && apt-get install -y \
    python3-pip \
    pkg-config \
    libhdf5-serial-dev \
    python3-h5py \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install uv

# ワークスペースの設定
WORKDIR /workspace
COPY . /workspace/src/graspnet_ros

# setup.bashを自動で読み込むように設定
RUN echo "source /opt/ros/humble/install/setup.bash" >> ~/.bashrc
RUN echo "if [ -f /workspace/install/setup.bash ]; then source /workspace/install/setup.bash; fi" >> ~/.bashrc
RUN echo "export LD_PRELOAD=/usr/local/lib/python3.8/dist-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0" >> ~/.bashrc

# Pythonモジュールの事前ビルド（依存関係）
RUN apt-get update && apt-get install -y libhdf5-serial-dev pkg-config python3-h5py && rm -rf /var/lib/apt/lists/*

# graspnet-baselineのコードをコピーしてインストール
COPY graspnet-baseline /tmp/graspnet-baseline
ENV TORCH_CUDA_ARCH_LIST="7.2;8.7"
RUN cd /tmp/graspnet-baseline && \
    sed -i '/h5py/d' requirements.txt && \
    sed -i '/h5py/d' graspnetAPI/setup.py && \
    HDF5_DIR=/usr/lib/aarch64-linux-gnu/hdf5/serial/ uv pip install --system -r requirements.txt && \
    cd pointnet2 && uv pip install --system --no-build-isolation . && \
    cd ../knn && uv pip install --system --no-build-isolation . && \
    cd ../graspnetAPI && uv pip install --system . && \
    uv pip install --system ipython && \
    rm -rf /tmp/graspnet-baseline

CMD ["/bin/bash"]
