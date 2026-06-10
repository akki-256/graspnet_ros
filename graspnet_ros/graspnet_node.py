#!/usr/bin/env python3

from pathlib import Path
import time

import numpy as np
import scipy.io as scio

import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from graspnet_ros.graspnet_detector import GraspNetDetector
from graspnet_ros.msg import GraspPattern

class GraspNetNode(Node):
    """
    GraspNetののROS2ノードクラス
    
    publishers:
    - self.grasp_publisher: 推論した把持パターンを出版するPublisher
    
    subscribers:
    - self.detection_image_subscriber: カメラから対象が映る部分のRGB画像を購読するSubscriber
    - self.detection_depth_subscriber: カメラから対象が映る部分の深度画像を購読するSubscriber
    
    members:
    - grasp_detector: GraspNetの推論を行うクラスのインスタンス
    - model : GraspNetのモデルを格納する変数
    - grasp_pattern: 推論した把持パターンを格納する変数
    - point_cloud: 深度画像から生成したポイントクラウドを格納する変数
    - camera_info: カメラの内部パラメータを格納する変数
    - bridge: CvBridgeのインスタンス（ROSの画像メッセージとOpenCV形式の画像の変換に使用）
    - package_root: graspnet_rosパッケージのルートディレクトリのパスを格納する変数
    - meta_path: meta.matファイルのパスを格納する変数
    - ts: message_filters.ApproximateTimeSynchronizerのインスタンス（RGB画像と深度画像の同期に使用）
    
    methods:
    - grasp_publisher: GraspPatternメッセージを出版するPublisher
    - detection_image_subscriber: RGB画像を購読するSubscriber
    - detection_depth_subscriber: 深度画像を購読するSubscriber
    - _create_point_cloud_from_depth: 深度画像からポイントクラウドを生成する関数
    
    """
    def __init__(self):
        super().__init__('graspnet_node')
        
        # パッケージのルートディレクトリとmeta.matのパスを設定
        self.package_root = Path(__file__).resolve().parents[1]
        self.meta_path = self.package_root / 'meta.mat' # TODO - realsenseカメラの内部パラメータは購読できるため，変更が必要
        if not self.meta_path.exists():
            raise FileNotFoundError(f'meta.mat not found: {self.meta_path}')

        meta = scio.loadmat(str(self.meta_path))
        intrinsic_matrix = meta['intrinsic_matrix']
        factor_depth = float(np.asarray(meta['factor_depth']).squeeze())
        self.camera_info = CameraInfo(
            width=1280.0,
            height=720.0,
            fx=float(intrinsic_matrix[0][0]),
            fy=float(intrinsic_matrix[1][1]),
            cx=float(intrinsic_matrix[0][2]),
            cy=float(intrinsic_matrix[1][2]),
            scale=factor_depth,
        )
        
        # Publisher
        self.grasp_publisher = self.create_publisher(GraspPattern, 'grasp_pattern', 10)
        
        # Subscriber
        self.detection_image_subscriber = message_filters.Subscriber(self, Image, '/camera/detection/image')
        self.detection_depth_subscriber = message_filters.Subscriber(self, Image, '/camera/detection/depth')
        self.ts = message_filters.ApproximateTimeSynchronizer([self.detection_image_subscriber, self.detection_depth_subscriber], queue_size=10, slop=0.1)
        self.ts.registerCallback(self._get_grasp)
        self.bridge = CvBridge()
        
        self.grasp_detector = GraspNetDetector()

        # rate limit / duplicate header guard
        self.declare_parameter('min_publish_interval', 0.0)
        self.min_publish_interval = float(self.get_parameter('min_publish_interval').value)
        self._last_publish_time = 0.0
        self._last_header_stamp = None
        
        self.get_logger().info("GraspNetNode has been started.")
        
    def _get_grasp(self, image_msg, depth_msg):
        """
        GraspNetの推論を実行する関数
        """
        
        # 同じ画像に対して複数回推論しないようにするためのガード
        now = time.time()
        header = image_msg.header
        stamp = (int(header.stamp.sec), int(header.stamp.nanosec))
        if self._last_header_stamp == stamp and (now - self._last_publish_time) < self.min_publish_interval:
            return
        if (now - self._last_publish_time) < self.min_publish_interval:
            return

        # ROSの画像メッセージをOpenCV形式の画像に変換
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        # 深度画像から点群データを生成
        point_cloud = self._create_point_cloud_from_depth(depth_image)
        
        # GraspNetの推論を実行して把持パターンを取得
        best_grasp = self.grasp_detector.get_best_grasp(point_cloud)
        grasp_pattern = self._grasp_to_msg(best_grasp, image_msg.header)

        # 把持パターンを出版
        self.grasp_publisher.publish(grasp_pattern)
        
        try:
            score = float(best_grasp.score) if best_grasp is not None else -1.0
        except Exception:
            score = -1.0
        self.get_logger().info(f'Published grasp pattern with score: {score:.4f}')
        
        # ヘッダースタンプを更新
        self._last_publish_time = now
        self._last_header_stamp = stamp

    def _create_point_cloud_from_depth(self, depth_image):
        """
        深度画像から点群データを生成する関数
        """
        
        depth = depth_image.astype(np.float32)
        if depth_image.dtype != np.float32:
            depth = depth / self.camera_info.scale

        height, width = depth.shape[:2]
        xmap = np.arange(width)
        ymap = np.arange(height)
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth
        points_x = (xmap - self.camera_info.cx) * points_z / self.camera_info.fx
        points_y = (ymap - self.camera_info.cy) * points_z / self.camera_info.fy
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        return cloud.reshape([-1, 3])

    def _grasp_to_msg(self, grasp, header):
        """ 
        GraspNetの把持パターンをGraspPatternメッセージに変換する関数 
        """
        
        msg = GraspPattern()
        msg.header = header

        if grasp is None:
            msg.score = -1.0
            msg.width = 0.0
            msg.height = 0.0
            msg.depth = 0.0
            msg.rotation_matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            msg.translation = [0.0, 0.0, 0.0]
            msg.object_id = -1
            return msg

        msg.score = float(grasp.score)
        msg.width = float(grasp.width)
        msg.height = float(grasp.height)
        msg.depth = float(grasp.depth)
        msg.rotation_matrix = [float(value) for value in grasp.rotation_matrix.reshape(-1)]
        msg.translation = [float(value) for value in grasp.translation.reshape(-1)]
        msg.object_id = int(grasp.object_id)
        return msg
    
class CameraInfo():
    """ Camera intrisics for point cloud creation. """
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


def main():
    rclpy.init()
    node = GraspNetNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
    