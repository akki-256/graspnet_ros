#!/usr/bin/env python3

import time

import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from graspnet_ros.graspnet_detector import GraspNetDetector
from graspnet_ros.msg import GraspPattern


class GraspNetNode(Node):
    """
    # GraspNetののROS2ノードクラス

    ## publishers:
    - self.grasp_publisher: 推論した把持パターンを出版するPublisher

    ## subscribers:
    - self.detection_image_subscriber: カメラから対象が映る部分のRGB画像を購読するSubscriber
    - self.detection_depth_subscriber: カメラから対象が映る部分の深度画像を購読するSubscriber
    - self.camera_info_subscriber: カメラの内部パラメータを購読するSubscriber

    ## members:
    - grasp_detector: GraspNetの推論を行うクラスのインスタンス
    - grasp_pattern: 推論した把持パターンを格納する変数
    - camera_info: カメラの内部パラメータを格納する変数
    - meta_path: meta.matファイルのパスを格納する変数
    - ts: message_filters.ApproximateTimeSynchronizerのインスタンス（RGB画像と深度画像の同期に使用）
    - bridge: CvBridgeのインスタンス（ROSの画像メッセージとOpenCV形式の画像の変換に使用）

    ## methods:
    - grasp_publisher: GraspPatternメッセージを出版するPublisher
    - _create_point_cloud_from_depth: 深度画像からポイントクラウドを生成する関数

    """

    def __init__(self):
        super().__init__("graspnet_node")

        # Publisher
        self.grasp_publisher = self.create_publisher(GraspPattern, "grasp_pattern", 10)

        # Subscriber
        self.detection_image_subscriber = message_filters.Subscriber(
            self, Image, "/camera/detection/image"
        )
        self.detection_depth_subscriber = message_filters.Subscriber(
            self, Image, "/camera/detection/depth"
        )
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.detection_image_subscriber, self.detection_depth_subscriber],
            queue_size=10,
            slop=0.1,
        )
        self.ts.registerCallback(self._get_grasp)

        self.camera_info_subscriber = self.create_subscription(
            CameraInfo, "/camera/camera/depth/camera_info", self._camera_info_callback, 10
        )

        # realsense_cameraのメタデータ
        self.camera_info = None

        # GraspNetの推論クラス
        self.grasp_detector = GraspNetDetector()

        self.bridge = CvBridge()

        # rate limit / duplicate header guard
        self.declare_parameter("min_publish_interval", 0.0)
        self.min_publish_interval = float(
            self.get_parameter("min_publish_interval").value
        )
        self._last_publish_time = 0.0
        self._last_header_stamp = None

        self.get_logger().info("GraspNetNode has been started.")

    def _camera_info_callback(self, msg):
        """
        カメラの内部パラメータを受け取るコールバック関数
        """

        self.camera_info = CameraIntrisics(
            width=msg.width,
            height=msg.height,
            fx=msg.k[0],
            fy=msg.k[4],
            cx=msg.k[2],
            cy=msg.k[5],
            scale=1000.0,  # 深度画像のスケール（mm単位の場合は1000.0）
        )
        self.get_logger().info("Camera info received and stored.")

    def _get_grasp(self, image_msg, depth_msg):
        """
        GraspNetの推論を実行する関数
        """

        if self.camera_info is None:
            self.get_logger().warning("Camera info has not been received yet.")
            return

        # 同じ画像に対して複数回推論しないようにするためのガード(おそらくいらない)
        now = time.time()
        header = image_msg.header
        stamp = (int(header.stamp.sec), int(header.stamp.nanosec))
        if (
            self._last_header_stamp == stamp
            and (now - self._last_publish_time) < self.min_publish_interval
        ):
            return
        if (now - self._last_publish_time) < self.min_publish_interval:
            return

        # ROSの画像メッセージをOpenCV形式の画像に変換
        color_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
        depth_image = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding="passthrough"
        )

        # TODO 色情報の順番が違う可能性あり(imgmsg_to_cv2時にrgb8を指定しているため、色の順番はRGBのはず)
        color = color_image.astype(np.float32) / 255.0
        # TODO 明らかにマニピュレータが届かない距離，ノイズなどを事前にフィルタリングする
        depth = depth_image.astype(np.float32)

        # GraspNetの推論を実行
        grasp_patterns = self.grasp_detector.get_grasps(color, depth, self.camera_info)

        # 把持パターンを出版
        self._publish_grasp_pattern(grasp_patterns, image_msg.header)

        # ヘッダースタンプを更新
        self._last_publish_time = now
        self._last_header_stamp = stamp

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
        msg.rotation_matrix = [
            float(value) for value in grasp.rotation_matrix.reshape(-1)
        ]
        msg.translation = [float(value) for value in grasp.translation.reshape(-1)]
        msg.object_id = int(grasp.object_id)
        return msg

    def _publish_grasp_pattern(self, grasp_patterns, header):
        """
        GraspPatternメッセージを出版する関数
        """

        if len(grasp_patterns) == 0:
            grasp_pattern_msg = self._grasp_to_msg(None, header)
        else:
            for grasp_pattern in grasp_patterns:
                grasp_pattern_msg = self._grasp_to_msg(grasp_pattern, header)
                self.grasp_publisher.publish(grasp_pattern_msg)
        self.get_logger().info(
            f"Published grasp pattern length: {len(grasp_patterns)}"
        )

    # def _publish_grasp_patterns(self, grasp_patterns, header):
    #     """
    #     複数のGraspPatternメッセージを出版する関数
    #     """

    #     if len(grasp_patterns) == 0:
    #         grasp_pattern_msg = [self._grasp_to_msg(None, header)]
    #     else:
    #         grasp_pattern_msg = [self._grasp_to_msg(grasp_pattern, header) for grasp_pattern in grasp_patterns]

    #     self.grasp_publisher.publish(grasp_pattern_msg)
    #     self.get_logger().info(
    #         f"Published grasp pattern with score: {grasp_pattern.score:.4f}"



class CameraIntrisics:
    """Camera intrisics for point cloud creation."""

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


if __name__ == "__main__":
    main()
