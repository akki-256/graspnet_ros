#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS 2 node to subscribe to Realsense color and depth images once,
publish them to /camera/detection/image and /camera/detection/depth,
subscribe to the grasp_pattern output topic, and visualize the predicted grasp patterns in 3D.
"""

import time
from pathlib import Path
import numpy as np
import open3d as o3d
import scipy.io as scio

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo as CameraInfoMsg
import message_filters
from cv_bridge import CvBridge

from graspnet_ros.msg import GraspPattern
from graspnetAPI import Grasp, GraspGroup

class CameraInfo:
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

class RealsenseBridge(Node):
    """
    Subscribes to RGB and depth images from realsense2_camera once,
    publishes them to detection topics, collects grasp patterns, and visualizes them in 3D.
    """
    def __init__(self):
        super().__init__('realsense_bridge')

        # Declare parameters for topics with default Realsense camera topics
        self.declare_parameter('color_topic', '/realsense_d435_camera/realsense_d435_camera/color/image_raw')
        self.declare_parameter('depth_topic', '/realsense_d435_camera/realsense_d435_camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('out_color_topic', '/camera/camera/detect_color/image_raw')
        self.declare_parameter('out_depth_topic', '/camera/camera/aligned_detect_depth/image_raw')
        # self.declare_parameter('grasp_topic', 'grasp_pattern')

        self.color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.out_color_topic = self.get_parameter('out_color_topic').get_parameter_value().string_value
        self.out_depth_topic = self.get_parameter('out_depth_topic').get_parameter_value().string_value
        # self.grasp_topic = self.get_parameter('grasp_topic').get_parameter_value().string_value

        # Derive camera info topic from color topic
        if 'image_raw' in self.color_topic:
            self.camera_info_topic = self.color_topic.replace('image_raw', 'camera_info')
        else:
            self.camera_info_topic = '/camera/camera/color/camera_info'

        self.get_logger().info(f"Input Color Topic: {self.color_topic}")
        self.get_logger().info(f"Input Depth Topic: {self.depth_topic}")
        self.get_logger().info(f"Input Camera Info Topic: {self.camera_info_topic}")
        self.get_logger().info(f"Output Color Topic: {self.out_color_topic}")
        self.get_logger().info(f"Output Depth Topic: {self.out_depth_topic}")
        # self.get_logger().info(f"Grasp Pattern Topic: {self.grasp_topic}")

        # Load default CameraInfo from meta.mat as fallback
        self.package_root = Path(__file__).resolve().parents[1]
        self.meta_path = self.package_root / 'meta.mat'
        self.camera_info = None
        if self.meta_path.exists():
            try:
                meta = scio.loadmat(str(self.meta_path))
                intrinsic_matrix = meta['intrinsic_matrix']
                factor_depth = float(meta['factor_depth'][0][0])
                self.camera_info = CameraInfo(
                    width=1280.0,
                    height=720.0,
                    fx=float(intrinsic_matrix[0][0]),
                    fy=float(intrinsic_matrix[1][1]),
                    cx=float(intrinsic_matrix[0][2]),
                    cy=float(intrinsic_matrix[1][2]),
                    scale=factor_depth,
                )
                self.get_logger().info("Loaded fallback camera info from meta.mat")
            except Exception as e:
                self.get_logger().error(f"Failed to load meta.mat fallback parameters: {e}")

        # Publisher QoS profile matches graspnet_ros test.py (Transient Local / Reliable)
        pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.color_pub = self.create_publisher(Image, self.out_color_topic, pub_qos)
        self.depth_pub = self.create_publisher(Image, self.out_depth_topic, pub_qos)

        # Subscriber QoS profile for realsense (Reliable / Volatile)
        sub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Subscribing to camera info
        self.camera_info_sub = self.create_subscription(
            CameraInfoMsg,
            self.camera_info_topic,
            self.camera_info_callback,
            sub_qos
        )

        # Subscribing to color and depth
        self.color_sub = message_filters.Subscriber(self, Image, self.color_topic, qos_profile=sub_qos)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=sub_qos)

        # Synchronize color and depth images
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)

        # # Grasp subscribers
        # self.grasp_sub = self.create_subscription(
        #     GraspPattern,
        #     self.grasp_topic,
        #     self.grasp_callback,
        #     50
        # )

        # Internal state
        self.received = False
        self.latest_color_msg = None
        self.latest_depth_msg = None

        # Grasp pattern accumulation state
        self._batch_header_stamp = None
        self._received_grasps = []
        self._batch_last_received_time = None
        self._batch_timeout_sec = 0.5
        self._batch_visualized = False

        # Timers
        self.visualize_timer = self.create_timer(0.1, self.maybe_visualize_grasps)

        self.get_logger().info("Waiting for first synced RGB and Depth image pair...")

    def camera_info_callback(self, msg: CameraInfoMsg):
        # Update camera_info with live data from camera
        self.camera_info = CameraInfo(
            width=float(msg.width),
            height=float(msg.height),
            fx=float(msg.k[0]),
            fy=float(msg.k[4]),
            cx=float(msg.k[2]),
            cy=float(msg.k[5]),
            scale=1000.0 # Standard RealSense units: 1000 steps per meter
        )

    def image_callback(self, color_msg: Image, depth_msg: Image):
        if self.received:
            return

        self.received = True
        self.get_logger().info("Received synced RGB and Depth image pair. Publishing to output topics...")

        # Save messages to build point cloud later
        self.latest_color_msg = color_msg
        self.latest_depth_msg = depth_msg

        # Publish messages as-is
        self.color_pub.publish(color_msg)
        self.depth_pub.publish(depth_msg)
        self.get_logger().info("Published successfully! Waiting for grasp pattern results...")

        # Unsubscribe from camera inputs so we only process this single frame
        self.destroy_subscription(self.color_sub.sub)
        self.destroy_subscription(self.depth_sub.sub)
        if hasattr(self, 'camera_info_sub') and self.camera_info_sub:
            self.destroy_subscription(self.camera_info_sub)

    def grasp_callback(self, msg: GraspPattern):
        if msg.score < 0:
            self.get_logger().warning("Received empty grasp pattern.")
            return

        stamp = (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))
        if self._batch_header_stamp is None or self._batch_header_stamp != stamp:
            self._batch_header_stamp = stamp
            self._received_grasps = []
            self._batch_last_received_time = None
            self._batch_visualized = False

        grasp = Grasp(
            msg.score,
            msg.width,
            msg.height,
            msg.depth,
            np.array(msg.rotation_matrix, dtype=np.float64).reshape(3, 3),
            np.array(msg.translation, dtype=np.float64),
            msg.object_id,
        )
        self._received_grasps.append(grasp)
        self._batch_last_received_time = time.monotonic()
        self.get_logger().info(
            f"Received grasp pattern {len(self._received_grasps)}: "
            f"score={msg.score:.4f}, width={msg.width:.4f}"
        )

    def maybe_visualize_grasps(self):
        if self._batch_visualized:
            return
        if not self._received_grasps or self._batch_last_received_time is None:
            return

        elapsed = time.monotonic() - self._batch_last_received_time
        if elapsed < self._batch_timeout_sec:
            return

        self.visualize_grasps()

    def create_point_cloud(self, depth_msg: Image, rgb_msg: Image, camera_info: CameraInfo) -> o3d.geometry.PointCloud:
        br = CvBridge()

        # Load depth array safely
        depth = br.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        if depth_msg.encoding == '32FC1' or np.nanmax(depth) < 20.0:
            actual_scale = 1.0
        else:
            actual_scale = 1000.0

        # Load RGB array safely (forces output to be properly shaped RGB)
        rgb = br.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')

        # Dynamically scale camera info if image size doesn't match
        img_width = float(depth_msg.width)
        img_height = float(depth_msg.height)

        # Create a local copy of camera_info to avoid mutating the original
        fx = camera_info.fx
        fy = camera_info.fy
        cx = camera_info.cx
        cy = camera_info.cy

        # ※ ここにあった scale = camera_info.scale は削除（上の actual_scale を使うため）

        if camera_info.width != img_width or camera_info.height != img_height:
            scale_x = img_width / camera_info.width
            scale_y = img_height / camera_info.height
            fx *= scale_x
            cx *= scale_x
            fy *= scale_y
            cy *= scale_y

        # ここでエンコーディングに応じた正しいスケールで割る
        depth_m = depth.astype(np.float32) / actual_scale
        height, width = depth_m.shape
        xmap, ymap = np.meshgrid(np.arange(width), np.arange(height))

        points_z = depth_m
        points_x = (xmap - cx) * points_z / fx
        points_y = (ymap - cy) * points_z / fy
        points = np.stack([points_x, points_y, points_z], axis=-1)

        mask = depth > 0
        points = points[mask]
        colors = rgb[mask].astype(np.float32) / 255.0

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        return point_cloud

    def visualize_grasps(self):
        if not self._received_grasps or not self.latest_color_msg or not self.latest_depth_msg:
            return

        self._batch_visualized = True
        self.get_logger().info("Starting 3D visualization...")

        if self.camera_info is None:
            self.get_logger().error("Camera intrinsic info not available! Cannot construct point cloud.")
            rclpy.shutdown()
            return

        # Reconstruct point cloud from stored images
        try:
            point_cloud = self.create_point_cloud(self.latest_depth_msg, self.latest_color_msg, self.camera_info)
        except Exception as e:
            self.get_logger().error(f"Failed to create point cloud: {e}")
            rclpy.shutdown()
            return

        grasps = self._received_grasps
        grasp_group = GraspGroup(np.array([g.grasp_array for g in grasps], dtype=np.float64))
        grasp_group.sort_by_score()

        # Prepare 3D geometries
        geometries = [point_cloud, *grasp_group.to_open3d_geometry_list()]
        # geometries[0] = geometries[0].voxel_down_sample(voxel_size=0.005)

        # Visualize
        o3d.visualization.draw_geometries(geometries)

        # Shutdown once visualization is closed
        self.get_logger().info("Visualization window closed. Shutting down node.")
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = RealsenseBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
