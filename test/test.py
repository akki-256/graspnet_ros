#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import time

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent.parent
INSTALL_PYTHON_PATH = WORKSPACE_ROOT / 'install' / 'graspnet_ros' / 'local' / 'lib' / 'python3.10' / 'dist-packages'
if INSTALL_PYTHON_PATH.exists() and str(INSTALL_PYTHON_PATH) not in sys.path:
	sys.path.insert(0, str(INSTALL_PYTHON_PATH))

if str(PACKAGE_ROOT) not in sys.path:
	sys.path.append(str(PACKAGE_ROOT))

import numpy as np
import open3d as o3d
import scipy.io as scio
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo as CameraInfoMsg
from sensor_msgs.msg import Image

from graspnet_ros.msg import GraspPattern
from graspnetAPI import Grasp, GraspGroup


class CameraInfoModel:
	def __init__(self, width, height, fx, fy, cx, cy, scale):
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.scale = scale


class SampleImagePublisher(Node):
	def __init__(self):
		super().__init__('sample_image_publisher')

		qos_profile = QoSProfile(
			history=HistoryPolicy.KEEP_LAST,
			depth=1,
			reliability=ReliabilityPolicy.RELIABLE,
			durability=DurabilityPolicy.TRANSIENT_LOCAL,
		)

		self.color_topic = '/camera/camera/detect_color/image_raw'
		self.depth_topic = '/camera/camera/aligned_detect_depth/image_raw'
		self.aligned_depth_info_topic = '/camera/camera/aligned_depth_to_color/camera_info'

		self.color_publisher = self.create_publisher(Image, self.color_topic, qos_profile)
		self.depth_publisher = self.create_publisher(Image, self.depth_topic, qos_profile)
		self.aligned_depth_info_publisher = self.create_publisher(
			CameraInfoMsg, self.aligned_depth_info_topic, qos_profile
		)
		self.grasp_subscriber = self.create_subscription(
			GraspPattern, 'grasp_pattern', self._on_grasp_pattern, 50
		)

		sample_dir = Path(__file__).resolve().parent / 'sample'
		self.rgb_path = sample_dir / 'rgb.png'
		self.depth_path = sample_dir / 'depth.png'
		self.workspace_mask_path = sample_dir / 'workspace_mask.png'
		self.meta_path = Path(__file__).resolve().parents[1] / 'meta.mat'

		self.rgb_array = self._load_rgb_array(self.rgb_path)
		self.depth_array = self._load_depth_array(self.depth_path)
		self.workspace_mask = self._load_workspace_mask(self.workspace_mask_path)

		self.rgb_array = self._apply_workspace_mask_to_rgb(self.rgb_array, self.workspace_mask)
		self.depth_array = self._apply_workspace_mask_to_depth(self.depth_array, self.workspace_mask)

		self.camera_info = self._load_camera_info(self.meta_path, self.rgb_array.shape[:2])
		self.color_image_msg = self._build_image_msg(
			self.rgb_array, 'rgb8', 'camera_color_optical_frame'
		)
		self.depth_image_msg = self._build_image_msg(
			self.depth_array, '16UC1', 'camera_depth_optical_frame'
		)
		self.color_camera_info_msg = self._build_camera_info_msg(
			self.camera_info, 'camera_color_optical_frame'
		)
		self.depth_camera_info_msg = self._build_camera_info_msg(
			self.camera_info, 'camera_depth_optical_frame'
		)
		self.aligned_depth_camera_info_msg = self._build_camera_info_msg(
			self.camera_info, 'camera_color_optical_frame'
		)

		self.point_cloud = self._create_point_cloud(self.depth_array, self.rgb_array, self.camera_info)

		self._publish_timer_fired = False
		self._batch_header_stamp = None
		self._received_grasps = []
		self._batch_last_received_time = None
		self._batch_timeout_sec = 0.5
		self._batch_visualized = False
		self.publish_timer = self.create_timer(0.5, self._publish_samples_once)
		self.visualize_timer = self.create_timer(0.1, self._maybe_visualize_received_grasps)

		self.get_logger().info(f'Loaded sample images from {sample_dir}')
		self.get_logger().info('Published topics mirror the RealSense topics.')
		self.get_logger().info('Waiting for grasp_pattern from graspnet_ros...')

	def _load_camera_info(self, path: Path, image_shape: tuple) -> CameraInfoModel:
		if not path.exists():
			raise FileNotFoundError(f'meta.mat not found: {path}')

		meta = scio.loadmat(str(path))
		intrinsic_matrix = meta['intrinsic_matrix']
		factor_depth = float(np.asarray(meta['factor_depth']).squeeze())
		height, width = image_shape

		return CameraInfoModel(
			width=float(width),
			height=float(height),
			fx=float(intrinsic_matrix[0][0]),
			fy=float(intrinsic_matrix[1][1]),
			cx=float(intrinsic_matrix[0][2]),
			cy=float(intrinsic_matrix[1][2]),
			scale=factor_depth,
		)

	def _load_rgb_array(self, path: Path) -> np.ndarray:
		if not path.exists():
			raise FileNotFoundError(f'RGB image not found: {path}')

		return np.array(PILImage.open(path).convert('RGB'), dtype=np.uint8)

	def _load_depth_array(self, path: Path) -> np.ndarray:
		if not path.exists():
			raise FileNotFoundError(f'Depth image not found: {path}')

		depth_array = np.array(PILImage.open(path))
		if depth_array.ndim != 2:
			raise ValueError(f'Depth image must be single-channel, got shape {depth_array.shape}')

		if depth_array.dtype != np.uint16 and depth_array.dtype != np.float32:
			depth_array = depth_array.astype(np.uint16, copy=False)

		return depth_array

	def _load_workspace_mask(self, path: Path) -> np.ndarray:
		if not path.exists():
			raise FileNotFoundError(f'workspace_mask image not found: {path}')

		mask_array = np.array(PILImage.open(path).convert('L'), dtype=np.uint8)
		return mask_array > 0

	def _apply_workspace_mask_to_rgb(self, rgb: np.ndarray, workspace_mask: np.ndarray) -> np.ndarray:
		if workspace_mask.shape != rgb.shape[:2]:
			raise ValueError(
				f'workspace_mask shape {workspace_mask.shape} does not match RGB shape {rgb.shape[:2]}'
			)

		masked_rgb = rgb.copy()
		masked_rgb[~workspace_mask] = 0
		return masked_rgb

	def _apply_workspace_mask_to_depth(self, depth: np.ndarray, workspace_mask: np.ndarray) -> np.ndarray:
		if workspace_mask.shape != depth.shape:
			raise ValueError(
				f'workspace_mask shape {workspace_mask.shape} does not match depth shape {depth.shape}'
			)

		masked_depth = depth.copy()
		masked_depth[~workspace_mask] = 0
		return masked_depth

	def _create_point_cloud(
		self,
		depth_array: np.ndarray,
		rgb_array: np.ndarray,
		camera_info: CameraInfoModel,
	) -> o3d.geometry.PointCloud:
		if depth_array.shape != (int(camera_info.height), int(camera_info.width)):
			raise ValueError(
				f'Depth image shape {depth_array.shape} does not match camera info '
				f'({camera_info.height}, {camera_info.width})'
			)

		depth_m = depth_array.astype(np.float32) / camera_info.scale
		height, width = depth_m.shape
		xmap, ymap = np.meshgrid(np.arange(width), np.arange(height))

		points_z = depth_m
		points_x = (xmap - camera_info.cx) * points_z / camera_info.fx
		points_y = (ymap - camera_info.cy) * points_z / camera_info.fy
		points = np.stack([points_x, points_y, points_z], axis=-1)

		mask = depth_array > 0
		points = points[mask]
		colors = rgb_array[mask].astype(np.float32) / 255.0

		point_cloud = o3d.geometry.PointCloud()
		point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
		point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
		return point_cloud

	def _build_image_msg(self, array: np.ndarray, encoding: str, frame_id: str) -> Image:
		msg = Image()
		msg.header.frame_id = frame_id
		msg.height = int(array.shape[0])
		msg.width = int(array.shape[1])
		msg.encoding = encoding
		msg.is_bigendian = 0
		msg.step = int(array.strides[0])
		msg.data = array.tobytes()
		return msg

	def _build_camera_info_msg(self, camera_info: CameraInfoModel, frame_id: str) -> CameraInfoMsg:
		msg = CameraInfoMsg()
		msg.header.frame_id = frame_id
		msg.width = int(camera_info.width)
		msg.height = int(camera_info.height)
		msg.distortion_model = 'plumb_bob'
		msg.d = [0.0] * 5
		msg.k = [
			float(camera_info.fx), 0.0, float(camera_info.cx),
			0.0, float(camera_info.fy), float(camera_info.cy),
			0.0, 0.0, 1.0,
		]
		msg.r = [
			1.0, 0.0, 0.0,
			0.0, 1.0, 0.0,
			0.0, 0.0, 1.0,
		]
		msg.p = [
			float(camera_info.fx), 0.0, float(camera_info.cx), 0.0,
			0.0, float(camera_info.fy), float(camera_info.cy), 0.0,
			0.0, 0.0, 1.0, 0.0,
		]
		msg.binning_x = 0
		msg.binning_y = 0
		return msg

	def _publish_samples_once(self) -> None:
		if self._publish_timer_fired:
			return

		stamp = self.get_clock().now().to_msg()

		self.color_image_msg.header.stamp = stamp
		self.depth_image_msg.header.stamp = stamp
		self.color_camera_info_msg.header.stamp = stamp
		self.depth_camera_info_msg.header.stamp = stamp
		self.aligned_depth_camera_info_msg.header.stamp = stamp
  
		self.aligned_depth_info_publisher.publish(self.aligned_depth_camera_info_msg)
		input("Press Enter to publish sample RGB, depth, and camera info once...")

		self.color_publisher.publish(self.color_image_msg)
		self.depth_publisher.publish(self.depth_image_msg)

		self._publish_timer_fired = True
		self.publish_timer.cancel()
		self.get_logger().info('Published sample RGB, depth, and camera info once.')

	def _on_grasp_pattern(self, msg: GraspPattern) -> None:
		if msg.score < 0:
			self.get_logger().warning('Received an empty grasp_pattern message; waiting for a valid grasp.')
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
			f'Received grasp_pattern {len(self._received_grasps)} with score={msg.score:.4f}, '
			f'width={msg.width:.4f}, height={msg.height:.4f}, depth={msg.depth:.4f}'
		)

	def _maybe_visualize_received_grasps(self) -> None:
		if self._batch_visualized:
			return
		if not self._received_grasps or self._batch_last_received_time is None:
			return

		elapsed = time.monotonic() - self._batch_last_received_time
		if elapsed < self._batch_timeout_sec:
			return

		self._visualize_received_grasps()

	def _visualize_received_grasps(self) -> None:
		if not self._received_grasps:
			return

		self._batch_visualized = True

		grasps = self._received_grasps
		grasp_group = GraspGroup(np.array([grasp.grasp_array for grasp in grasps], dtype=np.float64))
		grasp_group.sort_by_score()

		if len(grasp_group) > 0:
			grasp_group.sort_by_score()

		point_cloud = self.point_cloud.voxel_down_sample(voxel_size=0.005)
		geometries = [point_cloud]

		if len(grasp_group) > 0:
			geometries.extend(grasp_group.to_open3d_geometry_list())

		o3d.visualization.draw_geometries(geometries)
		if rclpy.ok():
			rclpy.shutdown()


def main() -> None:
	rclpy.init()
	node = SampleImagePublisher()

	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		if rclpy.ok():
			rclpy.shutdown()


if __name__ == '__main__':
	main()
