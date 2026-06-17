#!/usr/bin/env python3

from pathlib import Path
import time

import numpy as np
import open3d as o3d
import scipy.io as scio
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from graspnet_ros.msg import GraspPattern
from graspnetAPI import Grasp, GraspGroup


class SampleImagePublisher(Node):
	def __init__(self):
		super().__init__('sample_image_publisher')

		qos_profile = QoSProfile(
			history=HistoryPolicy.KEEP_LAST,
			depth=1,
			reliability=ReliabilityPolicy.RELIABLE,
			durability=DurabilityPolicy.TRANSIENT_LOCAL,
		)

		self.image_publisher = self.create_publisher(Image, '/camera/detection/image', qos_profile)
		self.depth_publisher = self.create_publisher(Image, '/camera/detection/depth', qos_profile)
		self.grasp_subscriber = self.create_subscription(GraspPattern, 'grasp_pattern', self._on_grasp_pattern, 50)

		sample_dir = Path(__file__).resolve().parent / 'sample'
		self.rgb_path = sample_dir / 'rgb.png'
		self.depth_path = sample_dir / 'depth.png'
		self.workspace_mask_path = sample_dir / 'workspace_mask.png'
		self.meta_path = Path(__file__).resolve().parents[1] / 'meta.mat'

		self.rgb_image = self._load_rgb_image(self.rgb_path)
		self.depth_image = self._load_depth_image(self.depth_path)
		self.workspace_mask = self._load_workspace_mask(self.workspace_mask_path)
		self.rgb_image = self._apply_workspace_mask_to_rgb(self.rgb_image, self.workspace_mask)
		self.depth_image = self._apply_workspace_mask_to_depth(self.depth_image, self.workspace_mask)
		self.camera_info = self._load_camera_info(self.meta_path)
		self.point_cloud = self._create_point_cloud(self.depth_image, self.rgb_image, self.camera_info)
		self._timer_fired = False
		self._batch_header_stamp = None
		self._received_grasps = []
		self._batch_last_received_time = None
		self._batch_timeout_sec = 0.5
		self._batch_visualized = False

		self.timer = self.create_timer(1.0, self._publish_samples_once)
		self.visualize_timer = self.create_timer(0.1, self._maybe_visualize_received_grasps)

		self.get_logger().info(f'Loaded sample images from {sample_dir}')
		self.get_logger().info('Waiting for grasp_pattern...')

	def _load_camera_info(self, path: Path):
		if not path.exists():
			raise FileNotFoundError(f'meta.mat not found: {path}')

		meta = scio.loadmat(str(path))
		intrinsic_matrix = meta['intrinsic_matrix']
		factor_depth = float(np.asarray(meta['factor_depth']).squeeze())
		return CameraInfo(
			width=float(self.rgb_image.width),
			height=float(self.rgb_image.height),
			fx=float(intrinsic_matrix[0][0]),
			fy=float(intrinsic_matrix[1][1]),
			cx=float(intrinsic_matrix[0][2]),
			cy=float(intrinsic_matrix[1][2]),
			scale=factor_depth,
		)

	def _load_rgb_image(self, path: Path) -> Image:
		if not path.exists():
			raise FileNotFoundError(f'RGB image not found: {path}')

		rgb_array = np.array(PILImage.open(path).convert('RGB'), dtype=np.uint8)
		return self._array_to_image_msg(rgb_array, 'rgb8')

	def _load_depth_image(self, path: Path) -> Image:
		if not path.exists():
			raise FileNotFoundError(f'Depth image not found: {path}')

		depth_pil = PILImage.open(path)
		depth_array = np.array(depth_pil)

		if depth_array.ndim != 2:
			raise ValueError(f'Depth image must be single-channel, got shape {depth_array.shape}')

		if depth_array.dtype == np.uint16:
			encoding = '16UC1'
		elif depth_array.dtype == np.float32:
			encoding = '32FC1'
		else:
			depth_array = depth_array.astype(np.uint16, copy=False)
			encoding = '16UC1'

		return self._array_to_image_msg(depth_array, encoding)

	def _load_workspace_mask(self, path: Path) -> np.ndarray:
		if not path.exists():
			raise FileNotFoundError(f'workspace_mask image not found: {path}')

		mask_array = np.array(PILImage.open(path).convert('L'), dtype=np.uint8)
		return mask_array > 0

	def _apply_workspace_mask_to_rgb(self, rgb_msg: Image, workspace_mask: np.ndarray) -> Image:
		rgb = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape((rgb_msg.height, rgb_msg.width, 3)).copy()
		if workspace_mask.shape != rgb.shape[:2]:
			raise ValueError(f'workspace_mask shape {workspace_mask.shape} does not match RGB shape {rgb.shape[:2]}')

		rgb[~workspace_mask] = 0
		return self._array_to_image_msg(rgb, 'rgb8')

	def _apply_workspace_mask_to_depth(self, depth_msg: Image, workspace_mask: np.ndarray) -> Image:
		if depth_msg.encoding == '32FC1':
			dtype = np.float32
		else:
			dtype = np.uint16

		depth = np.frombuffer(depth_msg.data, dtype=dtype).reshape((depth_msg.height, depth_msg.width)).copy()
		if workspace_mask.shape != depth.shape:
			raise ValueError(f'workspace_mask shape {workspace_mask.shape} does not match depth shape {depth.shape}')

		depth[~workspace_mask] = 0
		return self._array_to_image_msg(depth, depth_msg.encoding)

	def _create_point_cloud(self, depth_msg: Image, rgb_msg: Image, camera_info) -> o3d.geometry.PointCloud:
		if depth_msg.encoding == '32FC1':
			depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape((depth_msg.height, depth_msg.width))
		else:
			depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape((depth_msg.height, depth_msg.width))
		rgb = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape((rgb_msg.height, rgb_msg.width, 3))

		if depth.shape != (int(camera_info.height), int(camera_info.width)):
			raise ValueError(f'Depth image shape {depth.shape} does not match camera info {(camera_info.height, camera_info.width)}')

		depth_m = depth.astype(np.float32) / camera_info.scale
		height, width = depth_m.shape
		xmap, ymap = np.meshgrid(np.arange(width), np.arange(height))
		points_z = depth_m
		points_x = (xmap - camera_info.cx) * points_z / camera_info.fx
		points_y = (ymap - camera_info.cy) * points_z / camera_info.fy
		points = np.stack([points_x, points_y, points_z], axis=-1)

		mask = depth > 0
		points = points[mask]
		colors = rgb[mask].astype(np.float32) / 255.0

		point_cloud = o3d.geometry.PointCloud()
		point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
		point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
		return point_cloud

	def _array_to_image_msg(self, array: np.ndarray, encoding: str) -> Image:
		msg = Image()
		msg.header.stamp = self.get_clock().now().to_msg()
		msg.header.frame_id = 'camera'
		msg.height = int(array.shape[0])
		msg.width = int(array.shape[1])
		msg.encoding = encoding
		msg.is_bigendian = 0
		msg.step = int(array.strides[0])
		msg.data = array.tobytes()
		return msg

	def _publish_samples_once(self) -> None:
		if self._timer_fired:
			return

		self.rgb_image.header.stamp = self.get_clock().now().to_msg()
		self.depth_image.header.stamp = self.get_clock().now().to_msg()

		self.image_publisher.publish(self.rgb_image)
		self.depth_publisher.publish(self.depth_image)
		self._timer_fired = True
		self.timer.cancel()
		self.get_logger().info('Published sample RGB and depth images once.')

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
		geometries = [self.point_cloud, *grasp_group.to_open3d_geometry_list()]
		geometries[0] = geometries[0].voxel_down_sample(voxel_size=0.005)
		o3d.visualization.draw_geometries(geometries)
		rclpy.shutdown()


class CameraInfo:
	def __init__(self, width, height, fx, fy, cx, cy, scale):
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.scale = scale


def main() -> None:
	rclpy.init()
	node = SampleImagePublisher()

	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main()
