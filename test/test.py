#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


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

		sample_dir = Path(__file__).resolve().parent / 'sample'
		self.rgb_path = sample_dir / 'rgb.png'
		self.depth_path = sample_dir / 'depth.png'

		self.rgb_image = self._load_rgb_image(self.rgb_path)
		self.depth_image = self._load_depth_image(self.depth_path)

		self.timer = self.create_timer(1.0, self._publish_samples)

		self.get_logger().info(f'Loaded sample images from {sample_dir}')

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

	def _publish_samples(self) -> None:
		self.rgb_image.header.stamp = self.get_clock().now().to_msg()
		self.depth_image.header.stamp = self.get_clock().now().to_msg()

		self.image_publisher.publish(self.rgb_image)
		self.depth_publisher.publish(self.depth_image)


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
