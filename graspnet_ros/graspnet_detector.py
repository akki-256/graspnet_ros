import sys
from pathlib import Path

import numpy as np
import open3d
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
GRASPNET_BASELINE_ROOT = PACKAGE_ROOT / "graspnet-baseline"

sys.path.append(str(GRASPNET_BASELINE_ROOT / "utils"))
sys.path.append(str(GRASPNET_BASELINE_ROOT / "graspnetAPI"))
sys.path.append(str(GRASPNET_BASELINE_ROOT / "models"))

from collision_detector import ModelFreeCollisionDetector
from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup


class GraspNetDetector:
    """
    GraspNet推論クラス

    menbers:
    - device: cpu or cuda:0
    - model: GraspNetのモデルのインスタンス

    methods:
    - get_grasp: Graspnetの推論を行う関数
    - collision_detect: GraspNetの推論結果に対して衝突検出を行う関数
    - get_best_grasp: GraspNetの推論結果から最もスコアの高い把持を返す関数
    """

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = GraspNet(
            input_feature_dim=0,
            num_view=300,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        self.model.to(self.device)

        model_path = GRASPNET_BASELINE_ROOT / "checkpoint" / "checkpoint-rs.tar"
        if not model_path.exists():
            raise FileNotFoundError(f"GraspNet checkpoint not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()


    def get_grasps(self, color, depth, camera_info, grasp_quantity = 1)-> GraspGroup:
        """
        GraspNetの推論結果から最もスコアの高い把持を返す関数
        """

        # xyz点群を生成
        flat_points = self._create_flat_points(depth, camera_info)
        flat_color = color.reshape([-1, 3])

        if flat_points is None or len(flat_points) == 0:
            return GraspGroup()

        # 点群データをサンプリングして入力データ量を調整
        sampled_color,sampled_points = self._sample_input_data(flat_points, flat_points, num_points=20000)
        
        # GraspNetの推論に必要なend_pointsを作成
        end_points = self._create_end_points(sampled_color, sampled_points)
        
        # GraspNetの推論を実行
        with torch.no_grad():
            end_points = self.model(end_points)
            grasp_preds = pred_decode(end_points)
            
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        
        # Open3Dの点群を作成
        colored_points = self._coloring_cloud(flat_color, flat_points)

        # 衝突検出を行い、衝突する把持を除外
        collision_mask = self.collision_detect(gg, np.asarray(colored_points.points))
        gg = gg[~collision_mask]

        # 非最大抑制を行い、スコアの高い把持を優先
        gg.nms()
        gg.sort_by_score()

        grasp_quantity = 50
        return gg[:grasp_quantity]

    def _create_flat_points(self, depth, camera_info):
        """
        深度画像から生の3D点群を生成する関数
        """

        height, width = depth.shape[:2]
        xmap = np.arange(width)
        ymap = np.arange(height)
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth /1000.0 # 深度画像単位 mm -> m
        points_x = (xmap - camera_info.cx) * points_z / camera_info.fx
        points_y = (ymap - camera_info.cy) * points_z / camera_info.fy
        raw_points_np = np.stack([points_x, points_y, points_z], axis=-1)
        
        points_flat = raw_points_np.reshape([-1, 3])

        
        return points_flat

    def _sample_input_data(self, color, points_flat, num_points=20000):
        """
        点群を一定の数にサンプリングする関数
        
        """

        # 入力に用いる点群は一定の数でないといけない？らしい
        if len(points_flat) >= num_points:
            sample_idxs = np.random.choice(len(points_flat), num_points, replace=False)
        else:  # ので点群が少ない場合は量増しする
            idxs1 = np.arange(len(points_flat))
            idxs2 = np.random.choice(len(points_flat), num_points - len(points_flat), replace=True)
            sample_idxs = np.concatenate([idxs1, idxs2], axis=0)

        color_sampled = color[sample_idxs]
        points_sampled = points_flat[sample_idxs]

        return color_sampled, points_sampled


    def _create_end_points(self, color_sampled, points_sampled):
        """
        GraspNetの推論に必要なend_pointsを作成する関数
        """
        
        points_sampled_tensor = torch.from_numpy(points_sampled[np.newaxis].astype(np.float32))
        points_sampled_tensor = points_sampled_tensor.to(self.device)

        end_points = dict()
        end_points["cloud_colors"] = color_sampled
        end_points["point_clouds"] = points_sampled_tensor
        
        return end_points

    def _coloring_cloud(self, color, raw_points_np):
        """
        サンプリングされた色画像, 点群からOpen3Dの点群を作成する関数
        """

        colored_points = open3d.geometry.PointCloud()
        colored_points.points = open3d.utility.Vector3dVector(
            raw_points_np.astype(np.float32)
        )
        colored_points.colors = open3d.utility.Vector3dVector(
            color.astype(np.float32)
        )

        return colored_points
    
    def collision_detect(self, grasp_group, cloud):
        """
        GraspNetの推論結果に対して衝突検出を行う関数
        """
        if grasp_group is None or len(grasp_group) == 0:
            return np.array([], dtype=bool)

        collision_detector = ModelFreeCollisionDetector(cloud, voxel_size=0.01)
        collision_mask = collision_detector.detect(
            grasp_group, approach_dist=0.05, collision_thresh=0.01
        )
        return collision_mask

    def get_best_grasp(self, cloud):
        grasp_group = self.get_grasp(cloud)
        if len(grasp_group) == 0:
            return None
        return grasp_group[0]
