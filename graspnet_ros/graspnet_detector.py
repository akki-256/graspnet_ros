from pathlib import Path
import sys

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
GRASPNET_BASELINE_ROOT = PACKAGE_ROOT / 'graspnet-baseline'

sys.path.append(str(GRASPNET_BASELINE_ROOT / 'graspnetAPI'))
sys.path.append(str(GRASPNET_BASELINE_ROOT / 'models'))

from graspnetAPI import GraspGroup
from graspnet import GraspNet, pred_decode

class GraspNetDetector:
    """
    GraspNet推論クラス
    """
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_path = GRASPNET_BASELINE_ROOT / 'checkpoint' / 'checkpoint-rs.tar'

        if not self.model_path.exists():
            raise FileNotFoundError(f'GraspNet checkpoint not found: {self.model_path}')

        self.model = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                               cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],
                               is_training=False)

        if self.device.type == 'cpu':
            print("Warning: CUDA device not found. Running on CPU, which may be slow.")

        self.model.to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def get_grasp(self, cloud):
        """
        GraspNetの推論を行う関数
        """

        if cloud is None or len(cloud) == 0:
            return GraspGroup()

        if len(cloud) >= 300:
            idxs = np.random.choice(len(cloud), 300, replace=False)
        else:
            idxs1 = np.arange(len(cloud))
            idxs2 = np.random.choice(len(cloud), 300-len(cloud), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud[idxs]

        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        cloud_sampled = cloud_sampled.to(self.device)
        end_points['point_clouds'] = cloud_sampled

        with torch.no_grad():
            end_points = self.model(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)

        gg.nms()
        gg.sort_by_score()
        gg = gg[:10]

        return gg

    def get_best_grasp(self, cloud):
        grasp_group = self.get_grasp(cloud)
        if len(grasp_group) == 0:
            return None
        return grasp_group[0]