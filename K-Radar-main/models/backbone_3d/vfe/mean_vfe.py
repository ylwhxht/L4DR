# Modified from OpenPCDet (https://github.com/open-mmlab/OpenPCDet)
import torch
import torch.nn as nn

class MeanVFE(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.model_cfg = cfg.MODEL
        self.num_point_features = self.model_cfg.PRE_PROCESSOR.INPUT_DIM

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        if 'lidar_voxels' in batch_dict.keys():
            lidar_voxel_features, lidar_voxel_num_points = batch_dict['lidar_voxels'], batch_dict['lidar_voxel_num_points']
            lidar_points_mean = lidar_voxel_features[:, :, :].sum(dim=1, keepdim=False)
            lidar_normalizer = torch.clamp_min(lidar_voxel_num_points.view(-1, 1), min=1.0).type_as(lidar_voxel_features)
            lidar_points_mean = lidar_points_mean / lidar_normalizer
            batch_dict['lidar_voxel_features'] = lidar_points_mean.contiguous()
            radar_voxel_features, radar_voxel_num_points = batch_dict['radar_voxels'], batch_dict['radar_voxel_num_points']
            radar_points_mean = radar_voxel_features[:, :, :].sum(dim=1, keepdim=False)
            radar_normalizer = torch.clamp_min(radar_voxel_num_points.view(-1, 1), min=1.0).type_as(radar_voxel_features)
            radar_points_mean = radar_points_mean / radar_normalizer
            batch_dict['radar_voxel_features'] = radar_points_mean.contiguous()
        else:
            voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
            points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
            normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
            points_mean = points_mean / normalizer
            batch_dict['voxel_features'] = points_mean.contiguous()

        return batch_dict

