import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        if 'pillar_features' in batch_dict:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
            batch_spatial_features = []
            batch_size = coords[:, 0].max().int().item() + 1
            for batch_idx in range(batch_size):
                spatial_feature = torch.zeros(
                    self.num_bev_features,
                    self.nz * self.nx * self.ny,
                    dtype=pillar_features.dtype,
                    device=pillar_features.device)

                batch_mask = coords[:, 0] == batch_idx
                this_coords = coords[batch_mask, :]
                indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
                indices = indices.type(torch.long)
                pillars = pillar_features[batch_mask, :]
                pillars = pillars.t()
                spatial_feature[:, indices] = pillars
                batch_spatial_features.append(spatial_feature)

            batch_spatial_features = torch.stack(batch_spatial_features, 0)
            batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
            batch_dict['spatial_features'] = batch_spatial_features
        elif 'common_pillar_features' in batch_dict:
            lidar_pillar_features, lidar_coords = batch_dict['lidar_pillar_features'], batch_dict['lidar_voxel_coords']
            radar_pillar_features, radar_coords = batch_dict['radar_pillar_features'], batch_dict['radar_voxel_coords']
            com_pillar_features, com_coords = batch_dict['common_pillar_features'], batch_dict['com_voxel_coords']
            com_pillar_features = com_pillar_features.reshape(-1,com_pillar_features.shape[-1])
            com_coords = com_coords.reshape(-1,com_coords.shape[-1])
            radar_pillar_features = radar_pillar_features.reshape(-1,radar_pillar_features.shape[-1])
            radar_coords = radar_coords.reshape(-1,radar_coords.shape[-1])
            lidar_batch_spatial_features = []
            radar_batch_spatial_features = []
            com_batch_spatial_features = []
            batch_size = lidar_coords[:, 0].max().int().item() + 1
            for lidar_batch_idx in range(batch_size):
                lidar_spatial_feature = torch.zeros(
                    self.num_bev_features[0],
                    self.nz * self.nx * self.ny,
                    dtype=lidar_pillar_features.dtype,
                    device=lidar_pillar_features.device)

                lidar_batch_mask = lidar_coords[:, 0] == lidar_batch_idx
                lidar_this_coords = lidar_coords[lidar_batch_mask, :]
                lidar_indices = lidar_this_coords[:, 1] + lidar_this_coords[:, 2] * self.nx + lidar_this_coords[:, 3]
                lidar_indices = lidar_indices.type(torch.long)
                lidar_pillars = lidar_pillar_features[lidar_batch_mask, :]
                lidar_pillars = lidar_pillars.t()
                lidar_spatial_feature[:, lidar_indices] = lidar_pillars
                lidar_batch_spatial_features.append(lidar_spatial_feature)
            for radar_batch_idx in range(batch_size):
                radar_spatial_feature = torch.zeros(
                    self.num_bev_features[2],
                    self.nz * self.nx * self.ny,
                    dtype=radar_pillar_features.dtype,
                    device=radar_pillar_features.device)

                radar_batch_mask = radar_coords[:, 0] == radar_batch_idx
                radar_this_coords = radar_coords[radar_batch_mask, :]
                radar_indices = radar_this_coords[:, 1] + radar_this_coords[:, 2] * self.nx + radar_this_coords[:, 3]
                radar_indices = radar_indices.type(torch.long)
                radar_pillars = radar_pillar_features[radar_batch_mask, :]
                radar_pillars = radar_pillars.t()
                radar_spatial_feature[:, radar_indices] = radar_pillars
                radar_batch_spatial_features.append(radar_spatial_feature)
            for com_batch_idx in range(batch_size):
                com_spatial_feature = torch.zeros(
                    self.num_bev_features[1],
                    self.nz * self.nx * self.ny,
                    dtype=com_pillar_features.dtype,
                    device=com_pillar_features.device)

                com_batch_mask = com_coords[:, 0] == com_batch_idx
                com_this_coords = com_coords[com_batch_mask, :]
                com_indices = com_this_coords[:, 1] + com_this_coords[:, 2] * self.nx + com_this_coords[:, 3]
                com_indices = com_indices.type(torch.long)
                com_pillars = com_pillar_features[com_batch_mask, :]
                com_pillars = com_pillars.t()
                com_spatial_feature[:, com_indices] = com_pillars
                com_batch_spatial_features.append(com_spatial_feature)
            lidar_batch_spatial_features = torch.stack(lidar_batch_spatial_features, 0)
            radar_batch_spatial_features = torch.stack(radar_batch_spatial_features, 0)
            com_batch_spatial_features = torch.stack(com_batch_spatial_features, 0)
            lidar_batch_spatial_features = lidar_batch_spatial_features.view(batch_size, self.num_bev_features[0] * self.nz, self.ny, self.nx)
            radar_batch_spatial_features = radar_batch_spatial_features.view(batch_size, self.num_bev_features[2] * self.nz, self.ny, self.nx)
            com_batch_spatial_features = com_batch_spatial_features.view(batch_size, self.num_bev_features[1] * self.nz, self.ny, self.nx)
            batch_dict['lidar_spatial_features'] = lidar_batch_spatial_features
            batch_dict['radar_spatial_features'] = radar_batch_spatial_features
            batch_dict['common_spatial_features'] = com_batch_spatial_features
            # t = np.sum(lidar_batch_spatial_features.detach().cpu().numpy(), axis=1)    
            # for k in range(t.shape[0]):
            #     t_map = t[k]
            #     t_min = np.min(t_map, axis=(0,1), keepdims=True)
            #     t_max = np.max(t_map, axis=(0,1), keepdims=True)
            #     t_map = (t_map - t_min) / (t_max - t_min)
            #     plt.imsave(f'attn_map_l.png', t_map, cmap='jet')
            # t = np.sum(radar_batch_spatial_features.detach().cpu().numpy(), axis=1)      
            # for k in range(t.shape[0]):
            #     t_map = t[k]
            #     t_min = np.min(t_map, axis=(0,1), keepdims=True)
            #     t_max = np.max(t_map, axis=(0,1), keepdims=True)
            #     t_map = (t_map - t_min) / (t_max - t_min)
            #     plt.imsave(f'attn_map_r.png', t_map, cmap='jet')
            batch_dict['spatial_features'] = torch.cat((lidar_batch_spatial_features, radar_batch_spatial_features, com_batch_spatial_features), 1)
        else:
            lidar_pillar_features, lidar_coords = batch_dict['lidar_pillar_features'], batch_dict['lidar_voxel_coords']
            radar_pillar_features, radar_coords = batch_dict['radar_pillar_features'], batch_dict['radar_voxel_coords']
            lidar_batch_spatial_features = []
            radar_batch_spatial_features = []
            radar_pillar_features = radar_pillar_features.reshape(-1,radar_pillar_features.shape[-1])
            radar_coords = radar_coords.reshape(-1,radar_coords.shape[-1])
            batch_size = lidar_coords[:, 0].max().int().item() + 1
            
            for lidar_batch_idx in range(batch_size):
                lidar_spatial_feature = torch.zeros(
                    self.num_bev_features[0],
                    self.nz * self.nx * self.ny,
                    dtype=lidar_pillar_features.dtype,
                    device=lidar_pillar_features.device)

                lidar_batch_mask = lidar_coords[:, 0] == lidar_batch_idx
                lidar_this_coords = lidar_coords[lidar_batch_mask, :]
                lidar_indices = lidar_this_coords[:, 1] + lidar_this_coords[:, 2] * self.nx + lidar_this_coords[:, 3]
                lidar_indices = lidar_indices.type(torch.long)
                lidar_pillars = lidar_pillar_features[lidar_batch_mask, :]
                lidar_pillars = lidar_pillars.t()
                lidar_spatial_feature[:, lidar_indices] = lidar_pillars
                lidar_batch_spatial_features.append(lidar_spatial_feature)
            for radar_batch_idx in range(batch_size):
                radar_spatial_feature = torch.zeros(
                    self.num_bev_features[1],
                    self.nz * self.nx * self.ny,
                    dtype=radar_pillar_features.dtype,
                    device=radar_pillar_features.device)

                radar_batch_mask = radar_coords[:, 0] == radar_batch_idx
                radar_this_coords = radar_coords[radar_batch_mask, :]
                radar_indices = radar_this_coords[:, 1] + radar_this_coords[:, 2] * self.nx + radar_this_coords[:, 3]
                radar_indices = radar_indices.type(torch.long)
                radar_pillars = radar_pillar_features[radar_batch_mask, :]
                radar_pillars = radar_pillars.t()
                radar_spatial_feature[:, radar_indices] = radar_pillars
                radar_batch_spatial_features.append(radar_spatial_feature)

            lidar_batch_spatial_features = torch.stack(lidar_batch_spatial_features, 0)
            radar_batch_spatial_features = torch.stack(radar_batch_spatial_features, 0)
            lidar_batch_spatial_features = lidar_batch_spatial_features.view(batch_size, self.num_bev_features[0] * self.nz, self.ny, self.nx)
            radar_batch_spatial_features = radar_batch_spatial_features.view(batch_size, self.num_bev_features[1] * self.nz, self.ny, self.nx)
            batch_dict['lidar_spatial_features'] = lidar_batch_spatial_features
            batch_dict['radar_spatial_features'] = radar_batch_spatial_features
            if 'de_lidar_pillar_features' in batch_dict.keys():
                de_lidar_pillar_features, de_lidar_coords = batch_dict['de_lidar_pillar_features'], batch_dict['de_lidar_voxel_coords']
                de_lidar_batch_spatial_features = []
                batch_size = de_lidar_coords[:, 0].max().int().item() + 1
                
                for de_lidar_batch_idx in range(batch_size):
                    de_lidar_spatial_feature = torch.zeros(
                        self.num_bev_features[0],
                        self.nz * self.nx * self.ny,
                        dtype=de_lidar_pillar_features.dtype,
                        device=de_lidar_pillar_features.device)

                    de_lidar_batch_mask = de_lidar_coords[:, 0] == de_lidar_batch_idx
                    de_lidar_this_coords = de_lidar_coords[de_lidar_batch_mask, :]
                    de_lidar_indices = de_lidar_this_coords[:, 1] + de_lidar_this_coords[:, 2] * self.nx + de_lidar_this_coords[:, 3]
                    de_lidar_indices = de_lidar_indices.type(torch.long)
                    de_lidar_pillars = de_lidar_pillar_features[de_lidar_batch_mask, :]
                    de_lidar_pillars = de_lidar_pillars.t()
                    de_lidar_spatial_feature[:, de_lidar_indices] = de_lidar_pillars
                    de_lidar_batch_spatial_features.append(de_lidar_spatial_feature)
                
                de_lidar_batch_spatial_features = torch.stack(de_lidar_batch_spatial_features, 0)
                de_lidar_batch_spatial_features = de_lidar_batch_spatial_features.view(batch_size, self.num_bev_features[0] * self.nz, self.ny, self.nx)
                batch_dict['de_lidar_spatial_features'] = de_lidar_batch_spatial_features

            batch_dict['spatial_features'] = torch.cat((lidar_batch_spatial_features, radar_batch_spatial_features), 1)
        return batch_dict


class PointPillarScatter3d(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict