import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .vfe_template import VFETemplate
import matplotlib.pyplot as plt
import math

class interRAL(nn.Module):
    def __init__(self, channels):
        super(interRAL, self).__init__()
        self.linear = nn.Linear(10, channels, bias=True)
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = self.linear(x).permute(0, 2, 1)
        y = self.linear(y).permute(0, 2, 1)
        x_q = self.q_conv(x).permute(2, 0, 1) # b, n, c 
        y_k = self.k_conv(y).permute(2, 1, 0)# b, c, n        
        y_v = self.v_conv(y).permute(2, 0, 1)
        energy = torch.bmm(x_q, y_k) # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        y_r = torch.bmm(attention, y_v).permute(1, 2, 0) # b, c, n 
        y_r = self.act(self.after_norm(self.trans_conv(x - y_r)))
        x = x + y_r
        x = torch.max(x, dim=2, keepdim=True)[0]
        return x

class Radar7PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range,  **kwargs):
        super().__init__(model_cfg=model_cfg)

        num_point_features = 0
        self.use_norm = self.model_cfg.USE_NORM  # whether to use batchnorm in the PFNLayer
        self.use_xyz = self.model_cfg.USE_XYZ
        self.with_distance = self.model_cfg.USE_DISTANCE
        self.selected_indexes = []

        ## check if config has the correct params, if not, throw exception
        radar_config_params = ["USE_RCS", "USE_VR", "USE_VR_COMP", "USE_TIME", "USE_ELEVATION"]

        if all(hasattr(self.model_cfg, attr) for attr in radar_config_params):
            self.use_RCS = self.model_cfg.USE_RCS
            self.use_vr = self.model_cfg.USE_VR
            self.use_vr_comp = self.model_cfg.USE_VR_COMP
            self.use_time = self.model_cfg.USE_TIME
            self.use_elevation = self.model_cfg.USE_ELEVATION

        else:
            raise Exception("config does not have the right parameters, please use a radar config")

        self.available_features = ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']

        num_point_features += 6  # center_x, center_y, center_z, mean_x, mean_y, mean_z, time, we need 6 new

        self.x_ind = self.available_features.index('x')
        self.y_ind = self.available_features.index('y')
        self.z_ind = self.available_features.index('z')
        self.rcs_ind = self.available_features.index('rcs')
        self.vr_ind = self.available_features.index('v_r')
        self.vr_comp_ind = self.available_features.index('v_r_comp')
        self.time_ind = self.available_features.index('time')

        if self.use_xyz:  # if x y z coordinates are used, add 3 channels and save the indexes
            num_point_features += 3  # x, y, z
            self.selected_indexes.extend((self.x_ind, self.y_ind, self.z_ind))  # adding x y z channels to the indexes

        if self.use_RCS:  # add 1 if RCS is used and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.rcs_ind)  # adding  RCS channels to the indexes

        if self.use_vr:  # add 1 if vr is used and save the indexes. Note, we use compensated vr!
            num_point_features += 1
            self.selected_indexes.append(self.vr_ind)  # adding  v_r_comp channels to the indexes

        if self.use_vr_comp:  # add 1 if vr is used (as proxy for sensor cue) and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.vr_comp_ind)

        if self.use_time:  # add 1 if time is used and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.time_ind)  # adding  time channel to the indexes

        ### LOGGING USED FEATURES ###
        print("number of point features used: " + str(num_point_features))
        print("6 of these are 2 * (x y z)  coordinates realtive to mean and center of pillars")
        print(str(len(self.selected_indexes)) + " are selected original features: ")

        for k in self.selected_indexes:
            print(str(k) + ": " + self.available_features[k])

        self.selected_indexes = torch.LongTensor(self.selected_indexes)  # turning used indexes into Tensor

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        ## saving size of the voxel
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]

        ## saving offsets, start of point cloud in x, y, z + half a voxel, e.g. in y it starts around -39 m
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]  # number of outputs in last output channel

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        ## coordinate system notes
        # x is pointing forward, y is left right, z is up down
        # spconv returns voxel_coords as  [batch_idx, z_idx, y_idx, x_idx], that is why coords is indexed backwards

        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']

        if not self.use_elevation:  # if we ignore elevation (z) and v_z
            voxel_features[:, :, self.z_ind] = 0  # set z to zero before doing anything

        orig_xyz = voxel_features[:, :, :self.z_ind + 1]  # selecting x y z

        # calculate mean of points in pillars for x y z and save the offset from the mean
        # Note: they do not take the mean directly, as each pillar is filled up with 0-s. Instead, they sum and divide by num of points
        points_mean = orig_xyz.sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = orig_xyz - points_mean  # offset from cluster mean

        # calculate center for each pillar and save points' offset from the center. voxel_coordinate * voxel size + offset should be the center of pillar (coords are indexed backwards)
        f_center = torch.zeros_like(orig_xyz)
        f_center[:, :, 0] = voxel_features[:, :, self.x_ind] - (
                    coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, self.y_ind] - (
                    coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, self.z_ind] - (
                    coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        voxel_features = voxel_features[:, :, self.selected_indexes]  # filtering for used features

        features = [voxel_features, f_cluster, f_center]

        if self.with_distance:  # if with_distance is true, include range to the points as well
            points_dist = torch.norm(orig_xyz, 2, 2, keepdim=True)  # first 2: L2 norm second 2: along 2. dim
            features.append(points_dist)

        ## finishing up the feature extraction with correct shape and masking
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict



class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
  
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
            

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict


class BiDF_PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        self.use_preground_score = self.model_cfg.USE_RadarSCORE
        num_point_features_l = num_point_features[0]
        num_point_features_r = num_point_features[1]
        num_point_features_l += 6 if self.use_absolute_xyz else 3
        num_point_features_r += 6 if self.use_absolute_xyz else 3
        # center_x, center_y, center_z, mean_x, mean_y, mean_z we need 6 new
        if self.with_distance:
            num_point_features_l += 1
            num_point_features_r += 1
        if self.use_preground_score:
            num_point_features_r += 1
        # LiDAR : x y z cx cy cz dx dy dz I1
        # Radar : x y z cx cy cz dx dy dz I2 Score
        # Fusion : x y z Lcx Lcy Lcz mx my mz Rcx Rcy Rcz I1 I2 Score
        ex_point_features = num_point_features_l + num_point_features_r - 6
        num_point_features_r = ex_point_features
        num_point_features_l = ex_point_features
        self.num_point_features_r = num_point_features_r
        self.num_point_features_l = num_point_features_l
        print("common feature dim (use preground_score) = ", num_point_features_r)


        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features_l] + list(self.num_filters)

        l_pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            l_pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.l_pfn_layers = nn.ModuleList(l_pfn_layers)

        self.num_filters = self.model_cfg.NUM_FILTERS_Radar
        assert len(self.num_filters) > 0
        num_filters = [num_point_features_r] + list(self.num_filters)

        r_pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            r_pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.r_pfn_layers = nn.ModuleList(r_pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
  
        lidar_voxel_features, lidar_voxel_num_points, lidar_coords = batch_dict['lidar_voxels'], batch_dict['lidar_voxel_num_points'], batch_dict['lidar_voxel_coords']
        radar_voxel_features, radar_voxel_num_points, radar_coords = batch_dict['radar_voxels'], batch_dict['radar_voxel_num_points'], batch_dict['radar_voxel_coords']
        L_coords = lidar_coords[:,:]
        R_coords = radar_coords[:,:]

        lidar_points_mean = lidar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / lidar_voxel_num_points.type_as(lidar_voxel_features).view(-1, 1, 1)
        radar_points_mean = radar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / radar_voxel_num_points.type_as(radar_voxel_features).view(-1, 1, 1)
        lidar_f_cluster = lidar_voxel_features[:, :, :3] - lidar_points_mean
        radar_f_cluster = radar_voxel_features[:, :, :3] - radar_points_mean

        lidar_f_center = torch.zeros_like(lidar_voxel_features[:, :, :3])
        radar_f_center = torch.zeros_like(radar_voxel_features[:, :, :3])
        lidar_f_center[:, :, 0] = lidar_voxel_features[:, :, 0] - (lidar_coords[:, 3].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        lidar_f_center[:, :, 1] = lidar_voxel_features[:, :, 1] - (lidar_coords[:, 2].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        lidar_f_center[:, :, 2] = lidar_voxel_features[:, :, 2] - (lidar_coords[:, 1].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        radar_f_center[:, :, 0] = radar_voxel_features[:, :, 0] - (radar_coords[:, 3].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        radar_f_center[:, :, 1] = radar_voxel_features[:, :, 1] - (radar_coords[:, 2].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        radar_f_center[:, :, 2] = radar_voxel_features[:, :, 2] - (radar_coords[:, 1].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        dist_matrix = torch.sum((L_coords.unsqueeze(1) - R_coords)**2, dim=2)  

        # 找到距离为0的点,即A和B中相同的点  
        common_L, common_R = torch.where(dist_matrix==0)

        # mask = torch.ones(len(L_coords)).bool() 
        # mask[common_L] = False
        # only_L = torch.where(mask)[0].long()
        # # 找到R中独有的点
        # mask = torch.ones(len(R_coords)).bool()
        # mask[common_R] = False
        # only_R = torch.where(mask)[0].long()

        
        
        #print(len(L_coords), len(R_coords), len(only_L), len(only_R), len(common_L), len(common_R))
        #接下来把Lidar合并到radar voxel（包括特征合并）
        len_radar = 1
        if len(radar_voxel_num_points) > 0:
            len_radar = int(radar_voxel_num_points.max())

        com_features = torch.zeros((len(radar_voxel_num_points), len_radar, self.num_point_features_r)).cuda()
        
        #用radar的部分覆盖（radar点比较少，一般1~5，最多5个点，一次次来）  
        for i in range(len_radar):
            now_feature_idx = 0
            valid_mask = radar_voxel_num_points[common_R] >= i+1 #只覆盖非空的点
            valid_common_R = common_R[valid_mask]
            valid_common_L = common_L[valid_mask]
            #print(radar_voxel_features[valid_common_R[0], i, :3])

            
            com_features[:, i, now_feature_idx : now_feature_idx + 3] = radar_voxel_features[:, i, :3] #3
            now_feature_idx += 3

            #Intensity 覆盖为均值（radar部分的lidar特征设置为0）
            extraF_L = lidar_voxel_features[valid_common_L, :, 3:].sum(dim=1) / lidar_voxel_num_points[valid_common_L].type_as(lidar_voxel_features).view(-1, 1)
            com_features[valid_common_R, i, now_feature_idx : now_feature_idx + 1] = extraF_L #1
            now_feature_idx += 1
            
            # com_features[valid_common_L, replaced_idx, now_feature_idx : now_feature_idx + 1] = 0 #1
            # now_feature_idx += 1


            #radar to lidar偏移
            common_lidar_points_mean = lidar_voxel_features[valid_common_L, :, :3].sum(dim=1) / lidar_voxel_num_points[valid_common_L].type_as(lidar_voxel_features).view(-1, 1)
            radartolidar_f_cluster = radar_voxel_features[valid_common_R, i, :3] - common_lidar_points_mean
            com_features[valid_common_R, i, now_feature_idx : now_feature_idx + radartolidar_f_cluster.shape[-1]] = radartolidar_f_cluster #3
            now_feature_idx += radartolidar_f_cluster.shape[-1]
            
            com_features[:, i, now_feature_idx : now_feature_idx + radar_f_center.shape[-1]] = radar_f_center[:, i] #3
            now_feature_idx += radar_f_center.shape[-1]

            com_features[:, i, now_feature_idx : now_feature_idx + radar_f_cluster.shape[-1]] = radar_f_cluster[:, i] #3
            now_feature_idx += radar_f_cluster.shape[-1]

            
            #radar特征部分修改
            com_features[:, i, now_feature_idx : now_feature_idx + radar_voxel_features.shape[-1] - 3] = radar_voxel_features[:, i, 3:]
            now_feature_idx += radartolidar_f_cluster.shape[-1]
        
        len_lidar = int(lidar_voxel_num_points.max())
        l_ex_features = torch.zeros((len(lidar_voxel_num_points), 32, self.num_point_features_l)).cuda()
        now_feature_idx = 0
        l_ex_features[:, :, now_feature_idx : now_feature_idx + lidar_voxel_features.shape[-1]] = lidar_voxel_features #4
        now_feature_idx += lidar_voxel_features.shape[-1]
        #print(now_feature_idx)

        l_ex_features[:, :, now_feature_idx : now_feature_idx + lidar_f_cluster.shape[-1]] = lidar_f_cluster #3
        now_feature_idx += lidar_f_cluster.shape[-1]
        #print(now_feature_idx)

        l_ex_features[:, :, now_feature_idx : now_feature_idx + lidar_f_center.shape[-1]] = lidar_f_center #3
        now_feature_idx += lidar_f_center.shape[-1]
        #print(now_feature_idx)

        #计算lidar to radar共同部分中的cluster和feature均值用于特征传播
        #(N,3); (N,feature_dim-1)
        mask = self.get_paddings_indicator(radar_voxel_num_points[common_R], 32, axis=0)
        mask = mask & (radar_voxel_features[common_R, :, -2] == -1) 
        # 求valid且t=0的mask并求和算有多少个，t在-2维度
        num_valid = mask.sum(dim=1)
        l2r_mask = (num_valid > 0)

        l2r_com_L = common_L[l2r_mask]
        l2r_com_R = common_R[l2r_mask]
        num_valid = num_valid[l2r_mask]
        mask = mask[l2r_mask].unsqueeze(-1)
        #计算lidar to radar共同部分的cluster(注意只有common(L，R)的部分才有)

        common_radar_points_mean = (radar_voxel_features[l2r_com_R, :, :3] * mask).sum(dim=1, keepdim=True) / num_valid.type_as(radar_voxel_features).view(-1, 1, 1)
        lidartoradar_f_cluster = lidar_voxel_features[l2r_com_L, :, :3] - common_radar_points_mean

        l_ex_features[l2r_com_L, :, now_feature_idx : now_feature_idx + lidartoradar_f_cluster.shape[-1]] = lidartoradar_f_cluster #3
        now_feature_idx += lidartoradar_f_cluster.shape[-1]
        #radar特征部分先填充均值后面是radar的话会覆盖
        extraFea_R = (radar_voxel_features[l2r_com_R, :, 3:] * mask).sum(dim=1, keepdim=True) / num_valid.type_as(radar_voxel_features).view(-1, 1, 1)
        l_ex_features[l2r_com_L, :, now_feature_idx :] = extraFea_R #4
        now_feature_idx += extraFea_R.shape[-1]

        lidar_features = l_ex_features
        final_voxel_count = lidar_features.shape[1]
        mask = self.get_paddings_indicator(lidar_voxel_num_points, final_voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(lidar_features)
        lidar_features *= mask
        for pfn in self.l_pfn_layers:
            lidar_features = pfn(lidar_features)
        lidar_features = lidar_features.squeeze()

        radar_features = com_features
        final_voxel_count = radar_features.shape[1]
        mask = self.get_paddings_indicator(radar_voxel_num_points, final_voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(radar_features)
        radar_features *= mask
        for pfn in self.r_pfn_layers:
            radar_features = pfn(radar_features)
        radar_features = radar_features.squeeze()

        batch_dict['lidar_pillar_features'] = lidar_features
        batch_dict['radar_pillar_features'] = radar_features
        return batch_dict




class PillarVFE_CA(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        num_point_features_l = num_point_features[0]
        num_point_features_r = num_point_features[1]
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features_l += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features_l += 1
        self.use_preground_score = self.model_cfg.USE_RadarSCORE
        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features_l] + list(self.num_filters)

        lidar_pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            lidar_pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.lidar_pfn_layers = nn.ModuleList(lidar_pfn_layers)




        num_point_features_r = 0
        self.use_norm = self.model_cfg.USE_NORM  # whether to use batchnorm in the PFNLayer
        self.use_xyz = self.model_cfg.USE_XYZ
        self.with_distance = self.model_cfg.USE_DISTANCE
        self.selected_indexes = []

        ## check if config has the correct params, if not, throw exception
        radar_config_params = ["USE_RCS", "USE_VR", "USE_VR_COMP", "USE_TIME", "USE_ELEVATION"]

        if all(hasattr(self.model_cfg, attr) for attr in radar_config_params):
            self.use_RCS = self.model_cfg.USE_RCS
            self.use_vr = self.model_cfg.USE_VR
            self.use_vr_comp = self.model_cfg.USE_VR_COMP
            self.use_time = self.model_cfg.USE_TIME
            self.use_elevation = self.model_cfg.USE_ELEVATION

        else:
            raise Exception("config does not have the right parameters, please use a radar config")

        self.available_features = ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']

        num_point_features_r += 6  # center_x, center_y, center_z, mean_x, mean_y, mean_z, time, we need 6 new

        self.x_ind = self.available_features.index('x')
        self.y_ind = self.available_features.index('y')
        self.z_ind = self.available_features.index('z')
        self.rcs_ind = self.available_features.index('rcs')
        self.vr_ind = self.available_features.index('v_r')
        self.vr_comp_ind = self.available_features.index('v_r_comp')
        self.time_ind = self.available_features.index('time')

        if self.use_xyz:  # if x y z coordinates are used, add 3 channels and save the indexes
            num_point_features_r += 3  # x, y, z
            self.selected_indexes.extend((self.x_ind, self.y_ind, self.z_ind))  # adding x y z channels to the indexes

        if self.use_RCS:  # add 1 if RCS is used and save the indexes
            num_point_features_r += 1
            self.selected_indexes.append(self.rcs_ind)  # adding  RCS channels to the indexes

        if self.use_vr:  # add 1 if vr is used and save the indexes. Note, we use compensated vr!
            num_point_features_r += 1
            self.selected_indexes.append(self.vr_ind)  # adding  v_r_comp channels to the indexes

        if self.use_vr_comp:  # add 1 if vr is used (as proxy for sensor cue) and save the indexes
            num_point_features_r += 1
            self.selected_indexes.append(self.vr_comp_ind)

        if self.use_time:  # add 1 if time is used and save the indexes
            num_point_features_r += 1
            self.selected_indexes.append(self.time_ind)  # adding  time channel to the indexes

        ### LOGGING USED FEATURES ###
        print("number of point features used: " + str(num_point_features_r))
        print("6 of these are 2 * (x y z)  coordinates realtive to mean and center of pillars")
        print(str(len(self.selected_indexes)) + " are selected original features: ")
        if self.use_preground_score:
            num_point_features_r += 1
        for k in self.selected_indexes:
            print(str(k) + ": " + self.available_features[k])

        self.selected_indexes = torch.LongTensor(self.selected_indexes)  # turning used indexes into Tensor

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features_r] + list(self.num_filters)

        radar_pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            radar_pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.radar_pfn_layers = nn.ModuleList(radar_pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        self.l2r_attention_layers = Attention_Layer(list(self.num_filters)[-1], 64)
        self.r2l_attention_layers = Attention_Layer(list(self.num_filters)[-1], 64)

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        if 'voxels' in batch_dict:
            voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
            points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
            f_cluster = voxel_features[:, :, :3] - points_mean

            f_center = torch.zeros_like(voxel_features[:, :, :3])
            f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

            if self.use_absolute_xyz:
                features = [voxel_features, f_cluster, f_center]
            else:
                features = [voxel_features[..., 3:], f_cluster, f_center]
            
            if self.with_distance:
                points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
                features.append(points_dist)
            features = torch.cat(features, dim=-1)
            
            voxel_count = features.shape[1]
            mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
            mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
            features *= mask
            for pfn in self.pfn_layers:
                features = pfn(features)
            features = features.squeeze()
            batch_dict['pillar_features'] = features
        else:
            lidar_voxel_features, lidar_voxel_num_points, lidar_coords = batch_dict['lidar_voxels'], batch_dict['lidar_voxel_num_points'], batch_dict['lidar_voxel_coords']
            radar_voxel_features, radar_voxel_num_points, radar_coords = batch_dict['radar_voxels'], batch_dict['radar_voxel_num_points'], batch_dict['radar_voxel_coords']
            lidar_points_mean = lidar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / lidar_voxel_num_points.type_as(lidar_voxel_features).view(-1, 1, 1)
            radar_points_mean = radar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / radar_voxel_num_points.type_as(radar_voxel_features).view(-1, 1, 1)
            lidar_f_cluster = lidar_voxel_features[:, :, :3] - lidar_points_mean
            radar_f_cluster = radar_voxel_features[:, :, :3] - radar_points_mean

            lidar_f_center = torch.zeros_like(lidar_voxel_features[:, :, :3])
            radar_f_center = torch.zeros_like(radar_voxel_features[:, :, :3])
            lidar_f_center[:, :, 0] = lidar_voxel_features[:, :, 0] - (lidar_coords[:, 3].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            lidar_f_center[:, :, 1] = lidar_voxel_features[:, :, 1] - (lidar_coords[:, 2].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            lidar_f_center[:, :, 2] = lidar_voxel_features[:, :, 2] - (lidar_coords[:, 1].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
            radar_f_center[:, :, 0] = radar_voxel_features[:, :, 0] - (radar_coords[:, 3].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            radar_f_center[:, :, 1] = radar_voxel_features[:, :, 1] - (radar_coords[:, 2].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            radar_f_center[:, :, 2] = radar_voxel_features[:, :, 2] - (radar_coords[:, 1].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)


            if self.use_absolute_xyz:
                lidar_features = [lidar_voxel_features, lidar_f_cluster, lidar_f_center]
            else:
                lidar_features = [lidar_voxel_features[..., 3:], lidar_f_cluster, lidar_f_center]
            if self.use_absolute_xyz:
                radar_features = [radar_voxel_features, radar_f_cluster, radar_f_center]
            else:
                radar_features = [radar_voxel_features[..., 3:], radar_f_cluster, radar_f_center]


            if self.with_distance:
                lidar_points_dist = torch.norm(lidar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                lidar_features.append(lidar_points_dist)
            lidar_features = torch.cat(lidar_features, dim=-1)
            if self.with_distance:
                radar_points_dist = torch.norm(radar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                radar_features.append(radar_points_dist)
            radar_features = torch.cat(radar_features, dim=-1)

            lidar_voxel_count = lidar_features.shape[1]
            radar_voxel_count = radar_features.shape[1]
            lidar_mask = self.get_paddings_indicator(lidar_voxel_num_points, lidar_voxel_count, axis=0)
            radar_mask = self.get_paddings_indicator(radar_voxel_num_points, radar_voxel_count, axis=0)
            lidar_mask = torch.unsqueeze(lidar_mask, -1).type_as(lidar_voxel_features)
            radar_mask = torch.unsqueeze(radar_mask, -1).type_as(radar_voxel_features)
            lidar_features *= lidar_mask
            radar_features *= radar_mask

            for pfn in self.lidar_pfn_layers:
                lidar_features = pfn(lidar_features)
            lidar_features = lidar_features.squeeze()
            for pfn in self.radar_pfn_layers:
                radar_features = pfn(radar_features)
            radar_features = radar_features.squeeze()

            # safusionlayer2
            # lidar_features_output = self.interral(lidar_features, radar_features)
            # radar_features_output = self.interral(radar_features, lidar_features)
            # lidar_features = lidar_features_output.view([lidar_features_output.size()[0], lidar_features_output.size()[1]])
            # radar_features = radar_features_output.view([radar_features_output.size()[0], radar_features_output.size()[1]])
            radar_features = self.l2r_attention_layers(
                lidar_features, 
                radar_features, 
                lidar_coords,
                radar_coords
            )
            lidar_features = self.r2l_attention_layers(
                radar_features, 
                lidar_features, 
                radar_coords,
                lidar_coords
                )
            batch_dict['lidar_pillar_features'] = lidar_features
            batch_dict['radar_pillar_features'] = radar_features
        return batch_dict
        

class Fusion_PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        num_point_features_l = num_point_features[0]
        num_point_features_r = num_point_features[1]
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features_l += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features_l += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features_l] + list(self.num_filters)
        lidar_pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            lidar_pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.lidar_pfn_layers = nn.ModuleList(lidar_pfn_layers)
        radar_pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            radar_pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.radar_pfn_layers = nn.ModuleList(radar_pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        if 'voxels' in batch_dict:
            voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
            points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
            f_cluster = voxel_features[:, :, :3] - points_mean

            f_center = torch.zeros_like(voxel_features[:, :, :3])
            f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

            if self.use_absolute_xyz:
                features = [voxel_features, f_cluster, f_center]
            else:
                features = [voxel_features[..., 3:], f_cluster, f_center]
            
            if self.with_distance:
                points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
                features.append(points_dist)
            features = torch.cat(features, dim=-1)
            
            voxel_count = features.shape[1]
            mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
            mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
            features *= mask
            for pfn in self.pfn_layers:
                features = pfn(features)
            features = features.squeeze()
            batch_dict['pillar_features'] = features
        else:
            lidar_voxel_features, lidar_voxel_num_points, lidar_coords = batch_dict['lidar_voxels'], batch_dict['lidar_voxel_num_points'], batch_dict['lidar_voxel_coords']
            radar_voxel_features, radar_voxel_num_points, radar_coords = batch_dict['radar_voxels'], batch_dict['radar_voxel_num_points'], batch_dict['radar_voxel_coords']
            lidar_points_mean = lidar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / lidar_voxel_num_points.type_as(lidar_voxel_features).view(-1, 1, 1)
            radar_points_mean = radar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / radar_voxel_num_points.type_as(radar_voxel_features).view(-1, 1, 1)
            lidar_f_cluster = lidar_voxel_features[:, :, :3] - lidar_points_mean
            radar_f_cluster = radar_voxel_features[:, :, :3] - radar_points_mean

            lidar_f_center = torch.zeros_like(lidar_voxel_features[:, :, :3])
            radar_f_center = torch.zeros_like(radar_voxel_features[:, :, :3])
            lidar_f_center[:, :, 0] = lidar_voxel_features[:, :, 0] - (lidar_coords[:, 3].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            lidar_f_center[:, :, 1] = lidar_voxel_features[:, :, 1] - (lidar_coords[:, 2].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            lidar_f_center[:, :, 2] = lidar_voxel_features[:, :, 2] - (lidar_coords[:, 1].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
            radar_f_center[:, :, 0] = radar_voxel_features[:, :, 0] - (radar_coords[:, 3].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            radar_f_center[:, :, 1] = radar_voxel_features[:, :, 1] - (radar_coords[:, 2].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            radar_f_center[:, :, 2] = radar_voxel_features[:, :, 2] - (radar_coords[:, 1].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)


            if self.use_absolute_xyz:
                lidar_features = [lidar_voxel_features, lidar_f_cluster, lidar_f_center]
            else:
                lidar_features = [lidar_voxel_features[..., 3:], lidar_f_cluster, lidar_f_center]
            if self.use_absolute_xyz:
                radar_features = [radar_voxel_features, radar_f_cluster, radar_f_center]
            else:
                radar_features = [radar_voxel_features[..., 3:], radar_f_cluster, radar_f_center]


            if self.with_distance:
                lidar_points_dist = torch.norm(lidar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                lidar_features.append(lidar_points_dist)
            lidar_features = torch.cat(lidar_features, dim=-1)
            if self.with_distance:
                radar_points_dist = torch.norm(radar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                radar_features.append(radar_points_dist)
            radar_features = torch.cat(radar_features, dim=-1)

            lidar_voxel_count = lidar_features.shape[1]
            radar_voxel_count = radar_features.shape[1]
            lidar_mask = self.get_paddings_indicator(lidar_voxel_num_points, lidar_voxel_count, axis=0)
            radar_mask = self.get_paddings_indicator(radar_voxel_num_points, radar_voxel_count, axis=0)
            lidar_mask = torch.unsqueeze(lidar_mask, -1).type_as(lidar_voxel_features)
            radar_mask = torch.unsqueeze(radar_mask, -1).type_as(radar_voxel_features)
            lidar_features *= lidar_mask
            radar_features *= radar_mask

            for pfn in self.lidar_pfn_layers:
                lidar_features = pfn(lidar_features)
            lidar_features = lidar_features.squeeze()
            for pfn in self.radar_pfn_layers:
                radar_features = pfn(radar_features)
            radar_features = radar_features.squeeze()
            

            if 'de_lidar_voxels' in batch_dict.keys() :
                with torch.no_grad():
                    de_lidar_voxel_features, de_lidar_voxel_num_points, de_lidar_coords = batch_dict['de_lidar_voxels'], batch_dict['de_lidar_voxel_num_points'], batch_dict['de_lidar_voxel_coords']
                    de_lidar_points_mean = de_lidar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / de_lidar_voxel_num_points.type_as(de_lidar_voxel_features).view(-1, 1, 1)
                    de_lidar_f_cluster = de_lidar_voxel_features[:, :, :3] - de_lidar_points_mean
                    
                    de_lidar_f_center = torch.zeros_like(de_lidar_voxel_features[:, :, :3])
                    de_lidar_f_center[:, :, 0] = de_lidar_voxel_features[:, :, 0] - (de_lidar_coords[:, 3].to(de_lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
                    de_lidar_f_center[:, :, 1] = de_lidar_voxel_features[:, :, 1] - (de_lidar_coords[:, 2].to(de_lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
                    de_lidar_f_center[:, :, 2] = de_lidar_voxel_features[:, :, 2] - (de_lidar_coords[:, 1].to(de_lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
                    
                    if self.use_absolute_xyz:
                        de_lidar_features = [de_lidar_voxel_features, de_lidar_f_cluster, de_lidar_f_center]
                    else:
                        de_lidar_features = [de_lidar_voxel_features[..., 3:], de_lidar_f_cluster, de_lidar_f_center]

                    if self.with_distance:
                        de_lidar_points_dist = torch.norm(de_lidar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                        de_lidar_features.append(de_lidar_points_dist)
                    de_lidar_features = torch.cat(de_lidar_features, dim=-1)
                    
                    de_lidar_voxel_count = de_lidar_features.shape[1]
                    de_lidar_mask = self.get_paddings_indicator(de_lidar_voxel_num_points, de_lidar_voxel_count, axis=0)
                    de_lidar_mask = torch.unsqueeze(de_lidar_mask, -1).type_as(de_lidar_voxel_features)
                    de_lidar_features *= de_lidar_mask
                    
                    for pfn in self.lidar_pfn_layers:
                        de_lidar_features = pfn(de_lidar_features)
                    de_lidar_features = de_lidar_features.squeeze()
                    batch_dict['de_lidar_pillar_features'] = de_lidar_features
                
            # safusionlayer2
            # lidar_features_output = self.interral(lidar_features, radar_features)
            # radar_features_output = self.interral(radar_features, lidar_features)
            # lidar_features = lidar_features_output.view([lidar_features_output.size()[0], lidar_features_output.size()[1]])
            # radar_features = radar_features_output.view([radar_features_output.size()[0], radar_features_output.size()[1]])

            batch_dict['lidar_pillar_features'] = lidar_features
            batch_dict['radar_pillar_features'] = radar_features
        return batch_dict
        
class PositionalEmbedding(nn.Module):
    def __init__(self, demb=256):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    # pos_seq =  pos_seq = torch.arange(seq_len-1, -1, -1.0)
    def forward(self, pos_seq, batch_size=2):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if batch_size is not None:
            return pos_emb[None, :, :].expand(batch_size, -1, -1)
        else:
            return pos_emb[None, :, :]
class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


def pos2embed(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
    pos_z = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb

class Attention_Layer(nn.Module):
    def __init__(self, hidden_dim, pos_dim, head=4):
        super(Attention_Layer, self).__init__()
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        if self.pos_dim > 0:
            self.Q_linear = nn.Linear(self.hidden_dim + self.pos_dim, self.hidden_dim, bias=False)
            self.K_linear = nn.Linear(self.hidden_dim + self.pos_dim, self.hidden_dim, bias=False)
            self.V_linear = nn.Linear(self.hidden_dim + self.pos_dim, self.hidden_dim, bias=False)
            self.pos_embedding = nn.Sequential(
                nn.Linear(self.pos_dim * 2, self.pos_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.pos_dim, self.pos_dim)
            )
        else:
            
            self.Q_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.K_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.V_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.att = nn.MultiheadAttention(self.hidden_dim, head, batch_first=True)
        self.drop = nn.Dropout(0.05)
        self.norm = nn.LayerNorm(self.hidden_dim * 2)
        self.Align = nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False)
 

    def forward(self, inputs, Q_in, input_coords, Q_in_coords, fog_intensity = None): # 
        outs = []
        Q_in = Q_in.reshape(1, -1, Q_in.shape[-1]) #N,C
        if len(input_coords) ==0 :
            return torch.zeros([Q_in.shape[1], self.hidden_dim]).cuda()
        inputs = inputs.reshape(1, -1, inputs.shape[-1])
        batch_size = input_coords[:, 0].max().int().item() + 1
        feature = torch.zeros([Q_in.shape[1], self.hidden_dim * 2]).cuda()
        for bs in range(batch_size):
            input_bs_mask = input_coords[:,0] == bs
            Q_in_bs_mask = Q_in_coords[:,0] == bs

            inputs_bs = inputs[:, input_bs_mask, :]
            K = inputs_bs

            Q_in_bs = Q_in[:, Q_in_bs_mask, :]
            Q = Q_in_bs
            if self.pos_dim > 0:
                pos_input = self.pos_embedding(pos2embed(input_coords[input_bs_mask, 1:], num_pos_feats=self.pos_dim))
                K = torch.cat([inputs_bs, pos_input[None, :, :]], -1)

            if self.pos_dim > 0:
                pos_input = self.pos_embedding(pos2embed(Q_in_coords[Q_in_bs_mask, 1:], num_pos_feats=self.pos_dim))
                Q = torch.cat([Q_in_bs, pos_input[None, :, :]], -1)

            q = self.Q_linear(Q)
            k = self.K_linear(K)
            v = self.V_linear(K)
            if fog_intensity is not None:
                feature[Q_in_bs_mask, :] = self.norm(Q_in_bs + self.drop(self.att(q, k, v)[0].squeeze() * fog_intensity[bs]))
            else:
                feature[Q_in_bs_mask, :] = torch.cat([Q_in_bs, self.drop(self.att(q, k, v)[0])], dim = -1)
        feature = self.Align(self.norm(feature))
        return feature

class InterF_PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        num_point_features_l = num_point_features[0]
        num_point_features_r = num_point_features[1]
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features_l += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features_l += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features_l] + list(self.num_filters)
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.interral = interRAL(64)    # set the channel number of interRAL

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        if 'voxels' in batch_dict:
            voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
            points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
            f_cluster = voxel_features[:, :, :3] - points_mean

            f_center = torch.zeros_like(voxel_features[:, :, :3])
            f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

            if self.use_absolute_xyz:
                features = [voxel_features, f_cluster, f_center]
            else:
                features = [voxel_features[..., 3:], f_cluster, f_center]

            if self.with_distance:
                points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
                features.append(points_dist)
            features = torch.cat(features, dim=-1)

            voxel_count = features.shape[1]
            mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
            mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
            features *= mask
            for pfn in self.pfn_layers:
                features = pfn(features)
            features = features.squeeze()
            batch_dict['pillar_features'] = features
        else:
            lidar_voxel_features, lidar_voxel_num_points, lidar_coords = batch_dict['lidar_voxels'], batch_dict['lidar_voxel_num_points'], batch_dict['lidar_voxel_coords']
            radar_voxel_features, radar_voxel_num_points, radar_coords = batch_dict['radar_voxels'], batch_dict['radar_voxel_num_points'], batch_dict['radar_voxel_coords']
            lidar_points_mean = lidar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / lidar_voxel_num_points.type_as(lidar_voxel_features).view(-1, 1, 1)
            radar_points_mean = radar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / radar_voxel_num_points.type_as(radar_voxel_features).view(-1, 1, 1)
            lidar_f_cluster = lidar_voxel_features[:, :, :3] - lidar_points_mean
            radar_f_cluster = radar_voxel_features[:, :, :3] - radar_points_mean

            lidar_f_center = torch.zeros_like(lidar_voxel_features[:, :, :3])
            radar_f_center = torch.zeros_like(radar_voxel_features[:, :, :3])
            lidar_f_center[:, :, 0] = lidar_voxel_features[:, :, 0] - (lidar_coords[:, 3].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            lidar_f_center[:, :, 1] = lidar_voxel_features[:, :, 1] - (lidar_coords[:, 2].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            lidar_f_center[:, :, 2] = lidar_voxel_features[:, :, 2] - (lidar_coords[:, 1].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
            radar_f_center[:, :, 0] = radar_voxel_features[:, :, 0] - (radar_coords[:, 3].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            radar_f_center[:, :, 1] = radar_voxel_features[:, :, 1] - (radar_coords[:, 2].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            radar_f_center[:, :, 2] = radar_voxel_features[:, :, 2] - (radar_coords[:, 1].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)


            if self.use_absolute_xyz:
                lidar_features = [lidar_voxel_features, lidar_f_cluster, lidar_f_center]
            else:
                lidar_features = [lidar_voxel_features[..., 3:], lidar_f_cluster, lidar_f_center]
            if self.use_absolute_xyz:
                radar_features = [radar_voxel_features, radar_f_cluster, radar_f_center]
            else:
                radar_features = [radar_voxel_features[..., 3:], radar_f_cluster, radar_f_center]


            if self.with_distance:
                lidar_points_dist = torch.norm(lidar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                lidar_features.append(lidar_points_dist)
            lidar_features = torch.cat(lidar_features, dim=-1)
            if self.with_distance:
                radar_points_dist = torch.norm(radar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                radar_features.append(radar_points_dist)
            radar_features = torch.cat(radar_features, dim=-1)

            lidar_voxel_count = lidar_features.shape[1]
            radar_voxel_count = radar_features.shape[1]
            lidar_mask = self.get_paddings_indicator(lidar_voxel_num_points, lidar_voxel_count, axis=0)
            radar_mask = self.get_paddings_indicator(radar_voxel_num_points, radar_voxel_count, axis=0)
            lidar_mask = torch.unsqueeze(lidar_mask, -1).type_as(lidar_voxel_features)
            radar_mask = torch.unsqueeze(radar_mask, -1).type_as(radar_voxel_features)
            lidar_features *= lidar_mask
            radar_features *= radar_mask

            # for pfn in self.lidar_pfn_layers:
            #     lidar_features = pfn(lidar_features)
            # lidar_features = lidar_features.squeeze()
            # for pfn in self.radar_pfn_layers:
            #     radar_features = pfn(radar_features)
            # radar_features = radar_features.squeeze()

            #safusionlayer2
            lidar_features_output = self.interral(lidar_features, radar_features)
            radar_features_output = self.interral(radar_features, lidar_features)
            lidar_features = lidar_features_output.view([lidar_features_output.size()[0], lidar_features_output.size()[1]])
            radar_features = radar_features_output.view([radar_features_output.size()[0], radar_features_output.size()[1]])

            batch_dict['lidar_pillar_features'] = lidar_features
            batch_dict['radar_pillar_features'] = radar_features
        return batch_dict
