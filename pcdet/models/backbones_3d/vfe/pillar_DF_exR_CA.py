import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .vfe_template import VFETemplate
import matplotlib.pyplot as plt
import math


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
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb

class Attention_Layer(nn.Module):
    def __init__(self, hidden_dim, pos_dim, head=4):
        super(Attention_Layer, self).__init__()
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        self.Q_linear = nn.Linear(self.hidden_dim + self.pos_dim, self.hidden_dim, bias=False)
        self.K_linear = nn.Linear(self.hidden_dim + self.pos_dim, self.hidden_dim, bias=False)
        self.V_linear = nn.Linear(self.hidden_dim + self.pos_dim, self.hidden_dim, bias=False)
        self.att = nn.MultiheadAttention(self.hidden_dim, head, batch_first=True)
        self.drop = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(self.hidden_dim)

        self.pos_embedding = nn.Sequential(
            nn.Linear(self.pos_dim * 3, self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )

    def forward(self, inputs, Q_in, input_coords, Q_in_coords): # 
        outs = []
        Q_in = Q_in.reshape(1, -1, Q_in.shape[-1]) #N,C
        if len(input_coords) ==0 :
            return torch.zeros([Q_in.shape[1], self.hidden_dim]).cuda()
        inputs = inputs.reshape(1, -1, inputs.shape[-1])
        batch_size = input_coords[:, 0].max().int().item() + 1
        
        feature = torch.zeros([Q_in.shape[1], self.hidden_dim]).cuda()
        for bs in range(batch_size):
            input_bs_mask = input_coords[:,0] == bs
            Q_in_bs_mask = Q_in_coords[:,0] == bs

            inputs_bs = inputs[:, input_bs_mask, :]
            pos_input = self.pos_embedding(pos2embed(input_coords[input_bs_mask, 1:], num_pos_feats=32))
            inputs_pos = torch.cat([inputs_bs, pos_input[None, :, :]], -1)

            Q_in_bs = Q_in[:, Q_in_bs_mask, :]
            pos_input = self.pos_embedding(pos2embed(Q_in_coords[Q_in_bs_mask, 1:], num_pos_feats=32))
            Q_in_pos = torch.cat([Q_in_bs, pos_input[None, :, :]], -1)

            Q = self.Q_linear(Q_in_pos)
            K = self.K_linear(inputs_pos)
            V = self.V_linear(inputs_pos)
            feature[Q_in_bs_mask, :] = self.norm(Q_in_bs + self.drop(self.att(Q, K, V)[0].squeeze()))
        
        return feature

class DFA_PillarVFE(VFETemplate):
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
        # LiDAR : x y z cx cy cz dx dy dz I
        # Radar : x y z cx cy cz dx dy dz V R t
        # Fusion : x y z Lcx Lcy Lcz mx my mz Rcx Rcy Rcz I V R t
        num_point_features_r = num_point_features_l + num_point_features_r - 6
        self.num_point_features_r = num_point_features_r
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
        self.l2r_attention_layers = Attention_Layer(list(self.num_filters)[-1], 32)
        
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
            len_radar = radar_voxel_num_points.max()
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
        
        # ###########################原本的Radar Voxel部分########################################
        # now_feature_idx = 0
        # com_features[len(L_coords):, :, now_feature_idx : now_feature_idx + 3] = radar_voxel_features[only_R, :, :3] #3
        # now_feature_idx += 3

        # #Intensity 为0 跳过
        # now_feature_idx += 1

        # #到lidar 点中心跳过
        # now_feature_idx += 3

        # #radar 体素center 
        # com_features[len(L_coords):, :, now_feature_idx : now_feature_idx + radar_f_center.shape[-1]] = radar_f_center[only_R] #3
        # now_feature_idx += radar_f_center.shape[-1]

        # #到radar 点中心
        # com_features[len(L_coords):, :, now_feature_idx : now_feature_idx + radar_f_cluster.shape[-1]] = radar_f_cluster[only_R] #3
        # now_feature_idx += radar_f_cluster.shape[-1]

        # #radar特征部分
        # com_features[len(L_coords):, :, now_feature_idx : now_feature_idx + radar_voxel_features.shape[-1]] = radar_voxel_features[only_R, :, 3:] #3
        # now_feature_idx += radar_voxel_features.shape[-1]

        #创建新的voxel_num_points和coords

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
        # T = 10
        # LR_pillar_weight = F.softmax(torch.cat([lidar_voxel_num_points[common_L, None], radar_voxel_num_points[common_R, None]],dim = 1) / T)
        # min_pillar_num = torch.min(lidar_voxel_num_points[common_L], radar_voxel_num_points[common_R])
        # dif_pillar_num = torch.abs(lidar_voxel_num_points[common_L] - radar_voxel_num_points[common_R])
        # eps = 1e-6
        # com_weight = torch.tanh(min_pillar_num / (dif_pillar_num + eps)).reshape(-1).cuda()
        # print(min_pillar_num.shape, dif_pillar_num.shape, com_weight.shape)
        # for i in range(len(common_L)):
        #     print(lidar_voxel_num_points[common_L[i]].item(), radar_voxel_num_points[common_R[i]].item(),LR_pillar_weight[i][0].item(),LR_pillar_weight[i][1].item(), com_weight[i].item())
        # com_features = com_features * com_weight
        # lidar_features[common_L] = lidar_features[common_L] * LR_pillar_weight[i][0]
        # radar_features[common_R] = radar_features[common_R] * LR_pillar_weight[i][1]
        l2r_features = self.l2r_attention_layers(
            lidar_features, 
            radar_features, 
            lidar_coords,
            radar_coords,
            )
        radar_features = radar_features + l2r_features
        batch_dict['lidar_pillar_features'] = lidar_features
        batch_dict['radar_pillar_features'] = radar_features
        return batch_dict
        