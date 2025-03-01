'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import torch
import torch.nn as nn
import os
import torch
import torch.nn as nn
import numpy as np




from spconv.pytorch.utils import PointToVoxel

tv = None
try:
    import cumm.tensorview as tv
except:
    pass

class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points

class MMESparseProcessor(nn.Module):
    def __init__(self, cfg):
        super(MMESparseProcessor, self).__init__()
        self.cfg = cfg
        self.model_cfg = cfg.MODEL
        self.dataset_cfg = cfg.DATASET
        self.use_preground_score = self.model_cfg.PRE_PROCESSING.USE_RadarSCORE
        # class
        self.num_class = 0
        self.class_names = []

        self.is_pre_processing = self.model_cfg.PRE_PROCESSING.get('VER', None)
        self.shuffle_points = self.model_cfg.PRE_PROCESSING.get('SHUFFLE_POINTS', False)
        self.transform_points_to_voxels = self.model_cfg.PRE_PROCESSING.get('TRANSFORM_POINTS_TO_VOXELS', False)
        
        self.is_logging = cfg.GENERAL.LOGGING.IS_LOGGING
        
        num_point_features = [self.dataset_cfg.ldr64.n_used,self.dataset_cfg.rdr_sparse.n_used + 1]
        self.num_point_features = num_point_features
        point_cloud_range = np.array(self.dataset_cfg.roi.xyz)
        voxel_size = self.dataset_cfg.roi.voxel_size
        grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)
        model_info_dict = dict(
            module_list = [],
            num_rawpoint_features = num_point_features,
            num_point_features = num_point_features,
            grid_size = grid_size,
            point_cloud_range = point_cloud_range,
            voxel_size = voxel_size,
        )
        self.ldr_voxel_generator_train = VoxelGeneratorWrapper(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=num_point_features[0],
            max_num_points_per_voxel=self.model_cfg.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
            max_num_voxels=self.model_cfg.PRE_PROCESSING.MAX_NUMBER_OF_VOXELS['train'],
        )
        self.ldr_voxel_generator_test = VoxelGeneratorWrapper(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=num_point_features[0],
            max_num_points_per_voxel=self.model_cfg.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
            max_num_voxels=self.model_cfg.PRE_PROCESSING.MAX_NUMBER_OF_VOXELS['test'],
        )
        self.rdr_voxel_generator_train = VoxelGeneratorWrapper(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=num_point_features[1],
            max_num_points_per_voxel=self.model_cfg.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
            max_num_voxels=self.model_cfg.PRE_PROCESSING.MAX_NUMBER_OF_VOXELS['train'],
        )
        self.rdr_voxel_generator_test = VoxelGeneratorWrapper(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=num_point_features[1],
            max_num_points_per_voxel=self.model_cfg.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
            max_num_voxels=self.model_cfg.PRE_PROCESSING.MAX_NUMBER_OF_VOXELS['test'],
        )
        num_point_features_l = num_point_features[0]
        num_point_features_r = num_point_features[1]
        
        self.num_point_features_l = num_point_features_l + num_point_features_r - 3
        self.num_point_features_r = self.num_point_features_l
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        print("common feature dim (use preground_score) = ", self.num_point_features_r)

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def mme(self,batch_dict):
        lidar_voxel_features, lidar_voxel_num_points, lidar_coords = batch_dict['lidar_voxels'], batch_dict['lidar_voxel_num_points'], batch_dict['lidar_voxel_coords']
        radar_voxel_features, radar_voxel_num_points, radar_coords = batch_dict['radar_voxels'], batch_dict['radar_voxel_num_points'], batch_dict['radar_voxel_coords']
        L_coords = lidar_coords[:,:]
        R_coords = radar_coords[:,:]
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
            
            #radar特征部分修改
            com_features[:, i, now_feature_idx : now_feature_idx + radar_voxel_features.shape[-1] - 3] = radar_voxel_features[:, i, 3:]
        
        len_lidar = int(lidar_voxel_num_points.max())
        l_ex_features = torch.zeros((len(lidar_voxel_num_points), 32, self.num_point_features_l)).cuda()
        now_feature_idx = 0
        l_ex_features[:, :, now_feature_idx : now_feature_idx + lidar_voxel_features.shape[-1]] = lidar_voxel_features #4
        now_feature_idx += lidar_voxel_features.shape[-1]

        #计算lidar to radar共同部分中的cluster和feature均值用于特征传播
        #(N,3); (N,feature_dim-1)
        mask = self.get_paddings_indicator(radar_voxel_num_points[common_R], 32, axis=0)
        num_valid = mask.sum(dim=1)
        l2r_mask = (num_valid > 0)

        l2r_com_L = common_L[l2r_mask]
        l2r_com_R = common_R[l2r_mask]
        num_valid = num_valid[l2r_mask]
        mask = mask[l2r_mask].unsqueeze(-1)
        #计算lidar to radar共同部分的cluster(注意只有common(L，R)的部分才有)
        extraFea_R = (radar_voxel_features[l2r_com_R, :32, 3:] * mask).sum(dim=1, keepdim=True) / num_valid.type_as(radar_voxel_features).view(-1, 1, 1)
        l_ex_features[l2r_com_L, :, now_feature_idx :] = extraFea_R #4
        now_feature_idx += extraFea_R.shape[-1]

        lidar_features = l_ex_features
        radar_features = com_features
        batch_dict['lidar_voxels'] = lidar_features
        batch_dict['radar_voxels'] = radar_features
        return batch_dict

    def forward(self, batch_dict):
        if self.is_pre_processing is None:
            return batch_dict
        elif self.is_pre_processing == 'v1_0':
            # Shuffle (DataProcessor.shuffle_points)
            batched_rdr= batch_dict['rdr_sparse'].detach()
            batched_indices_rdr= batch_dict['batch_indices_rdr_sparse'].detach()
            list_points = []
            list_voxels = []
            list_voxel_coords = []
            list_voxel_num_points = []
            for batch_idx in range(batch_dict['batch_size']):
                temp_points = batched_rdr[torch.where(batched_indices_rdr == batch_idx)[0],:self.num_point_features[1]]
                
                if (self.shuffle_points) and (self.training):
                    shuffle_idx = np.random.permutation(temp_points.shape[0])
                    temp_points = temp_points[shuffle_idx,:]
                list_points.append(temp_points)
                
                if self.transform_points_to_voxels:
                    if self.training:
                        voxels, coordinates, num_points = self.rdr_voxel_generator_train.generate(temp_points.cpu().numpy())
                    else:
                        voxels, coordinates, num_points = self.rdr_voxel_generator_test.generate(temp_points.cpu().numpy())
                    voxel_batch_idx = np.full((coordinates.shape[0], 1), batch_idx, dtype=np.int64)
                    coordinates = np.concatenate((voxel_batch_idx, coordinates), axis=-1) # bzyx

                    list_voxels.append(voxels)
                    list_voxel_coords.append(coordinates)
                    list_voxel_num_points.append(num_points)
            
            batched_points = torch.cat(list_points, dim=0)
            batch_dict['radar_points'] = torch.cat((batched_indices_rdr.reshape(-1,1), batched_points), dim=1).cuda() # b, x, y, z, intensity
            batch_dict['radar_voxels'] = torch.from_numpy(np.concatenate(list_voxels, axis=0)).cuda()
            batch_dict['radar_voxel_coords'] = torch.from_numpy(np.concatenate(list_voxel_coords, axis=0)).cuda()
            batch_dict['radar_voxel_num_points'] = torch.from_numpy(np.concatenate(list_voxel_num_points, axis=0)).cuda()
            batch_dict['gt_boxes'] = batch_dict['gt_boxes'].cuda()
            batch_dict['points'] = batch_dict['radar_points']

            batched_ldr64 = batch_dict['ldr64']
            batched_indices_ldr64 = batch_dict['batch_indices_ldr64']
            list_points = []
            list_voxels = []
            list_voxel_coords = []
            list_voxel_num_points = []
            for batch_idx in range(batch_dict['batch_size']):
                temp_points = batched_ldr64[torch.where(batched_indices_ldr64 == batch_idx)[0],:self.num_point_features[0]]
                if (self.shuffle_points) and (self.training):
                    shuffle_idx = np.random.permutation(temp_points.shape[0])
                    temp_points = temp_points[shuffle_idx,:]
                list_points.append(temp_points)
                
                if self.transform_points_to_voxels:
                    if self.training:
                        voxels, coordinates, num_points = self.ldr_voxel_generator_train.generate(temp_points.cpu().numpy())
                    else:
                        voxels, coordinates, num_points = self.ldr_voxel_generator_test.generate(temp_points.cpu().numpy())
                    voxel_batch_idx = np.full((coordinates.shape[0], 1), batch_idx, dtype=np.int64)
                    coordinates = np.concatenate((voxel_batch_idx, coordinates), axis=-1) # bzyx

                    list_voxels.append(voxels)
                    list_voxel_coords.append(coordinates)
                    list_voxel_num_points.append(num_points)
            
            batched_points = torch.cat(list_points, dim=0)
            batch_dict['lidar_points'] = torch.cat((batched_indices_ldr64.reshape(-1,1), batched_points), dim=1).cuda() # b, x, y, z, intensity
            batch_dict['lidar_voxels'] = torch.from_numpy(np.concatenate(list_voxels, axis=0)).cuda()
            batch_dict['lidar_voxel_coords'] = torch.from_numpy(np.concatenate(list_voxel_coords, axis=0)).cuda()
            batch_dict['lidar_voxel_num_points'] = torch.from_numpy(np.concatenate(list_voxel_num_points, axis=0)).cuda()
            
            batch_dict = self.mme(batch_dict)
            return batch_dict