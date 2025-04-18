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

class LidarSparseProcessor(nn.Module):
    def __init__(self, cfg):
        super(LidarSparseProcessor, self).__init__()
        self.cfg = cfg
        self.model_cfg = cfg.MODEL
        self.dataset_cfg = cfg.DATASET

        # class
        self.num_class = 0
        self.class_names = []

        self.is_pre_processing = self.model_cfg.PRE_PROCESSING.get('VER', None)
        self.shuffle_points = self.model_cfg.PRE_PROCESSING.get('SHUFFLE_POINTS', False)
        self.transform_points_to_voxels = self.model_cfg.PRE_PROCESSING.get('TRANSFORM_POINTS_TO_VOXELS', False)
        
        self.is_logging = cfg.GENERAL.LOGGING.IS_LOGGING
        
        num_point_features = [self.dataset_cfg.ldr64.n_used,self.dataset_cfg.rdr_sparse.n_used]
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
            return batch_dict