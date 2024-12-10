from .detector3d_template import Detector3DTemplate
from ...datasets.processor.data_processor import VoxelGeneratorWrapper
import numpy as np
import torch
import queue
from ...utils import common_utils, commu_utils
class PointPillarMask(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.TP = common_utils.AverageMeter()
        self.P = common_utils.AverageMeter()
        self.TP_FN = common_utils.AverageMeter()
        self.NOR = common_utils.AverageMeter()
        self.TP_FP_FN = common_utils.AverageMeter()
        self.All = common_utils.AverageMeter()

    def forward(self, batch_dict):
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.point_head(batch_dict)
        pre_mask = batch_dict['point_cls_scores'] > 0.2
        labels = batch_dict['point_cls_labels'] > 0
        self.TP.update((pre_mask&labels).sum().item())
        self.P.update(pre_mask.sum().item())
        self.TP_FP_FN.update((pre_mask|labels).sum().item())
        self.NOR.update((~(pre_mask^labels)).sum().item())
        self.All.update(pre_mask.shape[0])
        self.TP_FN.update(labels.sum().item())
        # if self.TP.avg>0:
        #     print("recall = ",round(self.TP.sum/self.TP_FN.sum,4),end=" ; ")
        #     print("precise = ",round(self.TP.sum/self.P.sum,4),end=" ; ")
        #     print("mIoU = ",round(self.TP.sum/self.TP_FP_FN.sum,4),end=" ; ")
        #     print("PA = ",round(self.NOR.sum/self.All.sum,4))
        batch_dict['raw_radar_points'] = batch_dict['radar_points']
        batch_dict['radar_points'] = torch.cat([batch_dict['radar_points'][pre_mask],batch_dict['point_cls_scores'][pre_mask].reshape(-1,1)], dim=1)
        batch_dict = self.transform_points_to_voxels(batch_dict, batch_dict['radar_points'])
        batch_dict = self.vfe(batch_dict)
        
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)
        
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            if 'cen_loss' in batch_dict:
                cen_loss = batch_dict['cen_loss']
                loss = loss + cen_loss
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            pred_dicts[0]['batch_dict'] = batch_dict
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        
        loss = loss_rpn
        if self.model_cfg.get('POINT_HEAD', None) is not None:
            loss_point, tb_dict = self.point_head.get_loss()
            loss = loss_rpn + loss_point
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }
        
        
        return loss, tb_dict, disp_dict
    def transform_points_to_voxels(self, batch_dict, bs_radar_points):
        MAX_NUMBER_OF_VOXELS =  {
            'train': 16000,
            'test': 40000
        }
        self.voxel_generator_r = VoxelGeneratorWrapper(
            vsize_xyz=[0.16, 0.16, 4],
            coors_range_xyz=self.dataset.data_processor.point_cloud_range,
            num_point_features=self.dataset.data_processor.num_point_features[1] + 1,
            max_num_points_per_voxel=32,
            max_num_voxels=MAX_NUMBER_OF_VOXELS[self.dataset.data_processor.mode],
        )
        bs_radar_points = bs_radar_points.detach().cpu().numpy()
        batch_size = batch_dict['lidar_voxel_coords'][:, 0].max().int().item() + 1
        radar_voxels_list = []
        radar_coordinates_list = []
        radar_num_points_list = []
        for bs in range(batch_size):
            bs_mask = bs_radar_points[:,0] == bs
            radar_points = bs_radar_points[bs_mask,1:]
            
            # Generate output voxel_output for different modalities in sequence.
            radar_voxel_output = self.voxel_generator_r.generate(radar_points)
            radar_voxels, radar_coordinates, radar_num_points = radar_voxel_output
            if not batch_dict['use_lead_xyz'][0]:
                radar_voxels = radar_voxels[..., 3:]  # remove xyz in voxels(N, 3)c
            radar_voxels_list.append(radar_voxels)
            radar_coordinates_list.append(radar_coordinates)
            radar_num_points_list.append(radar_num_points)
        coors = []
        
        for i, coor in enumerate(radar_coordinates_list):
            coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
            coors.append(coor_pad)
        batch_dict['radar_voxel_coords'] = torch.tensor(np.concatenate(coors, axis=0)).cuda()
        batch_dict['radar_voxels'] = torch.tensor(np.concatenate(radar_voxels_list, axis=0)).cuda()
        batch_dict['radar_voxel_num_points'] = torch.tensor(np.concatenate(radar_num_points_list, axis=0)).cuda()
        return batch_dict
