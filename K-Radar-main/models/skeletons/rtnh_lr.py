
import os
import torch
import torch.nn as nn
import numpy as np
import heapq


from ops.iou3d_nms import iou3d_nms_utils
from utils.spconv_utils import find_all_spconv_keys
from models import pre_processor, backbone_2d, backbone_3d, head, roi_head
from models.backbone_2d import map_to_bev
from models.backbone_3d import pfe, vfe
from models.model_utils import model_nms_utils
from .utils import common_utils
tv = None
try:
    import cumm.tensorview as tv
except:
    pass

class RTNH_LR(nn.Module):
    def __init__(self, cfg):
        
        super().__init__()
        self.cfg = cfg
        self.num_class = 0
        self.class_names = []
        dict_label = self.cfg.DATASET.label.copy()
        list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
        for temp_key in list_for_pop:
            dict_label.pop(temp_key)
        self.dict_cls_name_to_id = dict()
        for k, v in dict_label.items():
            _, logit_idx, _, _ = v
            self.dict_cls_name_to_id[k] = logit_idx
            self.dict_cls_name_to_id['Background'] = 0
            if logit_idx > 0:
                self.num_class += 1
                self.class_names.append(k)
        self.cfg_model = cfg.MODEL
        
        self.list_module_names = [
            'pre_processor','pfe','pointhead', 'mme', 'vfe', 'backbone', 'head', 'roi_head'
        ]
        self.list_modules = []
        self.build_radar_detector()

    def build_radar_detector(self):
        for name_module in self.list_module_names:
            module = getattr(self, f'build_{name_module}')()
            
            if module is not None:
                self.add_module(name_module, module) # override nn.Module
                self.list_modules.append(module)

    def build_pre_processor(self):
        if self.cfg_model.get('PRE_PROCESSOR', None) is None:
            return None
        
        module = pre_processor.__all__[self.cfg_model.PRE_PROCESSOR.NAME](self.cfg)
        return module

    def build_mme(self):
        if self.cfg_model.get('MME', None) is None:
            return None
        
        module = pre_processor.__all__[self.cfg_model.MME.NAME](self.cfg)
        return module

    def build_vfe(self):
        if self.cfg_model.get('PRE_PROCESSOR', None) is None:
            return None
        
        module = backbone_3d.vfe.__all__[self.cfg_model.PRE_PROCESSOR.VFE](self.cfg)
        return module

    def build_pfe(self):
        if self.cfg_model.get('PFE', None) is None:
            return None
        
        module = backbone_3d.__all__[self.cfg_model.PFE.NAME](
            model_cfg=self.cfg_model.PFE,
            input_channels=self.cfg_model.PRE_PROCESSOR.INPUT_DIM,
        )
        return module
        
    def build_pointhead(self):
        if self.cfg_model.get('POINT_HEAD', None) is None:
            return None
        module = head.__all__[self.cfg_model.POINT_HEAD.NAME](
            model_cfg=self.cfg_model.POINT_HEAD,
            input_channels= self.cfg_model.POINT_HEAD.DIM,
            num_class=self.num_class if not self.cfg_model.POINT_HEAD.CLASS_AGNOSTIC else 1,
            predict_boxes_when_training=self.cfg_model.get('ROI_HEAD', False)
        )
        return module

    def build_backbone(self):
        cfg_backbone = self.cfg_model.get('BACKBONE', None)
        if cfg_backbone is None:
            return None
        
        if cfg_backbone.TYPE == '2D':
            return backbone_2d.__all__[cfg_backbone.NAME](self.cfg)
        elif cfg_backbone.TYPE == '3D':
            return backbone_3d.__all__[cfg_backbone.NAME](self.cfg)
        else:
            return None

    def build_head(self):
        if (self.cfg.MODEL.get('HEAD', None)) is None:
            return None
        module = head.__all__[self.cfg_model.HEAD.NAME](self.cfg)
        return module

    def build_roi_head(self):
        if (self.cfg.MODEL.get('ROI_HEAD', None)) is None:
            return None
        head_module = roi_head.__all__[self.cfg_model.ROI_HEAD.NAME](self.cfg)
        return head_module
    


    def mask_radar(self, batch_dict):
        pre_mask = batch_dict['point_cls_scores'] > self.cfg_model.PRE_PROCESSING.DENOISE_T
        #to keep inference when have few fore_radar_points
        if pre_mask.sum() < 200:
            pre_mask[:200] = 1
        extra_choice = torch.ones(batch_dict['point_cls_scores'][pre_mask].shape,dtype = bool)
        MAX_FORE_RADAR_NUM = 30000
        if pre_mask.sum() > MAX_FORE_RADAR_NUM:
            extra_choice[:] = 0
            arr_score = batch_dict['point_cls_scores'][pre_mask]
            topk_indices = torch.topk(arr_score, MAX_FORE_RADAR_NUM)[1]
            extra_choice[topk_indices] = 1
        batch_dict['raw_rdr_sparse'] = batch_dict['rdr_sparse'][pre_mask]

        batch_dict['batch_indices_rdr_sparse'] = batch_dict['batch_indices_rdr_sparse'][pre_mask][extra_choice]
        try:
            batch_dict['rdr_sparse'] = torch.cat([batch_dict['rdr_sparse'][pre_mask][extra_choice],batch_dict['point_cls_scores'][pre_mask][extra_choice].reshape(-1,1)], dim=1)
        except:
            batch_dict['rdr_sparse'] = torch.cat([batch_dict['rdr_sparse'][pre_mask][extra_choice],batch_dict['point_cls_scores'][pre_mask][extra_choice].reshape(-1,1).detach().cpu()], dim=1)
        print(batch_dict['rdr_sparse'].shape)
        return batch_dict

    def forward(self, x):
        module_idx = 0
        for module in self.list_modules:
            x = module(x)
            if self.list_module_names[module_idx] == 'pointhead':
                x = self.mask_radar(x)
            module_idx += 1
        return x

    def loss(self, dict_item):
        loss_rpn, _ = self.list_modules[2].get_loss()
        loss_point = self.list_modules[6].loss(dict_item)
        loss = loss_rpn + loss_point 

        return loss