from .detector3d_template import Detector3DTemplate


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

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
            print(loss_point)
            loss = loss_rpn + loss_point
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }
        
        
        return loss, tb_dict, disp_dict
