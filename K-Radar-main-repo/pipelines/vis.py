
def save_frame_vis(pred_dicts, batch_dict): 
    vis = {}
    vis['pred'] = []

    w = {}
    for k in pred_dicts.keys():
        w[k] = pred_dicts[k].detach().cpu().numpy()
    vis['pred'].append(w)
    if 'lidar_points' in batch_dict.keys():
        vis['lidar_points'] = batch_dict['lidar_points'].detach().cpu().numpy()
        print('lidar_points saved')
    if 'radar_points' in batch_dict.keys():
        vis['radar_points'] = batch_dict['radar_points'].detach().cpu().numpy()
        print('radar_points saved')
    if 'raw_radar_points' in batch_dict.keys():
        vis['raw_radar_points'] = batch_dict['raw_radar_points'].detach().cpu().numpy()
        print('raw_radar_points saved')
    if 'point_cls_scores' in batch_dict.keys():
        vis['point_cls_scores'] = batch_dict['point_cls_scores'].detach().cpu().numpy()
        print('point_cls_scores saved')
    if 'point_cls_labels' in batch_dict.keys():
        vis['point_cls_labels'] = batch_dict['point_cls_labels'].detach().cpu().numpy()
        print('point_cls_labels saved')
    if 'bfgt' in batch_dict.keys():
        vis['bfgt'] = batch_dict['bfgt'].detach().cpu().numpy()
    vis['gt'] = batch_dict['gt_boxes'].detach().cpu().numpy()
    print('gt_boxes saved')
    return vis