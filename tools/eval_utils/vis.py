
def save_frame_vis(pred_dicts, batch_dict): 
    vis = {}
    vis['pred'] = []

    for p in pred_dicts:
        w = {}
        for k in p.keys():
            w[k] = p[k].detach().cpu().numpy()
        vis['pred'].append(w)
    if 'points' in batch_dict.keys():
        vis['points'] = batch_dict['points'].detach().cpu().numpy()
    if 'lidar_points' in batch_dict.keys():
        vis['lidar_points'] = batch_dict['lidar_points'].detach().cpu().numpy()
    if 'radar_points' in batch_dict.keys():
        vis['radar_points'] = batch_dict['radar_points'].detach().cpu().numpy()
    if 'raw_radar_points' in batch_dict.keys():
        vis['raw_radar_points'] = batch_dict['raw_radar_points'].detach().cpu().numpy()
    if 'point_cls_scores' in batch_dict.keys():
        vis['point_cls_scores'] = batch_dict['point_cls_scores'].detach().cpu().numpy()
    if 'point_cls_labels' in batch_dict.keys():
        vis['point_cls_labels'] = batch_dict['point_cls_labels'].detach().cpu().numpy()
    if 'bfgt' in batch_dict.keys():
        vis['bfgt'] = batch_dict['bfgt'].detach().cpu().numpy()
    vis['gt'] = batch_dict['gt_boxes'].detach().cpu().numpy()
    return vis