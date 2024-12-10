import pickle
import numpy as np
from pcdet.utils import calibration_kitti
import copy
import math
from pathlib import Path
path = "/mnt/8tssd/kitti_det3d/detection/kitti_infos_trainval.pkl"
with open(path, 'rb') as f:
    infos = pickle.load(f)

def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info

def boxes3d_kitti_camera_to_lidar(boxes3d_camera, calib):
    """
    Args:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
        calib:

    Returns:
        boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    """
    boxes3d_camera_copy = copy.deepcopy(boxes3d_camera)
    xyz_camera, r = boxes3d_camera_copy[:, 0:3], boxes3d_camera_copy[:, 6:7]
    l, h, w = boxes3d_camera_copy[:, 3:4], boxes3d_camera_copy[:, 4:5], boxes3d_camera_copy[:, 5:6]

    xyz_lidar = calib.rect_to_lidar(xyz_camera)
    xyz_lidar[:, 2] += h[:, 0] / 2
    return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)

results = {
    'Car':{
        'Easy':{
            '0m-15m':0,
            '15m-30m':0,
            '30m-45m':0,
            '45m-inf':0,
        },
        'Moderate':{
            '0m-15m':0,
            '15m-30m':0,
            '30m-45m':0,
            '45m-inf':0,
        },
        'Hard':{
            '0m-15m':0,
            '15m-30m':0,
            '30m-45m':0,
            '45m-inf':0,
        },
    },
    'Pedestrian':{
        'Easy':{
            '0m-15m':0,
            '15m-30m':0,
            '30m-45m':0,
            '45m-inf':0,
        },
        'Moderate':{
            '0m-15m':0,
            '15m-30m':0,
            '30m-45m':0,
            '45m-inf':0,
        },
        'Hard':{
            '0m-15m':0,
            '15m-30m':0,
            '30m-45m':0,
            '45m-inf':0,
        },
    },
    'Cyclist':{
        'Easy':{
            '0m-15m':0,
            '15m-30m':0,
            '30m-45m':0,
            '45m-inf':0,
        },
        'Moderate':{
            '0m-15m':0,
            '15m-30m':0,
            '30m-45m':0,
            '45m-inf':0,
        },
        'Hard':{
            '0m-15m':0,
            '15m-30m':0,
            '30m-45m':0,
            '45m-inf':0,
        },
    }
}
DIFFICULTY  = ['Easy', 'Moderate', 'Hard']
DISTANCE  = ['0m-15m', '15m-30m', '30m-45m', '45m-inf']

def distance_to_origin(box):
    distance = box[0]
    if distance <= 15:
        return 0
    elif distance <= 30: 
        return 1
    elif distance <= 45:
        return 2
    else:
        return 3
    
for info in infos:
    calib = Path('/mnt/8tssd/kitti_det3d/detection/training/calib/') / ('%06d.txt' % (info['image']['image_idx']))
    calib = calibration_kitti.Calibration(calib)
    annos = info['annos']
    loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
    gt_names = annos['name']
    gt_dif = annos['difficulty']
    gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
    gt_boxes_lidar = boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
    for idx in range(gt_boxes_lidar.shape[0]):
        if gt_names[idx] not in results.keys():
            continue
        dif = gt_dif[idx]
        dis = distance_to_origin(gt_boxes_lidar[idx])
        results[gt_names[idx]][DIFFICULTY[dif]][DISTANCE[dis]] += 1
for t1 in results.keys():
    print("Class:",t1)
    for t2 in results[t1].keys():
        print("    Difficulty:",t2)
        for t3 in results[t1][t2].keys():
            print("        Distance:",t3," = ",results[t1][t2][t3])
