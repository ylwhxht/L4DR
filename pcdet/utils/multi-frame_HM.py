import os 
import  object3d_kitti, calibration_kitti
import numpy as np
import pickle
labelpath  = '/mnt/32THHD/view_of_delft_PUBLIC/lidar/training/label_2/'
calibpath  = '/mnt/ssd8T/AdverseWeather/view_of_delft_PUBLIC/lidar/training/calib/'
CLASS_NAME_TO_ID = {
        'Pedestrian': 1,
        'Car': 0,
        'Cyclist': 2,
}
files= os.listdir(labelpath)
def get_gt_box(obj_list, calib):
    annotations = {}
    annotations['name'] = np.array([obj.cls_type for obj in obj_list])
    annotations['truncated'] = -np.ones(len(obj_list))
    annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
    annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
    annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
    annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
    annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
    annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
    annotations['score'] = np.array([obj.score for obj in obj_list])
    annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

    num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
    num_gt = len(annotations['name'])
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)

    loc = annotations['location'][:num_objects]
    dims = annotations['dimensions'][:num_objects]
    rots = annotations['rotation_y'][:num_objects]
    name = annotations['name'][:num_objects]
    name_ID = np.array([CLASS_NAME_TO_ID[n] if n in CLASS_NAME_TO_ID else -99 for n in name])
    loc_lidar = calib.rect_to_lidar(loc)
    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc_lidar[:, 2] += h[:, 0] / 2

    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis]), name_ID[..., np.newaxis]], axis=1)
    return gt_boxes_lidar

GTs = {
    
}
mfGTs = {
}
files.sort()
for i, file in enumerate(files):
    idx = int(file[:-4])
    print("processing label ->", idx, " : ", end= " ")
    label_file = os.path.join(labelpath, file)
    calib_file = os.path.join(calibpath, file)
    object3d = object3d_kitti.get_objects_from_label(label_file)
    calib3d = calibration_kitti.Calibration(calib_file)
    GTs[idx] = get_gt_box(object3d, calib3d)
    tmp_GT = []
    for j in range(1,min(idx+1,5)):
        if idx-j in GTs.keys():
            print((np.ones(len(GTs[idx-j]))*-j).reshape(-1,1))
            tmp_GT.append(np.concatenate([GTs[idx-j],(np.ones(len(GTs[idx-j]))*-j).reshape(-1,1)], axis=1))
    print()
    if len(tmp_GT)>0:
        mfGTs[idx] = np.concatenate(tmp_GT)
        print(mfGTs[idx].shape)
    else:
        mfGTs[idx] = np.array([])
with open('/mnt/ssd8T/rlfusion_5f/vod_infos_-4fGT.pkl', 'wb') as f:
    pickle.dump(mfGTs, f)
print('vis save in /mnt/ssd8T/rlfusion_5f/vod_infos_-4fGT.pkl')
print("\nfinished")