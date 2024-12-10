import numpy as np
from pcdet.utils import calibration_kitti, object3d_kitti
from pcdet.datasets.kitti.viewer.viewer import Viewer
def get_pc():
    lidar_file = "/mnt/8tssd/AdverseWeather/view_of_delft_PUBLIC/rlfusion_5f/gt_database/lidar_01061_Car_10.bin"
    l_points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    
    radar_file = "/mnt/8tssd/AdverseWeather/view_of_delft_PUBLIC/rlfusion_5f/gt_database/radar_01061_Car_10.bin"
    r_points = np.fromfile(str(radar_file), dtype=np.float32).reshape(-1, 7)
    # means = [0, 0, 0, -13.0, -3.0, -0.1, 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
    # stds =  [1, 1, 1, 14.0,  8.0,  6.0, 1]
    # r_points = (r_points - means)/stds
    return l_points, r_points
def get_calib():
    l_calib_file = '/mnt/8tssd/AdverseWeather/view_of_delft_PUBLIC/lidar/training/calib/00000.txt'
    r_calib_file = '/mnt/8tssd/AdverseWeather/view_of_delft_PUBLIC/radar_5frames/training/calib/00000.txt'
    return calibration_kitti.Calibration(l_calib_file), calibration_kitti.Calibration(r_calib_file)
def get_label(calib):
    label_file = '/mnt/8tssd/AdverseWeather/view_of_delft_PUBLIC/lidar/training/label_2/00000.txt'
    annotations = {}
    obj_list = object3d_kitti.get_objects_from_label(label_file)
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
    loc = annotations['location']
    dims = annotations['dimensions']
    rots = annotations['rotation_y']
    loc_lidar = calib.rect_to_lidar(loc)
    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc_lidar[:, 2] += h[:, 0] / 2
    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
    annotations['gt_boxes_lidar'] = gt_boxes_lidar
    return gt_boxes_lidar

lidar_points, radar_points = get_pc()
# l_calib, r_calib = get_calib()

# radar_points = l_calib.rect_to_lidar(r_calib.lidar_to_rect(radar_points[:, 0:3]))
# gt_boxes = get_label(l_calib)
vi = Viewer()
print(lidar_points[:10, 0:3], radar_points[:10, 0:3])
vi.add_points(lidar_points[:, 0:3])
vi.add_points(radar_points[:, 0:3], radius = 3, color='red' )
# vi.add_3D_boxes(gt_boxes, color='red')
vi.show_3D()