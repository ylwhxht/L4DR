import numpy as np
from ...utils import box_utils
import math
import numpy as np 
class kitti_config():
    # Car and Van ==> Car class
    # Pedestrian and Person_Sitting ==> Pedestrian Class
    #[0, -25.6, -3, 51.2, 25.6, 2]
    CLASS_NAME_TO_ID = {
        'Pedestrian': -99,
        'Car': 0,
        'Cyclist': -99,
        'Van': 0,
        'Truck': 0,
        'Person_sitting': -99,
        'Tram': -99,
        'Misc': -99,
        'DontCare': -1
    }

    colors = [[0, 0, 255], [0, 255, 255], [255, 0, 0], [255, 120, 0],
            [255, 120, 120], [0, 120, 0], [120, 255, 255], [120, 0, 255]]

    #####################################################################################
    boundary = {
        "minX": 0,
        "maxX": 51.2,
        "minY": -25.6,
        "maxY": 25.6,
        "minZ": -3,
        "maxZ": 2
    }

    bound_size_x = boundary['maxX'] - boundary['minX']
    bound_size_y = boundary['maxY'] - boundary['minY']
    bound_size_z = boundary['maxZ'] - boundary['minZ']

    boundary_back = {
        "minX": -50,
        "maxX": 0,
        "minY": -25,
        "maxY": 25,
        "minZ": -2.73,
        "maxZ": 1.27
    }
    # BEV_WIDTH = 432  # across y axis -25m ~ 25m
    # BEV_HEIGHT = 432  # across x axis 0m ~ 50m
    BEV_WIDTH = 480  # across y axis -25m ~ 25m
    BEV_HEIGHT = 480  # across x axis 0m ~ 50m
    # BEV_WIDTH = 304  # across y axis -25m ~ 25m
    # BEV_HEIGHT = 304  # across x axis 0m ~ 50m
    DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT

    # maximum number of points per voxel
    T = 35

    # voxel size
    vd = 0.1  # z
    vh = 0.05  # y
    vw = 0.05  # x

    # voxel grid
    W = math.ceil(bound_size_x / vw)
    H = math.ceil(bound_size_y / vh)
    D = math.ceil(bound_size_z / vd)

    # Following parameters are calculated as an average from KITTI dataset for simplicity
    #####################################################################################
    Tr_velo_to_cam = np.array([
        [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
        [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
        [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
        [0, 0, 0, 1]
    ])

    # cal mean from train set
    R0 = np.array([
        [0.99992475, 0.00975976, -0.00734152, 0],
        [-0.0097913, 0.99994262, -0.00430371, 0],
        [0.00729911, 0.0043753, 0.99996319, 0],
        [0, 0, 0, 1]
    ])

    P2 = np.array([[719.787081, 0., 608.463003, 44.9538775],
                [0., 719.787081, 174.545111, 0.1066855],
                [0., 0., 1., 3.0106472e-03],
                [0., 0., 0., 0]
                ])

    R0_inv = np.linalg.inv(R0)
    Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam)
    P2_inv = np.linalg.pinv(P2)
    #####################################################################################




def transform_annotations_to_kitti_format(annos, map_name_to_kitti=None, info_with_fakelidar=False):
    """
    Args:
        annos:
        map_name_to_kitti: dict, map name to KITTI names (Car, Pedestrian, Cyclist)
        info_with_fakelidar:
    Returns:

    """
    for anno in annos:
        # For lyft and nuscenes, different anno key in info
        if 'name' not in anno:
            anno['name'] = anno['gt_names']
            anno.pop('gt_names')

        for k in range(anno['name'].shape[0]):
            anno['name'][k] = map_name_to_kitti[anno['name'][k]]

        anno['bbox'] = np.zeros((len(anno['name']), 4))
        anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
        anno['truncated'] = np.zeros(len(anno['name']))
        anno['occluded'] = np.zeros(len(anno['name']))
        if 'boxes_lidar' in anno:
            gt_boxes_lidar = anno['boxes_lidar'].copy()
        else:
            gt_boxes_lidar = anno['gt_boxes_lidar'].copy()

        if len(gt_boxes_lidar) > 0:
            if info_with_fakelidar:
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

            gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
            anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
            anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
            anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
            anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
            dxdydz = gt_boxes_lidar[:, 3:6]
            anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
            anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
            anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
        else:
            anno['location'] = anno['dimensions'] = np.zeros((0, 3))
            anno['rotation_y'] = anno['alpha'] = np.zeros(0)

    return annos


def calib_to_matricies(calib):
    """
    Converts calibration object to transformation matricies
    Args:
        calib: calibration.Calibration, Calibration object
    Returns
        V2R: (4, 4), Lidar to rectified camera transformation matrix
        P2: (3, 4), Camera projection matrix
    """
    V2C = np.vstack((calib.V2C, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    R0 = np.hstack((calib.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
    R0 = np.vstack((R0, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    V2R = R0 @ V2C
    P2 = calib.P2
    return V2R, P2