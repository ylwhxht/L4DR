import copy
import pickle
import math
import numpy as np
from skimage import io
import  torch
from pathlib import Path
import warnings
import random
warnings.filterwarnings("ignore")
from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate
from ..vod_evaluation.kitti_official_evaluate import get_official_eval_result
from .kitti_utils import kitti_config
class VodDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        self.vod_eva =  self.dataset_cfg.get('VOD_EVA', False)
        self.sim_info_path = None
        self.MMF = False
        self.useallgt = False
        self.filter_empty = False
        self.use_fog = 2
        # train = 2
        self.fog_I = 2
        self.train = (self.split != 'val')
        # 0 no fog 
        # 1 specific fog 
        # 2 random fog

        
        self.sim_info_path_list = [
            Path('/mnt/32THHD/view_of_delft_PUBLIC/fog_sim_lidar/_CVL_beta_0.005/'),
            Path('/mnt/32THHD/view_of_delft_PUBLIC/fog_sim_lidar/_CVL_beta_0.010/'),
            Path('/mnt/32THHD/view_of_delft_PUBLIC/fog_sim_lidar/_CVL_beta_0.020/'),
            Path('/mnt/32THHD/view_of_delft_PUBLIC/fog_sim_lidar/_CVL_beta_0.030/'),
        ]
        self.sim_info_path = Path('/mnt/32THHD/view_of_delft_PUBLIC/fog_sim_lidar/_CVL_beta_0.030/') 
        #w/o fog : use_fog = 0 / use_fog = 2
        #fog = 1 : use_fog = 1 & path = 030
        #fog = 2 : use_fog = 1 & path = 060
        #fog = 3 : use_fog = 1 & path = 100
        #fog = 4 : use_fog = 1 & path = 200
        self.sensor =  self.dataset_cfg.get('SENSOR', 'LiDAR')
        
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        self.max_objects = 50
        cnf = kitti_config
        self.hm_size = (cnf.BEV_WIDTH/2, cnf.BEV_HEIGHT/2)
        self.num_classes = 3
        self.vod_infos = []
        
        self.include_vod_data(self.mode)

    def include_vod_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading VoD dataset')
        vod_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            t_info_path = info_path
            info_path = self.root_path / info_path
            if self.sim_info_path is not None and self.use_fog==1:
                info_path = self.sim_info_path / t_info_path
                
            if not info_path.exists():
                continue
        
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                
                vod_infos.extend(infos)
        
        self.vod_infos.extend(vod_infos)
        if self.useallgt:
            with open('/mnt/ssd8T/rlfusion_5f/vod_infos_-4fGT.pkl', 'rb') as f:
                self.bfgt = pickle.load(f)
        if self.logger is not None:
            self.logger.info('Total samples for VoD dataset: %d' % (len(vod_infos)))

        if self.filter_empty:
            total = self.filter_empty_box()
            if self.logger is not None:
                self.logger.info('Total filter samples for VoD dataset: %d' % total)

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_pc(self, idx):
        lidar_file = self.root_split_path / 'lidar' / ('%s.bin' % idx)
        
        if self.sensor == 'LiDAR':
            assert lidar_file.exists()
            number_of_channels = 4  # ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']
            points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, number_of_channels)
        elif self.sensor == 'Radar':
            assert lidar_file.exists()
            number_of_channels = 7  # ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']
            points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, number_of_channels)
            # replace the list values with statistical values; for x, y, z and time, use 0 and 1 as means and std to avoid normalization
            means = [0, 0, 0, 0, 0, 0, 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            stds =  [1, 1, 1, 1, 1, 1, 1]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            # #in practice, you should use either train, or train+val values to calculate mean and stds. Note that x, y, z, and time are not normed, but you can experiment with that.
            means = [0, 0, 0, -13.0, -3.0, -0.1, 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            stds =  [1, 1, 1, 14.0,  8.0,  6.0, 1]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            
            #we then norm the channels
            points = (points - means)/stds
            
        elif self.sensor == 'Fusion':
            lidar_file = self.root_split_path / 'lidar' / ('%s.bin' % idx)
            if self.sim_info_path is not None and self.use_fog == 1:
                lidar_file = self.sim_info_path / ('%s.bin' % idx)
            if self.sim_info_path_list is not None and self.use_fog == 2 and self.train:
                aug = random.randint(0, 7)
                if aug < len(self.sim_info_path_list):
                    self.fog_I = aug + 1
                    lidar_file = self.sim_info_path_list[aug] / ('%s.bin' % idx)
                else:
                    self.fog_I = 0
            l_points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
            
            radar_file = self.root_split_path / 'radar_5f' / ('%s.bin' % idx)
            assert radar_file.exists()
            r_points = np.fromfile(str(radar_file), dtype=np.float32).reshape(-1, 7)
            means = [0, 0, 0, -13.0, -3.0, -0.1, 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            stds =  [1, 1, 1, 14.0,  8.0,  6.0, 1]
            r_points = (r_points - means) / stds
            return l_points, r_points
        return points

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.jpg' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)
    
    def get_catid(self, name):
        type_to_id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        if name not in type_to_id.keys():
            return -99
        return type_to_id[name]
    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx: str, Sample index
        Returns:
            depth: (H, W), Depth map
        """
        depth_file = self.root_split_path / 'depth_2' / ('%s.png' % idx)
        assert depth_file.exists()
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        return depth

    def get_calib(self, idx, getall = False):
        if self.sensor != 'Fusion' or not getall:
            calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
            return calibration_kitti.Calibration(calib_file)
        else:
            l_calib_file = self.root_split_path / 'lidar_calib' / ('%s.txt' % idx)
            assert l_calib_file.exists()
            r_calib_file = self.root_split_path / 'radar_calib' / ('%s.txt' % idx)
            assert r_calib_file.exists()
            return calibration_kitti.Calibration(l_calib_file), calibration_kitti.Calibration(r_calib_file)
            
       

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
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
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    
                    if self.sensor == 'LiDAR' or self.sensor == 'Radar':
                        points = self.get_pc(sample_idx)
                        calib = self.get_calib(sample_idx)
                        pts_rect = calib.lidar_to_rect(points[:, 0:3])

                        fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                        pts_fov = points[fov_flag]
                        corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                        num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                        for k in range(num_objects):
                            flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                            num_points_in_gt[k] = flag.sum()
                        annotations['num_points_in_gt'] = num_points_in_gt
                    else:
                        lidar_points, _ = self.get_pc(sample_idx)
                        l_calib = self.get_calib(sample_idx)                   
                        l_pts_rect = l_calib.lidar_to_rect(lidar_points[:, 0:3])
                        l_fov_flag = self.get_fov_flag(l_pts_rect, info['image']['image_shape'], l_calib)
                        l_pts_fov = lidar_points[l_fov_flag]
                        corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                        l_num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                        
                        for k in range(num_objects):
                            l_flag = box_utils.in_hull(l_pts_fov[:, 0:3], corners_lidar[k])
                            l_num_points_in_gt[k] = l_flag.sum()


                        annotations['num_points_in_gt'] = l_num_points_in_gt
                        

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('vod_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            if self.sensor == 'LiDAR' or self.sensor == 'Radar':
                points = self.get_pc(sample_idx)
            else:
                lidar_points, radar_points = self.get_pc(sample_idx)
                l_calib, r_calib = self.get_calib(sample_idx, True)  
                radar_points[:,:3] = l_calib.rect_to_lidar(r_calib.lidar_to_rect(radar_points[:, :3]))
                
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            if self.sensor == 'LiDAR' or self.sensor == 'Radar':
                point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
                ).numpy()  # (nboxes, npoints)
                for i in range(num_obj):
                    filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                    filepath = database_save_path / filename
                    gt_points = points[point_indices[i] > 0]

                    gt_points[:, :3] -= gt_boxes[i, :3]
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    if (used_classes is None) or names[i] in used_classes:
                        db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                        db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                                'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                                'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                        if names[i] in all_db_infos:
                            all_db_infos[names[i]].append(db_info)
                        else:
                            all_db_infos[names[i]] = [db_info]
            else:
                lidar_point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(lidar_points[:, 0:3]), torch.from_numpy(gt_boxes)
                ).numpy()  # (nboxes, npoints)
                radar_point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(radar_points[:, 0:3]), torch.from_numpy(gt_boxes)
                ).numpy()  # (nboxes, npoints)

                for i in range(num_obj):
                    # filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                    lidar_filename = '%s_%s_%s_%d.bin' % ('lidar', sample_idx, names[i], i)
                    radar_filename = '%s_%s_%s_%d.bin' % ('radar', sample_idx, names[i], i)
                    lidar_filepath = database_save_path / lidar_filename
                    radar_filepath = database_save_path / radar_filename
                    lidar_gt_points = lidar_points[lidar_point_indices[i] > 0]
                    radar_gt_points = radar_points[radar_point_indices[i] > 0]
                    # gt_points = np.concatenate((lidar_gt_points, radar_gt_points), axis=0)
                    
                    # gt_points[:, :3] -= gt_boxes[i, :3]
                    lidar_gt_points[:, :3] -= gt_boxes[i, :3]
                    radar_gt_points[:, :3] -= gt_boxes[i, :3]
                    # with open(filepath, 'w') as f:
                    #     gt_points.tofile(f)
                    radar_gt_points = radar_gt_points.astype(np.float32)
                    with open(lidar_filepath, 'w') as f:
                         lidar_gt_points.tofile(f)
                    with open(radar_filepath, 'w') as f:
                         radar_gt_points.tofile(f)

                    if (used_classes is None) or names[i] in used_classes:
                        # db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                        lidar_db_path = str(lidar_filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                        radar_db_path = str(radar_filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                        db_info = {'name': names[i], 'lidar_path': lidar_db_path, 'radar_path': radar_db_path, 'image_idx': sample_idx, 'gt_idx': i,
                                'box3d_lidar': gt_boxes[i], 'lidar_num_points_in_gt': lidar_gt_points.shape[0], 'radar_num_points_in_gt': radar_gt_points.shape[0],
                                'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                        if names[i] in all_db_infos:
                            all_db_infos[names[i]].append(db_info)
                        else:
                            all_db_infos[names[i]] = [db_info]
                
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)
        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.vod_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.vod_infos]

        if not self.vod_eva :
            results, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        else:
            results = {}
            ap_dict = {}
            results.update(get_official_eval_result(eval_gt_annos, eval_det_annos, class_names))
            results.update(get_official_eval_result(eval_gt_annos, eval_det_annos, class_names, custom_method=3))
        return results, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.vod_infos) * self.total_epochs

        return len(self.vod_infos)
    def filter_empty_box(self):
        cnt = 0
        sum  = 0
        for i in range(len(self.vod_infos)):
            annotations = self.vod_infos[i]['annos']
            cnt += (annotations['num_points_in_gt'] == 0).sum()
            sum += annotations['num_points_in_gt'].sum()
            mask  = annotations['num_points_in_gt'] > 0
            for k in annotations.keys():
                try:
                    annotations[k] = annotations[k][mask]
                except:
                    print(k)
            self.vod_infos[i]['annos'] = annotations
        return cnt
    
    def ez_filter_radar(self, radar):
        radar = radar[(radar[:,2]>-3)&(radar[:,2]<0.5)]
        return radar

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.vod_infos)

        info = copy.deepcopy(self.vod_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }
        if self.MMF:
            input_dict.update({
                    'hm_cen': 0,
                })
        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
            ###
            ###
            input_dict.update({
                    'gt_names': gt_names,
                    'gt_boxes': gt_boxes_lidar
                })

            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane
        
        if "points" in get_item_list:
            if self.sensor == 'LiDAR' or self.sensor == 'Radar':
                points = self.get_pc(sample_idx)
                if self.dataset_cfg.FOV_POINTS_ONLY:
                    if self.sensor == 'LiDAR' or self.sensor == 'Radar':
                        pts_rect = calib.lidar_to_rect(points[:, 0:3])
                        fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                        points = points[fov_flag]
                input_dict['points'] = points
            else:
                lidar_points, radar_points = self.get_pc(sample_idx)
                l_calib, r_calib = self.get_calib(sample_idx, True)  
                calib = self.get_calib(sample_idx)  
                radar_points[:,:3] = l_calib.rect_to_lidar(r_calib.lidar_to_rect(radar_points[:, 0:3]))
                if self.dataset_cfg.FOV_POINTS_ONLY:
                    lidar_pts_rect = calib.lidar_to_rect(lidar_points[:, 0:3])
                    lidar_fov_flag = self.get_fov_flag(lidar_pts_rect, img_shape, calib)
                    lidar_points = lidar_points[lidar_fov_flag]
                    radar_pts_rect = calib.lidar_to_rect(radar_points[:, 0:3])
                    radar_fov_flag = self.get_fov_flag(radar_pts_rect, img_shape, calib)
                    radar_points = radar_points[radar_fov_flag]
                # if len(gt_boxes_lidar):
                #     radar_point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                #         torch.from_numpy(radar_points[:, 0:3]), torch.from_numpy(gt_boxes_lidar)
                #     ).numpy()
                #     radar_point_indices = np.max(radar_point_indices, axis=0).reshape(-1)
                #     print(radar_points.shape, radar_point_indices.shape)
                #     if (radar_point_indices > 0).sum()< 5 :
                #         radar_point_indices[:5] = 1
                #     radar_points = radar_points[radar_point_indices > 0]
                #     print(radar_points.shape)
                input_dict['lidar_points'] = lidar_points
                input_dict['radar_points'] = radar_points

            

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        input_dict['calib'] = calib
        if self.useallgt:
            input_dict['bfgt'] = self.bfgt[int(sample_idx)].reshape(-1,9)
        data_dict = self.prepare_data(data_dict=input_dict)
        if self.useallgt:
            data_dict['bfgt'] = np.concatenate([np.concatenate([data_dict['gt_boxes'],np.zeros(len(data_dict['gt_boxes'])).reshape(-1,1)],axis=1), data_dict['bfgt']])

        if self.MMF:
            hflipped = False
            catid = []
            gt_boxes = data_dict['gt_boxes']
            mf_gt = np.concatenate([gt_boxes, data_dict['bfgt']])
            targets = self.build_targets(mf_gt, hflipped)
            data_dict.update({
                'hm_cen': targets['hm_cen'],
                'cen_offset': targets['cen_offset'],
                'direction': targets['direction'],
                'z_coor': targets['z_coor'],
                'dim': targets['dim'],
                'indices_center': targets['indices_center'],
                'obj_mask': targets['obj_mask']
            })
            data_dict.pop('bfgt')
        data_dict['image_shape'] = img_shape
        data_dict['fog_intensity'] = self.fog_I
        return data_dict
    ########################## targets #######################################
    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0

        return h

    def gen_hm_radius(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


    def compute_radius(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return min(r1, r2, r3)

    cnf = kitti_config
    # def get_cen_label():
    def build_targets(self, labels, hflipped):
        cnf = kitti_config
        minX = cnf.boundary['minX']
        maxX = cnf.boundary['maxX']
        minY = cnf.boundary['minY']
        maxY = cnf.boundary['maxY']
        minZ = cnf.boundary['minZ']
        maxZ = cnf.boundary['maxZ']

        num_objects = min(len(labels), self.max_objects)
        hm_l, hm_w = self.hm_size

        hm_main_center = np.zeros((self.num_classes,int(hm_l), int(hm_w)), dtype=np.float32)
        cen_offset = np.zeros((self.max_objects, 2), dtype=np.float32)
        direction = np.zeros((self.max_objects, 2), dtype=np.float32)
        z_coor = np.zeros((self.max_objects, 1), dtype=np.float32)
        dimension = np.zeros((self.max_objects, 3), dtype=np.float32)
        #anglebin = np.zeros((self.max_objects, 2), dtype=np.float32)
        #angleoffset = np.zeros((self.max_objects, 2), dtype=np.float32)
        indices_center = np.zeros((self.max_objects), dtype=np.int64)
        obj_mask = np.zeros((self.max_objects), dtype=np.uint8)
        for k in range(num_objects):
            x, y, z, l, w, h, yaw, cls_id = labels[k]
            cls_id = int(cls_id-1)
            # Invert yaw angle
            yaw = -yaw
            if not ((minX <= x <= maxX) and (minY <= y <= maxY) and (minZ <= z <= maxZ)):
                continue
            if (h <= 0) or (w <= 0) or (l <= 0):
                continue

            bbox_l = l / cnf.bound_size_x * hm_l
            bbox_w = w / cnf.bound_size_y * hm_w
            radius = self.compute_radius((math.ceil(bbox_l), math.ceil(bbox_w)))
            radius = max(0, int(radius))
            center_x = (x - minX) / cnf.bound_size_x * hm_w  # x --> y (invert to 2D image space)
            center_y = (y - minY) / cnf.bound_size_y * hm_l  # y --> x
            center = np.array([center_x, center_y], dtype=np.float32)

            if hflipped:
                center[1] = hm_l - center[1] - 1

            center_int = center.astype(np.int32)

            if cls_id < 0:
                # ignore_ids = [_ for _ in range(self.num_classes)] if cls_id == - 1 else [- cls_id - 2]
                # ignore_ids = [_ for _ in range(self.num_classes)] if cls_id == - 1 else [cls_id]
                # # Consider to make mask ignore
                # for cls_ig in ignore_ids:
                #     self.gen_hm_radius(hm_main_center[cls_ig], center_int, radius)
                # hm_main_center[ignore_ids, center_int[1], center_int[0]] = 0.9999
                continue
            # Generate heatmaps for main center
            self.gen_hm_radius(hm_main_center[cls_id], center, radius)
            # Index of the center
            indices_center[k] = center_int[1] * hm_w + center_int[0]

            # targets for center offset
            cen_offset[k] = center - center_int

            # targets for dimension
            dimension[k, 0] = h
            dimension[k, 1] = w
            dimension[k, 2] = l

            # targets for direction
            direction[k, 0] = math.sin(float(yaw))  # im
            direction[k, 1] = math.cos(float(yaw))  # re
            # im -->> -im
            if hflipped:
                direction[k, 0] = -direction[k, 0]

            # targets for depth
            z_coor[k] = z

            # Generate object masks
            obj_mask[k] = 1
        targets = {
            'hm_cen': hm_main_center,
            'cen_offset': cen_offset,
            'direction': direction,
            'z_coor': z_coor,
            'dim': dimension,
            'indices_center': indices_center,
            'obj_mask': obj_mask,
        }

        '''img = np.zeros_like(targets['hm_cen'], np.uint8)

        for i in range(108):
            for j in range(108):
                for k in range(3):
                    if  targets['hm_cen'][k ,i,j] > 0:
                        print( targets['hm_cen'][k,i,j])
                img[:,i,j] = targets['hm_cen'][:,i,j]*100

        hetmap = img
        print(hetmap.shape)
        hetmap = hetmap.transpose(1,2,0)
        print(hetmap.shape)
        hetmap = cv2.resize(hetmap,(800,800))
        print(hetmap.shape)
        cv2.imshow('x',hetmap)

        cv2.waitKey(0)'''

        return targets
    ##########################targets#######################################

def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = VodDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('vod_infos_%s.pkl' % train_split)
    val_filename = save_path / ('vod_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'vod_infos_trainval.pkl'
    test_filename = save_path / 'vod_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    # dataset.set_split(train_split)
    # vod_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    # with open(train_filename, 'wb') as f:
    #     pickle.dump(vod_infos_train, f)
    # print('Vod info train file is saved to %s' % train_filename)

    # dataset.set_split(val_split)
    # vod_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    # with open(val_filename, 'wb') as f:
    #     pickle.dump(vod_infos_val, f)
    # print('Vod info val file is saved to %s' % val_filename)

    # with open(trainval_filename, 'wb') as f:
    #     pickle.dump(vod_infos_train + vod_infos_val, f)
    # print('Vod info trainval file is saved to %s' % trainval_filename)

    # dataset.set_split('test')
    # vod_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    # with open(test_filename, 'wb') as f:
    #     pickle.dump(vod_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import syskitti_infos
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path= Path('/mnt/32THHD/view_of_delft_PUBLIC/rlfusion_5f/'),
            save_path= Path('/mnt/32THHD/view_of_delft_PUBLIC/rlfusion_5f/')
        )



