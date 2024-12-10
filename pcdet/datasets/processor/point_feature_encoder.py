import numpy as np


class PointFeatureEncoder(object):
    def __init__(self, config, point_cloud_range=None):
        super().__init__()
        self.point_encoding_config = config
        
        if self.point_encoding_config.get('src_feature_list_l', None) is not None:
            assert list(self.point_encoding_config.src_feature_list_r[0:3]) == ['x', 'y', 'z']
            assert list(self.point_encoding_config.src_feature_list_l[0:3]) == ['x', 'y', 'z']
            self.used_feature_list_l = self.point_encoding_config.used_feature_list_l
            self.src_feature_list_l = self.point_encoding_config.src_feature_list_l
            self.used_feature_list_r = self.point_encoding_config.used_feature_list_r
            self.src_feature_list_r = self.point_encoding_config.src_feature_list_r
        else:
            assert list(self.point_encoding_config.src_feature_list[0:3]) == ['x', 'y', 'z']
            self.used_feature_list = self.point_encoding_config.used_feature_list
            self.src_feature_list = self.point_encoding_config.src_feature_list
        self.point_cloud_range = point_cloud_range

    @property
    def num_point_features(self):
        if self.point_encoding_config.get('src_feature_list_l', None) is not None:
            return getattr(self, self.point_encoding_config.encoding_type)(points=None, used_feature_list = self.used_feature_list_l, src_feature_list = self.src_feature_list_l), getattr(self, self.point_encoding_config.encoding_type)(points=None, used_feature_list = self.used_feature_list_r, src_feature_list = self.src_feature_list_r)
        return getattr(self, self.point_encoding_config.encoding_type)(points=None)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        if 'points' in data_dict:
            data_dict['points'], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
                data_dict['points']
            )
        else:
            data_dict['lidar_points'], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
                data_dict['lidar_points'], self.used_feature_list_l, self.src_feature_list_l
            )
            data_dict['radar_points'], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
                data_dict['radar_points'], self.used_feature_list_r, self.src_feature_list_r
            )
        data_dict['use_lead_xyz'] = use_lead_xyz
       
        if self.point_encoding_config.get('filter_sweeps', False) and 'timestamp' in self.src_feature_list:
            max_sweeps = self.point_encoding_config.max_sweeps
            idx = self.src_feature_list.index('timestamp')
            dt = np.round(data_dict['points'][:, idx], 2)
            max_dt = sorted(np.unique(dt))[min(len(np.unique(dt))-1, max_sweeps-1)]
            data_dict['points'] = data_dict['points'][dt <= max_dt]
        
        return data_dict

    def absolute_coordinates_encoding(self, points=None, used_feature_list=None, src_feature_list = None):
        if used_feature_list is not None:
            self.used_feature_list = used_feature_list 
        if src_feature_list is not None:
            self.src_feature_list = src_feature_list        
        if points is None:
            num_output_features = len(self.used_feature_list)
            return num_output_features
        
        assert points.shape[-1] == len(self.src_feature_list)
        point_feature_list = [points[:, 0:3]]
        for x in self.used_feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.src_feature_list.index(x)
            point_feature_list.append(points[:, idx:idx+1])
        point_features = np.concatenate(point_feature_list, axis=1)
        
        return point_features, True
