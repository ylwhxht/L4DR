'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

from .rdr_base import RadarBase
from .ldr_base import LidarBase
from .pvrcnn_pp import PVRCNNPlusPlus
from .pp_radar import PointPillar_RADAR
from .pp_rlf import PointPillar_RLF
from .second_net import SECONDNet
from .pp_lidar import PointPillar
from .pp_inter import PointPillar_InterF
from .rtnh_lidar import RTNH_L
from .rtnh_lr import RTNH_LR
def build_skeleton(cfg):
    return __all__[cfg.MODEL.SKELETON](cfg)

__all__ = {
    'RadarBase': RadarBase,
    'LidarBase': LidarBase,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'SECONDNet': SECONDNet,
    'PointPillar':PointPillar,
    'PointPillar_RADAR' : PointPillar_RADAR,
    'PointPillar_RLF':PointPillar_RLF,
    'PointPillar_InterF' : PointPillar_InterF,
    'RTNH_L' : RTNH_L,
    'RTNH_LR' : RTNH_LR
}
