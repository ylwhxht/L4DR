'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

from .rdr_spcube_head import RdrSpcubeHead
from .ldr_pillars_head import LdrPillarsHead
from .rdr_spcube_head_multi_cls import RdrSpcubeHeadMultiCls
from .anchor_head_single import AnchorHeadSingle
from .anchor_head import AnchorHeadSingle_KR
from .point_head_simple import PointHeadSimple
from .center_head import CenterHead
from .point_head_box import PointHeadPreMask
__all__ = {
    'RdrSpcubeHead': RdrSpcubeHead,
    'LdrPillarsHead': LdrPillarsHead,
    'RdrSpcubeHeadMultiCls': RdrSpcubeHeadMultiCls,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointHeadSimple': PointHeadSimple,
    'CenterHead': CenterHead,
    'AnchorHeadSingle_KR' : AnchorHeadSingle_KR,
    'PointHeadPreMask': PointHeadPreMask
}
