# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)
from .iRMB import C2f_iRMB,iRMB
from .DynamicHead import DetectDynamicHead,SegmentDynamicHead
from .GhostNetV3 import GhostNetV3
from .MobileNetV4 import MobileNetV4
from .StarNet import StarNet_s
from .HGNetV2 import HGNetv2
from .CBMA import CBAM
from .SE import C2fCIB, C2f_SE
from .CA import CoordAtt
from .EMA import EMA, C2f_EMA
from .EfficientVit import *
from .ShuffleNetV2 import ShuffleNetV2
from .FasterNet import FasterNet
from .RFAConv_Head import DetectRFAConv, SegmentRFAConv
from .BiFPN import BiFPN_Add2, BiFPN_Add3
from .AFPN import ASFF_2, ASFF_3
from .SlimNeck import VoVGSCSP, GSConv
from .HSFPN import *
from .EfficientVit import *
from .DSConvHead import DetectDSConv, SegmentDSConv
from .RFAConv_Head import DetectRFAConv, SegmentRFAConv
from .DBBHead import DetectDBB, SegmentDBB
from .RepConvHead import DetectRepConv, SegmentRepConv
from .DWConvHead import DWConvHead
from .SAHead import Detect_SA, Segment_SA
from .FRMHead import Detect_FRM

#__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
#           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
#           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
#           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
#          'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
#           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP')
