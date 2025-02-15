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

from .block import *
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, Conv_vis, Conv_rgb,Conv_rgb1, Conv_ir, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)

__all__ = ('Conv', 'Conv2', 'Conv_vis','LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer', 'Conv_rgb','Conv_rgb1', 'Conv_ir',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP', 'ResNetLayer',
           'OBB', 'IN','IN1','IN2','IN3', 'Multiin', 'MF', 'Add', 'Add2', 'Add5','SelfAttention', 'myTransformerBlock', 'GPT', 'mutual_alignment', 
           'alignment', 'alignment1', 'alignment2', 'Concat3', 'VSS','VSS0', 'VSS1', 'VSS2', 'VSS3','VSS4', 'mutual_alignment1', 'mutual_alignment2',
           'DecomNet1','DecomNet2','MambaBlock','MambaBlock1','MambaBlock2','mutual_align1','mutual_align2')
