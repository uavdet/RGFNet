# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import sys
sys.path.append('/data/zhangwei/yolov8_multi/Burstormer-main/Burst De-noising')
from utils.metrics import PSNR
psnr_fn = PSNR(boundary_ignore=40)

import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from torch.nn.init import _calculate_fan_in_and_fan_out
import torch.utils.checkpoint as checkpoint

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda import amp
from torchvision.ops import DeformConv2d
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock
from torch.nn import init, Sequential

import pytorch_lightning as pl
from torchvision.ops import DeformConv2d
from pytorch_lightning import seed_everything
from einops import rearrange
import numbers

from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
# from utils.metrics import PSNR
# psnr_fn = PSNR(boundary_ignore=40)

seed_everything(13)


__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost', 'IN3','IN4',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3', 'ResNetLayer', 'IN','IN1','IN2', 'Multiin', 
           'MF', 'Add','Add2','Add5', 'SelfAttention', 'myTransformerBlock', 'GPT', 'mutual_alignment', 'mutual_alignment1', 
           'mutual_alignment2','alignment', 'alignment1', 'alignment2', 'Concat3', 'VSS', 'VSS0', 'VSS1', 'VSS2', 'VSS3','VSS4',
           'DecomNet1','DecomNet2', 'MambaBlock','MambaBlock1','MambaBlock2','mutual_align1','mutual_align2')

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

###暗光增强###
class DecomNet1(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet1, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size,
                                    padding=1, padding_mode='replicate')

    def forward(self, input_im):
        b,c,h,w = input_im.shape
        x_im = input_im
        x_new = input_im.clone()
        input_max= torch.max(input_im, dim=1, keepdim=True)[0]
        input_img= torch.cat((input_max, input_im), dim=1)
        feats0   = self.net1_conv0(input_img)
        featss   = self.net1_convs(feats0)
        outs     = self.net1_recon(featss)
        R        = torch.sigmoid(outs[:, 0:3, :, :])
        L        = torch.sigmoid(outs[:, 3:4, :, :])
        L1 = x_im * L
        L2 = x_im * L + x_im
        L3 = x_im * R
        L4 = x_im * R + x_im


        for i in range(b):           
            x_new[i][(x_new[i])<0.196] = 0.1
            x_new[i][(x_new[i])>=0.196] = 0
            

            if (x_new[i]*10).sum() / (h*w*3) > 0.18:
                x_im[i] = R[i]
       
        light = (x_im, L, R, L1, L2, L3, L4)
        return light


  

# RLN 输入 x_in(B、H、W、C) → 输出 x_out(B、H、W、C)，rescale，rebias
class RLN(nn.Module):
    r"""Revised LayerNorm"""
    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias



class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = VSS1(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = Mlp(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, H, W, Just):
       # [B,H,W,C]
        B, L, C = input.shape
        input = input.view(B, H, W, C).contiguous()
        ###
        # input_new = input[...,:(C//3)*2]
        ###
        x = self.ln_1(input)
        # x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        x = input*self.skip_scale + self.drop_path(self.self_attention(x, Just))
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).view(B, L, C).contiguous()).view(B, H, W, C).contiguous()
        x = x.view(B, L, C)
        return x



class MambaBlock(nn.Module):
    def __init__(self,
                 dim,
                 depth = 1,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 use_checkpoint=False):
        super().__init__()
         
        self.dim = dim*3+1
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        # 用于调整通道数的卷积层
        self.conv1 = nn.Conv2d(3, self.dim//6, kernel_size=1)
        self.conv2 = nn.Conv2d(self.dim//12, self.dim//6, kernel_size=1)

        

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = VSSBlock(hidden_dim=self.dim //2,
                             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                             norm_layer=nn.LayerNorm,
                             attn_drop_rate=0,
                             d_state=d_state,
                             expand=self.mlp_ratio,
                             )
            self.blocks.append(block)


    def forward(self, x): 

        if type(x[0]) == tuple :
            rgb = x[0][0]  # rgb_fea (tensor): dim:(B, C, H, W)
            ir = x[0][1]   # ir_fea (tensor): dim:(B, C, H, W)
            R = x[0][2]
            Just = x[1][1] 

        elif type(x[0]) == list :
            rgb = x[0][0]  # rgb_fea (tensor): dim:(B, C, H, W)
            ir = x[0][1]  # ir_fea (tensor): dim:(B, C, H, W)
            R = x[0][2]
            Just = x[1][1]
                   
        else:
            if len(x) == 3:
                rgb = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
                ir = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
                R = x[2][2]
                Just = x[2][1]
                R = self.conv1(R)  # 调整通道数到128
            else :
                rgb = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
                ir = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
                R = x[2][2]
                Just = x[3][1]
                R = self.conv2(R)  # 调整通道数到128
            B,C,H,W = rgb.shape
            # 使用双线性插值将空间维度调整为 (80, 80)
            R = F.interpolate(R, size=(H, W), mode='bilinear', align_corners=False)
            # L = F.interpolate(L, size=(H, W), mode='bilinear', align_corners=False)
        
        x = torch.concat((rgb,ir,R) , dim = 1)

        B,C,H,W = x.shape    
        x = x.permute(0,2,3,1).view(B,H*W,C)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, H,W,Just)
        x = x.view(B,H,W,C).permute(0,3,1,2)

        rgb_out = x[:,:C//3,...]
        ir_out = x[:,C//3:(C//3)*2,...]
        r_out = x[:,(C//3)*2:,...]


        return rgb_out, ir_out, r_out
        # return rgb_fea_out, ir_fea_out

class MambaBlock1(nn.Module):
    def __init__(self,
                 dim,
                 depth = 1,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 use_checkpoint=False):
        super().__init__()
         
        self.dim = dim*2
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        # 用于调整通道数的卷积层
        self.conv1 = nn.Conv2d(3, self.dim//6, kernel_size=1)
        self.conv2 = nn.Conv2d(self.dim//12, self.dim//6, kernel_size=1)

        

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = VSSBlock(hidden_dim=self.dim //2,
                             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                             norm_layer=nn.LayerNorm,
                             attn_drop_rate=0,
                             d_state=d_state,
                             expand=self.mlp_ratio,
                             )
            self.blocks.append(block)


    def forward(self, x): 

        if type(x[0]) == tuple :
            rgb = x[0][0]  # rgb_fea (tensor): dim:(B, C, H, W)
            ir = x[0][1]   # ir_fea (tensor): dim:(B, C, H, W)
            # R = x[0][2]
            Just = x[1][1] 


        elif type(x[0]) == list :
            rgb = x[0][0]  # rgb_fea (tensor): dim:(B, C, H, W)
            ir = x[0][1]  # ir_fea (tensor): dim:(B, C, H, W)
            # R = x[0][2]
            Just = x[1][1]

                   
        else:
            if len(x) == 3:
                rgb = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
                ir = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
                R = x[2][2]
                Just = x[2][1]
                R = self.conv1(R)  # 调整通道数到128
            else :
                rgb = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
                ir = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
                R = x[2][2]
                Just = x[3][1]
                R = self.conv2(R)  # 调整通道数到128
            B,C,H,W = rgb.shape
            # 使用双线性插值将空间维度调整为 (80, 80)
            R = F.interpolate(R, size=(H, W), mode='bilinear', align_corners=False)
            # L = F.interpolate(L, size=(H, W), mode='bilinear', align_corners=False)
        
        x = torch.concat((rgb,ir) , dim = 1)

        B,C,H,W = x.shape    
        x = x.permute(0,2,3,1).view(B,H*W,C)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, H,W,Just)
        x = x.view(B,H,W,C).permute(0,3,1,2)

        rgb_out = x[:,:C//2,...]
        ir_out = x[:,C//2:(C//2)*2,...]
        # r_out = x[:,(C//3)*2:,...]


        return rgb_out, ir_out
        # return rgb_fea_out, ir_fea_out



class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class NAMAttention(nn.Module):
    def __init__(self, channels, out_channels=None, no_spatial=True):
        super(NAMAttention, self).__init__()
        self.Channel_Att = Channel_Att(channels)
    def forward(self, x):
        x_out1=self.Channel_Att(x)
        return x_out1

class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
      
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)


    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        x = torch.sigmoid(x) * residual #
        
        return x
class Conv_CBAM(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_CBAM, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()
        self.ca = ChannelAttention(c2)
        self.sa = SpatialAttention()
 
    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x
 
    def fuseforward(self, x):
        return self.act(self.conv(x))
    


# VSS1 输入 B、C、H、W → 输出 B、C、H、W
class VSS1(nn.Module): # 四方向选择性扫描操作
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model//3
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank # math.ceil 向上取整

        # 用于调整通道数的卷积层
        # self.conv1 = nn.Conv2d(3, self.d_model, kernel_size=1)

        # self.duckness = nn.Parameter(torch.tensor(0.338)) #  0.44 0.3380  0.2  0.6802             
        # self.duckvalue = nn.Parameter(torch.tensor(0.7848))  # 0.6124 0.6082 0.6802 0.6682 0.8049 0.7848 0.7847 0.6340 1.2734 0.8758 0.7736 1.0531 



        self.act = nn.SiLU()

# RGB
###---------------------------------------------------------------------------------------------------------------------------------####
        
        # in_proj：一个线性层，将输入特征投影到内部特征维度的两倍，用于输入的预处理。
        self.in_proj1 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs) # 128 256 
        self.conv2d1 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
			padding_mode='reflect',
            **factory_kwargs,
        )
        # self.act1 = nn.SiLU()

        # 四个线性层组成的元组，用于将内部特征投影到矩阵分解的维度。
        self.x_proj1 = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )

        # 将 x_proj 中四个线性层的权重堆叠成一个参数张量。
        self.x_proj_weight1 = nn.Parameter(torch.stack([t.weight for t in self.x_proj1], dim=0))  # (K=4, N, inner)
        del self.x_proj1

        # 四个矩阵分解初始化函数组成的元组，用于初始化矩阵分解。
        self.dt_projs1 = ( # dt_init返回经过初始化后的linear(dt_rank,d_inner),weight.shape:[d_inner,dt_rank],bias.shape:[d_inner,]
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )

        # 分别将 dt_projs 中四个矩阵分解初始化函数的权重和偏置堆叠成参数张量。
        self.dt_projs_weight1 = nn.Parameter(torch.stack([t.weight for t in self.dt_projs1], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias1 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs1], dim=0))  # (K=4, inner)
        del self.dt_projs1

        # 分别初始化了用于选择性扫描的参数
        self.A_logs1 = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N),生成一个[copies * d_inner * d_state,] 的矩阵
        self.Ds1 = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, N),生成一个[copies * d_inner,] 的矩阵

        # 选择性扫描函数。
        self.selective_scan1 = selective_scan_fn 

        # 一个层归一化层，用于规范化输出。
        self.out_norm1 = nn.LayerNorm(self.d_inner)

        # 一个线性层，将内部特征投影回输出特征维度。
        self.out_proj1 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # 一个Dropout层，用于防止过拟合。
        self.dropout1 = nn.Dropout(dropout) if dropout > 0. else None


###---------------------------------------------------------------------------------------------------------------------------------####

# IR
###---------------------------------------------------------------------------------------------------------------------------------####
        
        # in_proj：一个线性层，将输入特征投影到内部特征维度的两倍，用于输入的预处理。
        self.in_proj2 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs) # 128 256 
        self.conv2d2 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
			padding_mode='reflect',
            **factory_kwargs,
        )
        # self.act1 = nn.SiLU()

        # 四个线性层组成的元组，用于将内部特征投影到矩阵分解的维度。
        self.x_proj2 = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )

        # 将 x_proj 中四个线性层的权重堆叠成一个参数张量。
        self.x_proj_weight2 = nn.Parameter(torch.stack([t.weight for t in self.x_proj2], dim=0))  # (K=4, N, inner)
        del self.x_proj2

        # 四个矩阵分解初始化函数组成的元组，用于初始化矩阵分解。
        self.dt_projs2 = ( # dt_init返回经过初始化后的linear(dt_rank,d_inner),weight.shape:[d_inner,dt_rank],bias.shape:[d_inner,]
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )

        # 分别将 dt_projs 中四个矩阵分解初始化函数的权重和偏置堆叠成参数张量。
        self.dt_projs_weight2 = nn.Parameter(torch.stack([t.weight for t in self.dt_projs2], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias2 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs2], dim=0))  # (K=4, inner)
        del self.dt_projs2

        # 分别初始化了用于选择性扫描的参数
        self.A_logs2 = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N),生成一个[copies * d_inner * d_state,] 的矩阵
        self.Ds2 = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, N),生成一个[copies * d_inner,] 的矩阵

        # 选择性扫描函数。
        self.selective_scan2 = selective_scan_fn 

        # 一个层归一化层，用于规范化输出。
        self.out_norm2 = nn.LayerNorm(self.d_inner)

        # 一个线性层，将内部特征投影回输出特征维度。
        self.out_proj2 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # 一个Dropout层，用于防止过拟合。
        self.dropout2 = nn.Dropout(dropout) if dropout > 0. else None
###---------------------------------------------------------------------------------------------------------------------------------####

# R1
###---------------------------------------------------------------------------------------------------------------------------------####
        
        # in_proj：一个线性层，将输入特征投影到内部特征维度的两倍，用于输入的预处理。
        self.in_proj3 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs) # 128 256 
        self.conv2d3 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
			padding_mode='reflect',
            **factory_kwargs,
        )
        # self.act1 = nn.SiLU()

        # 四个线性层组成的元组，用于将内部特征投影到矩阵分解的维度。
        self.x_proj3 = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )

        # 将 x_proj 中四个线性层的权重堆叠成一个参数张量。
        self.x_proj_weight3 = nn.Parameter(torch.stack([t.weight for t in self.x_proj3], dim=0))  # (K=4, N, inner)
        del self.x_proj3

        # 四个矩阵分解初始化函数组成的元组，用于初始化矩阵分解。
        self.dt_projs3 = ( # dt_init返回经过初始化后的linear(dt_rank,d_inner),weight.shape:[d_inner,dt_rank],bias.shape:[d_inner,]
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )

        # 分别将 dt_projs 中四个矩阵分解初始化函数的权重和偏置堆叠成参数张量。
        self.dt_projs_weight3 = nn.Parameter(torch.stack([t.weight for t in self.dt_projs3], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias3 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs3], dim=0))  # (K=4, inner)
        del self.dt_projs3

        # 分别初始化了用于选择性扫描的参数
        self.A_logs3 = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N),生成一个[copies * d_inner * d_state,] 的矩阵
        self.Ds3 = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, N),生成一个[copies * d_inner,] 的矩阵

        # 选择性扫描函数。
        self.selective_scan3 = selective_scan_fn 

        # 一个层归一化层，用于规范化输出。
        self.out_norm3 = nn.LayerNorm(self.d_inner)

        # 一个线性层，将内部特征投影回输出特征维度。
        self.out_proj3 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # 一个Dropout层，用于防止过拟合。
        self.dropout3 = nn.Dropout(dropout) if dropout > 0. else None


###---------------------------------------------------------------------------------------------------------------------------------####

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        

        # 矩阵分解初始化
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # 初始化 dt 偏置，使得 F.softplus(dt_bias) 位于 dt_min 和 dt_max 之间。
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor) # 生成shape为[d_inner,]的张量
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # [d_inner,]
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj
    


    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device), # 生成1到d_state（包含首尾）的等差数列张量，形状为[d_state,] 
            "n -> d n",  # repeat操作，形状为[d_inner,d_state] 
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies) # repeat操作，形状为[copies,d_inner,d_state] 
            if merge:
                A_log = A_log.flatten(0, 1) # [copies * d_inner * d_state,] 
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    # forward_core方法执行了模型的核心运算，包括数据的多视角合并处理和多个特殊投影的应用。
    # 这部分利用了特有的时间步长项目（dt_projs）和自定义的扫描函数（selective_scan）进行复杂的数据转换和处理，最终产生了四个主要的输出。
    def forward_core(self, x: torch.Tensor): # 输入B，C，H，W  输出B，C，H，W
        # 变量解释：
        # 这个函数接受一个torch.Tensor类型的输入x,其形状为(batch_size, channels, height, width)。
        # B, C, H, W = x.shape: 这行代码从输入张量x的形状中提取了四个维度的值，分别是批大小（batch_size）、通道数（channels）、高度（height）和宽度（width）。
        # L = H * W: 这行代码计算了输入张量中空间维度的元素数量，即图片的像素数量。
        # K = 4: 这个变量定义了一个值为4的常数K
        rgb = x[0]
        ir = x[1]
        R1 = x[2]
        Just = x[3]
        Just = torch.tensor(Just,dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).cuda()


        B, C, H, W = rgb.shape
        L = H * W
        K = 4
# RGB
###---------------------------------------------------------------------------------------------------------------------------------####
        # 数据处理：
        # x_hwwh = ...: 这行代码对输入张量进行了一系列操作，将其重塑为一个新的张量x_hwwh，形状为(batch_size, 2, K, L)。
        # 具体操作包括将原始张量展平为(batch_size, channels, L)的形状，然后将其沿着空间维度进行堆叠和转置，以得到期望的形状。
        x_hwwh1 = torch.stack([rgb.view(B, -1, L), torch.transpose(rgb, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # xs = torch.cat([...], dim=1): 这行代码将x_hwwh和其在空间维度上的翻转拼接在一起，形成一个新的张量xs，其形状为(batch_size, 4, 2, L)。
        xs1 = torch.cat([x_hwwh1, torch.flip(x_hwwh1, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        # 张量运算：
        # x_dbl = torch.einsum(...): 这行代码使用torch.einsum函数执行张量乘法，将xs与self.x_proj_weight相乘，并对其中的维度进行合并和变换。结果被分割成三个张量dts、Bs和Cs。
        x_dbl1 = torch.einsum("b k d l, k c d -> b k c l", xs1.view(B, K, -1, L), self.x_proj_weight1) # 张量乘法，输出[]
        dts1, Bs1, Cs1 = torch.split(x_dbl1, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        # dts = torch.einsum(...): 这行代码类似于上一行，将dts张量再次与self.dt_projs_weight相乘，并对其进行合并和变换。
        dts1 = torch.einsum("b k r l, k d r -> b k d l", dts1.view(B, K, -1, L), self.dt_projs_weight1)
        
        # 数据类型转换：
        # xs = xs.float().view(...), dts = dts.contiguous().float().view(...), Bs = Bs.float().view(...), Cs = Cs.float().view(...): 
        # 这几行代码将一些张量转换为浮点型，并重新调整其形状，是为了与后续的操作兼容。
        xs1 = xs1.float().view(B, -1, L)
        dts1 = dts1.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs1 = Bs1.float().view(B, K, -1, L)
        Cs1 = Cs1.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds1 = self.Ds1.float().view(-1)
        As1 = -torch.exp(self.A_logs1.float()).view(-1, self.d_state)
        dt_projs_bias1 = self.dt_projs_bias1.float().view(-1) # (k * d)
        # print(As)


###---------------------------------------------------------------------------------------------------------------------------------####
        
# IR      
###---------------------------------------------------------------------------------------------------------------------------------####
        # 数据处理：
        # x_hwwh = ...: 这行代码对输入张量进行了一系列操作，将其重塑为一个新的张量x_hwwh，形状为(batch_size, 2, K, L)。
        # 具体操作包括将原始张量展平为(batch_size, channels, L)的形状，然后将其沿着空间维度进行堆叠和转置，以得到期望的形状。
        x_hwwh2 = torch.stack([ir.view(B, -1, L), torch.transpose(ir, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # xs = torch.cat([...], dim=1): 这行代码将x_hwwh和其在空间维度上的翻转拼接在一起，形成一个新的张量xs，其形状为(batch_size, 4, 2, L)。
        xs2 = torch.cat([x_hwwh2, torch.flip(x_hwwh2, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        # 张量运算：
        # x_dbl = torch.einsum(...): 这行代码使用torch.einsum函数执行张量乘法，将xs与self.x_proj_weight相乘，并对其中的维度进行合并和变换。结果被分割成三个张量dts、Bs和Cs。
        x_dbl2 = torch.einsum("b k d l, k c d -> b k c l", xs2.view(B, K, -1, L), self.x_proj_weight2) # 张量乘法，输出[]
        dts2, Bs2, Cs2 = torch.split(x_dbl2, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        # dts = torch.einsum(...): 这行代码类似于上一行，将dts张量再次与self.dt_projs_weight相乘，并对其进行合并和变换。
        dts2 = torch.einsum("b k r l, k d r -> b k d l", dts2.view(B, K, -1, L), self.dt_projs_weight2)
        
        # 数据类型转换：
        # xs = xs.float().view(...), dts = dts.contiguous().float().view(...), Bs = Bs.float().view(...), Cs = Cs.float().view(...): 
        # 这几行代码将一些张量转换为浮点型，并重新调整其形状，是为了与后续的操作兼容。
        xs2 = xs2.float().view(B, -1, L)
        dts2 = dts2.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs2 = Bs2.float().view(B, K, -1, L)
        Cs2 = Cs2.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds2 = self.Ds2.float().view(-1)
        As2 = -torch.exp(self.A_logs2.float()).view(-1, self.d_state)
        dt_projs_bias2 = self.dt_projs_bias2.float().view(-1) # (k * d)
        # print(As)


###---------------------------------------------------------------------------------------------------------------------------------####

# R1
###---------------------------------------------------------------------------------------------------------------------------------####
        # 数据处理：
        # x_hwwh = ...: 这行代码对输入张量进行了一系列操作，将其重塑为一个新的张量x_hwwh，形状为(batch_size, 2, K, L)。
        # 具体操作包括将原始张量展平为(batch_size, channels, L)的形状，然后将其沿着空间维度进行堆叠和转置，以得到期望的形状。
        x_hwwh3 = torch.stack([R1.view(B, -1, L), torch.transpose(R1, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # xs = torch.cat([...], dim=1): 这行代码将x_hwwh和其在空间维度上的翻转拼接在一起，形成一个新的张量xs，其形状为(batch_size, 4, 2, L)。
        xs3 = torch.cat([x_hwwh3, torch.flip(x_hwwh3, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        # 张量运算：
        # x_dbl = torch.einsum(...): 这行代码使用torch.einsum函数执行张量乘法，将xs与self.x_proj_weight相乘，并对其中的维度进行合并和变换。结果被分割成三个张量dts、Bs和Cs。
        x_dbl3 = torch.einsum("b k d l, k c d -> b k c l", xs3.view(B, K, -1, L), self.x_proj_weight3) # 张量乘法，输出[]
        dts3, Bs3, Cs3 = torch.split(x_dbl3, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        # dts = torch.einsum(...): 这行代码类似于上一行，将dts张量再次与self.dt_projs_weight相乘，并对其进行合并和变换。
        dts3 = torch.einsum("b k r l, k d r -> b k d l", dts3.view(B, K, -1, L), self.dt_projs_weight3)
        
        # 数据类型转换：
        # xs = xs.float().view(...), dts = dts.contiguous().float().view(...), Bs = Bs.float().view(...), Cs = Cs.float().view(...): 
        # 这几行代码将一些张量转换为浮点型，并重新调整其形状，是为了与后续的操作兼容。
        xs3 = xs3.float().view(B, -1, L)
        dts3 = dts3.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs3 = Bs3.float().view(B, K, -1, L)
        Cs3 = Cs3.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds3 = self.Ds3.float().view(-1)
        As3 = -torch.exp(self.A_logs3.float()).view(-1, self.d_state)
        dt_projs_bias3 = self.dt_projs_bias3.float().view(-1) # (k * d)
        # print(As)

# RGB
###---------------------------------------------------------------------------------------------------------------------------------####
        
        # out_y = self.selective_scan(...): 这行代码调用了一个自定义函数selective_scan，传递了一系列张量作为参数，并返回一个张量out_y。  dts1*Just+dts2+dts3 dts1*Just+dts3
        out_y1 = self.selective_scan1(
            xs1, dts1*Just+dts3,
            As1, Bs1, Cs1+Cs2+Cs3, Ds1, z=None,
            delta_bias=dt_projs_bias1,

            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y1.dtype == torch.float

        # inv_y = ..., wh_y = ..., invwh_y = ...: 这几行代码对out_y进行了一些处理，将其切片、翻转和重塑为期望的形状，并将结果分别赋给了inv_y、wh_y和invwh_y。
        inv_y1 = torch.flip(out_y1[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y1 = torch.transpose(out_y1[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y1 = torch.transpose(inv_y1[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        rgb_shu = (out_y1[:, 0], inv_y1[:, 0], wh_y1, invwh_y1)
####        
        # return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y: 这行代码返回了四个张量作为函数的输出。
###---------------------------------------------------------------------------------------------------------------------------------####

# IR
###---------------------------------------------------------------------------------------------------------------------------------####

        # out_y = self.selective_scan(...): 这行代码调用了一个自定义函数selective_scan，传递了一系列张量作为参数，并返回一个张量out_y。
        # VSS2
        out_y2 = self.selective_scan2(
            xs2, dts2,
            As2, Bs2, Cs1+Cs2+Cs3, Ds2, z=None,
            delta_bias=dt_projs_bias2,

            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y2.dtype == torch.float

        # inv_y = ..., wh_y = ..., invwh_y = ...: 这几行代码对out_y进行了一些处理，将其切片、翻转和重塑为期望的形状，并将结果分别赋给了inv_y、wh_y和invwh_y。
        inv_y2 = torch.flip(out_y2[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y2 = torch.transpose(out_y2[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y2 = torch.transpose(inv_y2[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
####        
        # return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y: 这行代码返回了四个张量作为函数的输出。
        ir_shu = (out_y2[:, 0], inv_y2[:, 0], wh_y2, invwh_y2)
###---------------------------------------------------------------------------------------------------------------------------------####

# R1
###---------------------------------------------------------------------------------------------------------------------------------####
        
        # out_y = self.selective_scan(...): 这行代码调用了一个自定义函数selective_scan，传递了一系列张量作为参数，并返回一个张量out_y。
        out_y3 = self.selective_scan3(
            xs3, dts3,
            As3, Bs3, Cs1+Cs2+Cs3, Ds3, z=None,
            delta_bias=dt_projs_bias3,

            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y3.dtype == torch.float

        # inv_y = ..., wh_y = ..., invwh_y = ...: 这几行代码对out_y进行了一些处理，将其切片、翻转和重塑为期望的形状，并将结果分别赋给了inv_y、wh_y和invwh_y。
        inv_y3 = torch.flip(out_y3[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y3 = torch.transpose(out_y3[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y3 = torch.transpose(inv_y3[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        r1_shu = (out_y3[:, 0], inv_y3[:, 0], wh_y3, invwh_y3)
####        
        # return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y: 这行代码返回了四个张量作为函数的输出。
###---------------------------------------------------------------------------------------------------------------------------------####

        result = (rgb_shu, ir_shu, r1_shu)

        return result

        # return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    

    # 然后，在forward中，这四个输出通过特定的变换和组合，最后通过归一化、激活函数和最后的out_proj层，生成最终的输出。
    def forward(self, x: torch.Tensor, Just,**kwargs):

###---------------------------------------------------------------------------------------------------------------------------------####
        B, H, W, C = x.shape
        rgb1 = x[:,...,:C//3]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir1 = x[:,...,C//3:(C//3)*2]   # ir_fea (tensor): dim:(B, C, H, W) 
        r1 = x[:,...,(C//3)*2:(C//3)*3]

        
# RGB               
###---------------------------------------------------------------------------------------------------------------------------------####
        # rgb1 = rgb.permute(0,2,3,1)
        B1, H1, W1, C1 = rgb1.shape


        rgb1z1 = self.in_proj1(rgb1) # Linear
        rgb1, z1 = rgb1z1.chunk(2, dim=-1)

        rgb1 = rgb1.permute(0, 3, 1, 2).contiguous()
        rgb1 = self.act(self.conv2d1(rgb1))
###---------------------------------------------------------------------------------------------------------------------------------####
        # ir1 = ir.permute(0,2,3,1)
        B2, H2, W2, C2 = ir1.shape

        ir1z2 = self.in_proj2(ir1)
        ir1, z2 = ir1z2.chunk(2, dim=-1)

        ir1 = ir1.permute(0, 3, 1, 2).contiguous()
        ir1 = self.act(self.conv2d2(ir1))
###---------------------------------------------------------------------------------------------------------------------------------####
        # r1 = R1.permute(0,2,3,1)
        B3, H3, W3, C3 = r1.shape

        r1z3 = self.in_proj3(r1)
        r1, z3 = r1z3.chunk(2, dim=-1)

        r1 = r1.permute(0, 3, 1, 2).contiguous()
        r1 = self.act(self.conv2d3(r1))
###---------------------------------------------------------------------------------------------------------------------------------####
        fea = (rgb1, ir1, r1, Just)

        rgb_fea ,ir_fea, r1_fea = self.forward_core(fea)
     

        rgb_y1, rgb_y2, rgb_y3, rgb_y4 = rgb_fea
        ir_y1, ir_y2, ir_y3, ir_y4 = ir_fea
        r1_y1, r1_y2, r1_y3, r1_y4 = r1_fea
        
        assert rgb_y1.dtype == torch.float32
        y_rgb1 = rgb_y1 + rgb_y2 + rgb_y3 + rgb_y4

        assert ir_y1.dtype == torch.float32
        y_ir1 = ir_y1 + ir_y2 + ir_y3 + ir_y4

        assert r1_y1.dtype == torch.float32
        y_r1 = r1_y1 + r1_y2 + r1_y3 + r1_y4

        y_rgb = y_rgb1 + y_r1
        y_ir = y_ir1 + y_r1


        y_rgb = torch.transpose(y_rgb, dim0=1, dim1=2).contiguous().view(B1, H1, W1, -1)
        y_rgb = self.out_norm1(y_rgb)
        y_rgb = y_rgb * F.silu(z1)
        rgb_out = self.out_proj1(y_rgb)
        if self.dropout1 is not None:
            out = self.dropout1(out)
        # rgb_out = rgb_out.permute(0,3,1,2)



        y_ir = torch.transpose(y_ir, dim0=1, dim1=2).contiguous().view(B2, H2, W2, -1)
        y_ir = self.out_norm2(y_ir)
        y_ir = y_ir * F.silu(z2)
        ir_out = self.out_proj2(y_ir)
        if self.dropout2 is not None:
            out = self.dropout2(out)
        # ir_out = ir_out.permute(0,3,1,2)
        

        y_r1 = torch.transpose(y_r1, dim0=1, dim1=2).contiguous().view(B3, H3, W3, -1)
        y_r1 = self.out_norm3(y_r1)
        y_r1 = y_r1 * F.silu(z3)
        r1_out = self.out_proj3(y_r1)
        if self.dropout3 is not None:
            out = self.dropout3(out)
        # r1_out = r1_out.permute(0,3,1,2)


# #####################################################
        # rgb_out = rgb_out + r1_out
        # ir_out = ir_out + r1_out
# #####################################################

        out = torch.concat([rgb_out,ir_out,r1_out] , dim = -1)

        # return rgb_out, ir_out
        return out
    
# VSS2 输入 B、C、H、W → 输出 B、C、H、W
class VSS2(nn.Module): # 四方向选择性扫描操作
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank # math.ceil 向上取整

        # 用于调整通道数的卷积层
        self.conv1 = nn.Conv2d(3, self.d_model, kernel_size=1)

        self.duckness = nn.Parameter(torch.tensor(0.2))
        self.duckvalue = nn.Parameter(torch.tensor(0.3))



        self.act = nn.SiLU()

# RGB
###---------------------------------------------------------------------------------------------------------------------------------####
        
        # in_proj：一个线性层，将输入特征投影到内部特征维度的两倍，用于输入的预处理。
        self.in_proj1 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs) # 128 256 
        self.conv2d1 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
			padding_mode='reflect',
            **factory_kwargs,
        )
        # self.act1 = nn.SiLU()

        # 四个线性层组成的元组，用于将内部特征投影到矩阵分解的维度。
        self.x_proj1 = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )

        # 将 x_proj 中四个线性层的权重堆叠成一个参数张量。
        self.x_proj_weight1 = nn.Parameter(torch.stack([t.weight for t in self.x_proj1], dim=0))  # (K=4, N, inner)
        del self.x_proj1

        # 四个矩阵分解初始化函数组成的元组，用于初始化矩阵分解。
        self.dt_projs1 = ( # dt_init返回经过初始化后的linear(dt_rank,d_inner),weight.shape:[d_inner,dt_rank],bias.shape:[d_inner,]
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )

        # 分别将 dt_projs 中四个矩阵分解初始化函数的权重和偏置堆叠成参数张量。
        self.dt_projs_weight1 = nn.Parameter(torch.stack([t.weight for t in self.dt_projs1], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias1 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs1], dim=0))  # (K=4, inner)
        del self.dt_projs1

        # 分别初始化了用于选择性扫描的参数
        self.A_logs1 = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N),生成一个[copies * d_inner * d_state,] 的矩阵
        self.Ds1 = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, N),生成一个[copies * d_inner,] 的矩阵

        # 选择性扫描函数。
        self.selective_scan1 = selective_scan_fn 

        # 一个层归一化层，用于规范化输出。
        self.out_norm1 = nn.LayerNorm(self.d_inner)

        # 一个线性层，将内部特征投影回输出特征维度。
        self.out_proj1 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # 一个Dropout层，用于防止过拟合。
        self.dropout1 = nn.Dropout(dropout) if dropout > 0. else None


###---------------------------------------------------------------------------------------------------------------------------------####

# IR
###---------------------------------------------------------------------------------------------------------------------------------####
        
        # in_proj：一个线性层，将输入特征投影到内部特征维度的两倍，用于输入的预处理。
        self.in_proj2 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs) # 128 256 
        self.conv2d2 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
			padding_mode='reflect',
            **factory_kwargs,
        )
        # self.act1 = nn.SiLU()

        # 四个线性层组成的元组，用于将内部特征投影到矩阵分解的维度。
        self.x_proj2 = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )

        # 将 x_proj 中四个线性层的权重堆叠成一个参数张量。
        self.x_proj_weight2 = nn.Parameter(torch.stack([t.weight for t in self.x_proj2], dim=0))  # (K=4, N, inner)
        del self.x_proj2

        # 四个矩阵分解初始化函数组成的元组，用于初始化矩阵分解。
        self.dt_projs2 = ( # dt_init返回经过初始化后的linear(dt_rank,d_inner),weight.shape:[d_inner,dt_rank],bias.shape:[d_inner,]
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )

        # 分别将 dt_projs 中四个矩阵分解初始化函数的权重和偏置堆叠成参数张量。
        self.dt_projs_weight2 = nn.Parameter(torch.stack([t.weight for t in self.dt_projs2], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias2 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs2], dim=0))  # (K=4, inner)
        del self.dt_projs2

        # 分别初始化了用于选择性扫描的参数
        self.A_logs2 = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N),生成一个[copies * d_inner * d_state,] 的矩阵
        self.Ds2 = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, N),生成一个[copies * d_inner,] 的矩阵

        # 选择性扫描函数。
        self.selective_scan2 = selective_scan_fn 

        # 一个层归一化层，用于规范化输出。
        self.out_norm2 = nn.LayerNorm(self.d_inner)

        # 一个线性层，将内部特征投影回输出特征维度。
        self.out_proj2 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # 一个Dropout层，用于防止过拟合。
        self.dropout2 = nn.Dropout(dropout) if dropout > 0. else None
###---------------------------------------------------------------------------------------------------------------------------------####

# R1
###---------------------------------------------------------------------------------------------------------------------------------####
        
        # in_proj：一个线性层，将输入特征投影到内部特征维度的两倍，用于输入的预处理。
        self.in_proj3 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs) # 128 256 
        self.conv2d3 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
			padding_mode='reflect',
            **factory_kwargs,
        )
        # self.act1 = nn.SiLU()

        # 四个线性层组成的元组，用于将内部特征投影到矩阵分解的维度。
        self.x_proj3 = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )

        # 将 x_proj 中四个线性层的权重堆叠成一个参数张量。
        self.x_proj_weight3 = nn.Parameter(torch.stack([t.weight for t in self.x_proj3], dim=0))  # (K=4, N, inner)
        del self.x_proj3

        # 四个矩阵分解初始化函数组成的元组，用于初始化矩阵分解。
        self.dt_projs3 = ( # dt_init返回经过初始化后的linear(dt_rank,d_inner),weight.shape:[d_inner,dt_rank],bias.shape:[d_inner,]
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )

        # 分别将 dt_projs 中四个矩阵分解初始化函数的权重和偏置堆叠成参数张量。
        self.dt_projs_weight3 = nn.Parameter(torch.stack([t.weight for t in self.dt_projs3], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias3 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs3], dim=0))  # (K=4, inner)
        del self.dt_projs3

        # 分别初始化了用于选择性扫描的参数
        self.A_logs3 = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N),生成一个[copies * d_inner * d_state,] 的矩阵
        self.Ds3 = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, N),生成一个[copies * d_inner,] 的矩阵

        # 选择性扫描函数。
        self.selective_scan3 = selective_scan_fn 

        # 一个层归一化层，用于规范化输出。
        self.out_norm3 = nn.LayerNorm(self.d_inner)

        # 一个线性层，将内部特征投影回输出特征维度。
        self.out_proj3 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # 一个Dropout层，用于防止过拟合。
        self.dropout3 = nn.Dropout(dropout) if dropout > 0. else None


###---------------------------------------------------------------------------------------------------------------------------------####

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        

        # 矩阵分解初始化
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # 初始化 dt 偏置，使得 F.softplus(dt_bias) 位于 dt_min 和 dt_max 之间。
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor) # 生成shape为[d_inner,]的张量
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # [d_inner,]
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj
    


    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device), # 生成1到d_state（包含首尾）的等差数列张量，形状为[d_state,] 
            "n -> d n",  # repeat操作，形状为[d_inner,d_state] 
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies) # repeat操作，形状为[copies,d_inner,d_state] 
            if merge:
                A_log = A_log.flatten(0, 1) # [copies * d_inner * d_state,] 
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    # forward_core方法执行了模型的核心运算，包括数据的多视角合并处理和多个特殊投影的应用。
    # 这部分利用了特有的时间步长项目（dt_projs）和自定义的扫描函数（selective_scan）进行复杂的数据转换和处理，最终产生了四个主要的输出。
    def forward_core(self, x: torch.Tensor): # 输入B，C，H，W  输出B，C，H，W
        # 变量解释：
        # 这个函数接受一个torch.Tensor类型的输入x,其形状为(batch_size, channels, height, width)。
        # B, C, H, W = x.shape: 这行代码从输入张量x的形状中提取了四个维度的值，分别是批大小（batch_size）、通道数（channels）、高度（height）和宽度（width）。
        # L = H * W: 这行代码计算了输入张量中空间维度的元素数量，即图片的像素数量。
        # K = 4: 这个变量定义了一个值为4的常数K
        rgb = x[0]
        ir = x[1]
        R1 = x[2]
        Just = x[3]
        Just = torch.tensor(Just,dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).cuda()


        B, C, H, W = rgb.shape
        L = H * W
        K = 4
# RGB
###---------------------------------------------------------------------------------------------------------------------------------####
        # 数据处理：
        # x_hwwh = ...: 这行代码对输入张量进行了一系列操作，将其重塑为一个新的张量x_hwwh，形状为(batch_size, 2, K, L)。
        # 具体操作包括将原始张量展平为(batch_size, channels, L)的形状，然后将其沿着空间维度进行堆叠和转置，以得到期望的形状。
        x_hwwh1 = torch.stack([rgb.view(B, -1, L), torch.transpose(rgb, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # xs = torch.cat([...], dim=1): 这行代码将x_hwwh和其在空间维度上的翻转拼接在一起，形成一个新的张量xs，其形状为(batch_size, 4, 2, L)。
        xs1 = torch.cat([x_hwwh1, torch.flip(x_hwwh1, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        # 张量运算：
        # x_dbl = torch.einsum(...): 这行代码使用torch.einsum函数执行张量乘法，将xs与self.x_proj_weight相乘，并对其中的维度进行合并和变换。结果被分割成三个张量dts、Bs和Cs。
        x_dbl1 = torch.einsum("b k d l, k c d -> b k c l", xs1.view(B, K, -1, L), self.x_proj_weight1) # 张量乘法，输出[]
        dts1, Bs1, Cs1 = torch.split(x_dbl1, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        # dts = torch.einsum(...): 这行代码类似于上一行，将dts张量再次与self.dt_projs_weight相乘，并对其进行合并和变换。
        dts1 = torch.einsum("b k r l, k d r -> b k d l", dts1.view(B, K, -1, L), self.dt_projs_weight1)
        
        # 数据类型转换：
        # xs = xs.float().view(...), dts = dts.contiguous().float().view(...), Bs = Bs.float().view(...), Cs = Cs.float().view(...): 
        # 这几行代码将一些张量转换为浮点型，并重新调整其形状，是为了与后续的操作兼容。
        xs1 = xs1.float().view(B, -1, L)
        dts1 = dts1.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs1 = Bs1.float().view(B, K, -1, L)
        Cs1 = Cs1.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds1 = self.Ds1.float().view(-1)
        As1 = -torch.exp(self.A_logs1.float()).view(-1, self.d_state)
        dt_projs_bias1 = self.dt_projs_bias1.float().view(-1) # (k * d)
        # print(As)


###---------------------------------------------------------------------------------------------------------------------------------####
        
# IR      
###---------------------------------------------------------------------------------------------------------------------------------####
        # 数据处理：
        # x_hwwh = ...: 这行代码对输入张量进行了一系列操作，将其重塑为一个新的张量x_hwwh，形状为(batch_size, 2, K, L)。
        # 具体操作包括将原始张量展平为(batch_size, channels, L)的形状，然后将其沿着空间维度进行堆叠和转置，以得到期望的形状。
        x_hwwh2 = torch.stack([ir.view(B, -1, L), torch.transpose(ir, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # xs = torch.cat([...], dim=1): 这行代码将x_hwwh和其在空间维度上的翻转拼接在一起，形成一个新的张量xs，其形状为(batch_size, 4, 2, L)。
        xs2 = torch.cat([x_hwwh2, torch.flip(x_hwwh2, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        # 张量运算：
        # x_dbl = torch.einsum(...): 这行代码使用torch.einsum函数执行张量乘法，将xs与self.x_proj_weight相乘，并对其中的维度进行合并和变换。结果被分割成三个张量dts、Bs和Cs。
        x_dbl2 = torch.einsum("b k d l, k c d -> b k c l", xs2.view(B, K, -1, L), self.x_proj_weight2) # 张量乘法，输出[]
        dts2, Bs2, Cs2 = torch.split(x_dbl2, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        # dts = torch.einsum(...): 这行代码类似于上一行，将dts张量再次与self.dt_projs_weight相乘，并对其进行合并和变换。
        dts2 = torch.einsum("b k r l, k d r -> b k d l", dts2.view(B, K, -1, L), self.dt_projs_weight2)
        
        # 数据类型转换：
        # xs = xs.float().view(...), dts = dts.contiguous().float().view(...), Bs = Bs.float().view(...), Cs = Cs.float().view(...): 
        # 这几行代码将一些张量转换为浮点型，并重新调整其形状，是为了与后续的操作兼容。
        xs2 = xs2.float().view(B, -1, L)
        dts2 = dts2.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs2 = Bs2.float().view(B, K, -1, L)
        Cs2 = Cs2.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds2 = self.Ds2.float().view(-1)
        As2 = -torch.exp(self.A_logs2.float()).view(-1, self.d_state)
        dt_projs_bias2 = self.dt_projs_bias2.float().view(-1) # (k * d)
        # print(As)


###---------------------------------------------------------------------------------------------------------------------------------####

# R1
###---------------------------------------------------------------------------------------------------------------------------------####
        # 数据处理：
        # x_hwwh = ...: 这行代码对输入张量进行了一系列操作，将其重塑为一个新的张量x_hwwh，形状为(batch_size, 2, K, L)。
        # 具体操作包括将原始张量展平为(batch_size, channels, L)的形状，然后将其沿着空间维度进行堆叠和转置，以得到期望的形状。
        x_hwwh3 = torch.stack([R1.view(B, -1, L), torch.transpose(R1, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # xs = torch.cat([...], dim=1): 这行代码将x_hwwh和其在空间维度上的翻转拼接在一起，形成一个新的张量xs，其形状为(batch_size, 4, 2, L)。
        xs3 = torch.cat([x_hwwh3, torch.flip(x_hwwh3, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        # 张量运算：
        # x_dbl = torch.einsum(...): 这行代码使用torch.einsum函数执行张量乘法，将xs与self.x_proj_weight相乘，并对其中的维度进行合并和变换。结果被分割成三个张量dts、Bs和Cs。
        x_dbl3 = torch.einsum("b k d l, k c d -> b k c l", xs3.view(B, K, -1, L), self.x_proj_weight3) # 张量乘法，输出[]
        dts3, Bs3, Cs3 = torch.split(x_dbl3, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        # dts = torch.einsum(...): 这行代码类似于上一行，将dts张量再次与self.dt_projs_weight相乘，并对其进行合并和变换。
        dts3 = torch.einsum("b k r l, k d r -> b k d l", dts3.view(B, K, -1, L), self.dt_projs_weight3)
        
        # 数据类型转换：
        # xs = xs.float().view(...), dts = dts.contiguous().float().view(...), Bs = Bs.float().view(...), Cs = Cs.float().view(...): 
        # 这几行代码将一些张量转换为浮点型，并重新调整其形状，是为了与后续的操作兼容。
        xs3 = xs3.float().view(B, -1, L)
        dts3 = dts3.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs3 = Bs3.float().view(B, K, -1, L)
        Cs3 = Cs3.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds3 = self.Ds3.float().view(-1)
        As3 = -torch.exp(self.A_logs3.float()).view(-1, self.d_state)
        dt_projs_bias3 = self.dt_projs_bias3.float().view(-1) # (k * d)
        # print(As)

# RGB
###---------------------------------------------------------------------------------------------------------------------------------####
        
        # out_y = self.selective_scan(...): 这行代码调用了一个自定义函数selective_scan，传递了一系列张量作为参数，并返回一个张量out_y。  dts1*Just+dts2+dts3
        out_y1 = self.selective_scan1(
            xs1, dts1*Just+dts3,
            As1, Bs1, Cs1+Cs2+Cs3, Ds1, z=None,
            delta_bias=dt_projs_bias1,

            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y1.dtype == torch.float

        # inv_y = ..., wh_y = ..., invwh_y = ...: 这几行代码对out_y进行了一些处理，将其切片、翻转和重塑为期望的形状，并将结果分别赋给了inv_y、wh_y和invwh_y。
        inv_y1 = torch.flip(out_y1[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y1 = torch.transpose(out_y1[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y1 = torch.transpose(inv_y1[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        rgb_shu = (out_y1[:, 0], inv_y1[:, 0], wh_y1, invwh_y1)
####        
        # return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y: 这行代码返回了四个张量作为函数的输出。
###---------------------------------------------------------------------------------------------------------------------------------####

# IR
###---------------------------------------------------------------------------------------------------------------------------------####

        # out_y = self.selective_scan(...): 这行代码调用了一个自定义函数selective_scan，传递了一系列张量作为参数，并返回一个张量out_y。
        # VSS2
        out_y2 = self.selective_scan2(
            xs2, dts2,
            As2, Bs2, Cs1+Cs2+Cs3, Ds2, z=None,
            delta_bias=dt_projs_bias2,

            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y2.dtype == torch.float

        # inv_y = ..., wh_y = ..., invwh_y = ...: 这几行代码对out_y进行了一些处理，将其切片、翻转和重塑为期望的形状，并将结果分别赋给了inv_y、wh_y和invwh_y。
        inv_y2 = torch.flip(out_y2[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y2 = torch.transpose(out_y2[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y2 = torch.transpose(inv_y2[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
####        
        # return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y: 这行代码返回了四个张量作为函数的输出。
        ir_shu = (out_y2[:, 0], inv_y2[:, 0], wh_y2, invwh_y2)
###---------------------------------------------------------------------------------------------------------------------------------####

# R1
###---------------------------------------------------------------------------------------------------------------------------------####
        
        # out_y = self.selective_scan(...): 这行代码调用了一个自定义函数selective_scan，传递了一系列张量作为参数，并返回一个张量out_y。
        out_y3 = self.selective_scan3(
            xs3, dts3,
            As3, Bs3, Cs1+Cs2+Cs3, Ds3, z=None,
            delta_bias=dt_projs_bias3,

            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y3.dtype == torch.float

        # inv_y = ..., wh_y = ..., invwh_y = ...: 这几行代码对out_y进行了一些处理，将其切片、翻转和重塑为期望的形状，并将结果分别赋给了inv_y、wh_y和invwh_y。
        inv_y3 = torch.flip(out_y3[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y3 = torch.transpose(out_y3[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y3 = torch.transpose(inv_y3[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        r1_shu = (out_y3[:, 0], inv_y3[:, 0], wh_y3, invwh_y3)
####        
        # return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y: 这行代码返回了四个张量作为函数的输出。
###---------------------------------------------------------------------------------------------------------------------------------####

        result = (rgb_shu, ir_shu, r1_shu)

        return result

        # return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    

    # 然后，在forward中，这四个输出通过特定的变换和组合，最后通过归一化、激活函数和最后的out_proj层，生成最终的输出。
    def forward(self, x: torch.Tensor, **kwargs):

###---------------------------------------------------------------------------------------------------------------------------------####
        if type(x[0]) == tuple :
            rgb = x[0][0]  # rgb_fea (tensor): dim:(B, C, H, W)
            ir = x[0][1]   # ir_fea (tensor): dim:(B, C, H, W) 
            R1 = x[1][2]
            L1 = x[1][1]
            R1 = self.conv1(R1)  # 调整通道数到128
            B,C,H,W = rgb.shape
            # 使用双线性插值将空间维度调整为 (80, 80)
            R1 = F.interpolate(R1, size=(H, W), mode='bilinear', align_corners=False)
            L = F.interpolate(L1, size=(H, W), mode='bilinear', align_corners=False)

            # R1 = self.pool1(R1)  # 下采样空间尺寸到80x80 
            Just = []
            for i in range(B):           
                L[i][(L[i])<self.duckvalue] = 0.1
                L[i][(L[i])>=self.duckvalue] = 0
            

                if (L[i]*10).sum() / (H*W) > self.duckness:
                    Just.append(0)
                else:
                    Just.append(1)
        
        elif type(x[0]) == list :
            rgb = x[0][0]  # rgb_fea (tensor): dim:(B, C, H, W)
            ir = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
            R1 = x[2][2]
            L1 = x[2][1]
            R1 = self.conv1(R1)  # 调整通道数到128
            B,C,H,W = rgb.shape
            # 使用双线性插值将空间维度调整为 (80, 80)
            R1 = F.interpolate(R1, size=(H, W), mode='bilinear', align_corners=False)
            L = F.interpolate(L1, size=(H, W), mode='bilinear', align_corners=False)

            # R1 = self.pool1(R1)  # 下采样空间尺寸到80x80 
            Just = []
            for i in range(B):           
                L[i][(L[i])<self.duckvalue] = 0.1
                L[i][(L[i])>=self.duckvalue] = 0
            

                if (L[i]*10).sum() / (H*W) > self.duckness:
                    Just.append(0)
                else:
                    Just.append(1)

        elif type(x[1]) == list :
            rgb = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
            ir = x[1][0]   # ir_fea (tensor): dim:(B, C, H, W) 
            R1 = x[2][2]
            L1 = x[2][1]
            R1 = self.conv1(R1)  # 调整通道数到128
            B,C,H,W = rgb.shape
            # 使用双线性插值将空间维度调整为 (80, 80)
            R1 = F.interpolate(R1, size=(H, W), mode='bilinear', align_corners=False)
            L = F.interpolate(L1, size=(H, W), mode='bilinear', align_corners=False)

            # R1 = self.pool1(R1)  # 下采样空间尺寸到80x80 
            Just = []
            for i in range(B):           
                L[i][(L[i])<self.duckvalue] = 0.1
                L[i][(L[i])>=self.duckvalue] = 0
            

                if (L[i]*10).sum() / (H*W) > self.duckness:
                    Just.append(0)
                else:
                    Just.append(1)
                   
        else:
            rgb = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
            ir = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
            R1 = x[2][2]
            L1 = x[2][1]
            R1 = self.conv1(R1)  # 调整通道数到128
            B,C,H,W = rgb.shape
            # 使用双线性插值将空间维度调整为 (80, 80)
            R1 = F.interpolate(R1, size=(H, W), mode='bilinear', align_corners=False)
            L = F.interpolate(L1, size=(H, W), mode='bilinear', align_corners=False)

            # R1 = self.pool1(R1)  # 下采样空间尺寸到80x80 
            Just = []
            for i in range(B):           
                L[i][(L[i])<self.duckvalue] = 0.1
                L[i][(L[i])>=self.duckvalue] = 0
            

                if (L[i]*10).sum() / (H*W) > self.duckness:
                    Just.append(0)
                else:
                    Just.append(1)
# RGB               
###---------------------------------------------------------------------------------------------------------------------------------####
        rgb1 = rgb.permute(0,2,3,1)
        B1, H1, W1, C1 = rgb1.shape

        rgb1z1 = self.in_proj1(rgb1)
        rgb1, z1 = rgb1z1.chunk(2, dim=-1)

        rgb1 = rgb1.permute(0, 3, 1, 2).contiguous()
        rgb1 = self.act(self.conv2d1(rgb1))
###---------------------------------------------------------------------------------------------------------------------------------####
        ir1 = ir.permute(0,2,3,1)
        B2, H2, W2, C2 = ir1.shape

        ir1z2 = self.in_proj2(ir1)
        ir1, z2 = ir1z2.chunk(2, dim=-1)

        ir1 = ir1.permute(0, 3, 1, 2).contiguous()
        ir1 = self.act(self.conv2d2(ir1))
###---------------------------------------------------------------------------------------------------------------------------------####
        r1 = R1.permute(0,2,3,1)
        B3, H3, W3, C3 = r1.shape

        r1z3 = self.in_proj3(r1)
        r1, z3 = r1z3.chunk(2, dim=-1)

        r1 = r1.permute(0, 3, 1, 2).contiguous()
        r1 = self.act(self.conv2d3(r1))
###---------------------------------------------------------------------------------------------------------------------------------####
        fea = (rgb1, ir1, r1, Just)

        rgb_fea ,ir_fea, r1_fea = self.forward_core(fea)
     

        rgb_y1, rgb_y2, rgb_y3, rgb_y4 = rgb_fea
        ir_y1, ir_y2, ir_y3, ir_y4 = ir_fea
        r1_y1, r1_y2, r1_y3, r1_y4 = r1_fea
        
        assert rgb_y1.dtype == torch.float32
        y_rgb = rgb_y1 + rgb_y2 + rgb_y3 + rgb_y4
        y_rgb = torch.transpose(y_rgb, dim0=1, dim1=2).contiguous().view(B1, H1, W1, -1)
        y_rgb = self.out_norm1(y_rgb)
        y_rgb = y_rgb * F.silu(z1)
        rgb_out = self.out_proj1(y_rgb)
        # if self.dropout is not None:
        #     out = self.dropout(out)
        rgb_out = rgb_out.permute(0,3,1,2)

        assert ir_y1.dtype == torch.float32
        y_ir = ir_y1 + ir_y2 + ir_y3 + ir_y4
        y_ir = torch.transpose(y_ir, dim0=1, dim1=2).contiguous().view(B2, H2, W2, -1)
        y_ir = self.out_norm2(y_ir)
        y_ir = y_ir * F.silu(z2)
        ir_out = self.out_proj2(y_ir)
        # if self.dropout is not None:
        #     out = self.dropout(out)
        ir_out = ir_out.permute(0,3,1,2)
        
        assert r1_y1.dtype == torch.float32
        y_r1 = r1_y1 + r1_y2 + r1_y3 + r1_y4
        y_r1 = torch.transpose(y_r1, dim0=1, dim1=2).contiguous().view(B3, H3, W3, -1)
        y_r1 = self.out_norm3(y_r1)
        y_r1 = y_r1 * F.silu(z3)
        r1_out = self.out_proj3(y_r1)
        # if self.dropout is not None:
        #     out = self.dropout(out)
        r1_out = r1_out.permute(0,3,1,2)

        rgb_out = rgb_out + r1_out
        ir_out = ir_out + r1_out

        # out = torch.concat([rgb_out,ir_out] , dim = 1)

        return rgb_out, ir_out
        # return out



class VSS4(nn.Module): # 四方向选择性扫描操作
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model//2
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank # math.ceil 向上取整

        # 用于调整通道数的卷积层
        # self.conv1 = nn.Conv2d(3, self.d_model, kernel_size=1)

        # self.duckness = nn.Parameter(torch.tensor(0.338)) #  0.44 0.3380  0.2  0.6802             
        # self.duckvalue = nn.Parameter(torch.tensor(0.7848))  # 0.6124 0.6082 0.6802 0.6682 0.8049 0.7848 0.7847 0.6340 1.2734 0.8758 0.7736 1.0531 



        self.act = nn.SiLU()

# RGB
###---------------------------------------------------------------------------------------------------------------------------------####
        
        # in_proj：一个线性层，将输入特征投影到内部特征维度的两倍，用于输入的预处理。
        self.in_proj1 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs) # 128 256 
        self.conv2d1 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
			padding_mode='reflect',
            **factory_kwargs,
        )
        # self.act1 = nn.SiLU()

        # 四个线性层组成的元组，用于将内部特征投影到矩阵分解的维度。
        self.x_proj1 = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )

        # 将 x_proj 中四个线性层的权重堆叠成一个参数张量。
        self.x_proj_weight1 = nn.Parameter(torch.stack([t.weight for t in self.x_proj1], dim=0))  # (K=4, N, inner)
        del self.x_proj1

        # 四个矩阵分解初始化函数组成的元组，用于初始化矩阵分解。
        self.dt_projs1 = ( # dt_init返回经过初始化后的linear(dt_rank,d_inner),weight.shape:[d_inner,dt_rank],bias.shape:[d_inner,]
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )

        # 分别将 dt_projs 中四个矩阵分解初始化函数的权重和偏置堆叠成参数张量。
        self.dt_projs_weight1 = nn.Parameter(torch.stack([t.weight for t in self.dt_projs1], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias1 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs1], dim=0))  # (K=4, inner)
        del self.dt_projs1

        # 分别初始化了用于选择性扫描的参数
        self.A_logs1 = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N),生成一个[copies * d_inner * d_state,] 的矩阵
        self.Ds1 = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, N),生成一个[copies * d_inner,] 的矩阵

        # 选择性扫描函数。
        self.selective_scan1 = selective_scan_fn 

        # 一个层归一化层，用于规范化输出。
        self.out_norm1 = nn.LayerNorm(self.d_inner)

        # 一个线性层，将内部特征投影回输出特征维度。
        self.out_proj1 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # 一个Dropout层，用于防止过拟合。
        self.dropout1 = nn.Dropout(dropout) if dropout > 0. else None


###---------------------------------------------------------------------------------------------------------------------------------####

# IR
###---------------------------------------------------------------------------------------------------------------------------------####
        
        # in_proj：一个线性层，将输入特征投影到内部特征维度的两倍，用于输入的预处理。
        self.in_proj2 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs) # 128 256 
        self.conv2d2 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
			padding_mode='reflect',
            **factory_kwargs,
        )
        # self.act1 = nn.SiLU()

        # 四个线性层组成的元组，用于将内部特征投影到矩阵分解的维度。
        self.x_proj2 = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )

        # 将 x_proj 中四个线性层的权重堆叠成一个参数张量。
        self.x_proj_weight2 = nn.Parameter(torch.stack([t.weight for t in self.x_proj2], dim=0))  # (K=4, N, inner)
        del self.x_proj2

        # 四个矩阵分解初始化函数组成的元组，用于初始化矩阵分解。
        self.dt_projs2 = ( # dt_init返回经过初始化后的linear(dt_rank,d_inner),weight.shape:[d_inner,dt_rank],bias.shape:[d_inner,]
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )

        # 分别将 dt_projs 中四个矩阵分解初始化函数的权重和偏置堆叠成参数张量。
        self.dt_projs_weight2 = nn.Parameter(torch.stack([t.weight for t in self.dt_projs2], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias2 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs2], dim=0))  # (K=4, inner)
        del self.dt_projs2

        # 分别初始化了用于选择性扫描的参数
        self.A_logs2 = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N),生成一个[copies * d_inner * d_state,] 的矩阵
        self.Ds2 = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, N),生成一个[copies * d_inner,] 的矩阵

        # 选择性扫描函数。
        self.selective_scan2 = selective_scan_fn 

        # 一个层归一化层，用于规范化输出。
        self.out_norm2 = nn.LayerNorm(self.d_inner)

        # 一个线性层，将内部特征投影回输出特征维度。
        self.out_proj2 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # 一个Dropout层，用于防止过拟合。
        self.dropout2 = nn.Dropout(dropout) if dropout > 0. else None
###---------------------------------------------------------------------------------------------------------------------------------####




###---------------------------------------------------------------------------------------------------------------------------------####

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        

        # 矩阵分解初始化
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # 初始化 dt 偏置，使得 F.softplus(dt_bias) 位于 dt_min 和 dt_max 之间。
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor) # 生成shape为[d_inner,]的张量
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # [d_inner,]
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj
    


    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device), # 生成1到d_state（包含首尾）的等差数列张量，形状为[d_state,] 
            "n -> d n",  # repeat操作，形状为[d_inner,d_state] 
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies) # repeat操作，形状为[copies,d_inner,d_state] 
            if merge:
                A_log = A_log.flatten(0, 1) # [copies * d_inner * d_state,] 
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    # forward_core方法执行了模型的核心运算，包括数据的多视角合并处理和多个特殊投影的应用。
    # 这部分利用了特有的时间步长项目（dt_projs）和自定义的扫描函数（selective_scan）进行复杂的数据转换和处理，最终产生了四个主要的输出。
    def forward_core(self, x: torch.Tensor): # 输入B，C，H，W  输出B，C，H，W
        # 变量解释：
        # 这个函数接受一个torch.Tensor类型的输入x,其形状为(batch_size, channels, height, width)。
        # B, C, H, W = x.shape: 这行代码从输入张量x的形状中提取了四个维度的值，分别是批大小（batch_size）、通道数（channels）、高度（height）和宽度（width）。
        # L = H * W: 这行代码计算了输入张量中空间维度的元素数量，即图片的像素数量。
        # K = 4: 这个变量定义了一个值为4的常数K
        rgb = x[0]
        ir = x[1]
        # R1 = x[2]
        Just = x[2]
        Just = torch.tensor(Just,dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).cuda()


        B, C, H, W = rgb.shape
        L = H * W
        K = 4
# RGB
###---------------------------------------------------------------------------------------------------------------------------------####
        # 数据处理：
        # x_hwwh = ...: 这行代码对输入张量进行了一系列操作，将其重塑为一个新的张量x_hwwh，形状为(batch_size, 2, K, L)。
        # 具体操作包括将原始张量展平为(batch_size, channels, L)的形状，然后将其沿着空间维度进行堆叠和转置，以得到期望的形状。
        x_hwwh1 = torch.stack([rgb.view(B, -1, L), torch.transpose(rgb, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # xs = torch.cat([...], dim=1): 这行代码将x_hwwh和其在空间维度上的翻转拼接在一起，形成一个新的张量xs，其形状为(batch_size, 4, 2, L)。
        xs1 = torch.cat([x_hwwh1, torch.flip(x_hwwh1, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        # 张量运算：
        # x_dbl = torch.einsum(...): 这行代码使用torch.einsum函数执行张量乘法，将xs与self.x_proj_weight相乘，并对其中的维度进行合并和变换。结果被分割成三个张量dts、Bs和Cs。
        x_dbl1 = torch.einsum("b k d l, k c d -> b k c l", xs1.view(B, K, -1, L), self.x_proj_weight1) # 张量乘法，输出[]
        dts1, Bs1, Cs1 = torch.split(x_dbl1, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        # dts = torch.einsum(...): 这行代码类似于上一行，将dts张量再次与self.dt_projs_weight相乘，并对其进行合并和变换。
        dts1 = torch.einsum("b k r l, k d r -> b k d l", dts1.view(B, K, -1, L), self.dt_projs_weight1)
        
        # 数据类型转换：
        # xs = xs.float().view(...), dts = dts.contiguous().float().view(...), Bs = Bs.float().view(...), Cs = Cs.float().view(...): 
        # 这几行代码将一些张量转换为浮点型，并重新调整其形状，是为了与后续的操作兼容。
        xs1 = xs1.float().view(B, -1, L)
        dts1 = dts1.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs1 = Bs1.float().view(B, K, -1, L)
        Cs1 = Cs1.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds1 = self.Ds1.float().view(-1)
        As1 = -torch.exp(self.A_logs1.float()).view(-1, self.d_state)
        dt_projs_bias1 = self.dt_projs_bias1.float().view(-1) # (k * d)
        # print(As)


###---------------------------------------------------------------------------------------------------------------------------------####
        
# IR      
###---------------------------------------------------------------------------------------------------------------------------------####
        # 数据处理：
        # x_hwwh = ...: 这行代码对输入张量进行了一系列操作，将其重塑为一个新的张量x_hwwh，形状为(batch_size, 2, K, L)。
        # 具体操作包括将原始张量展平为(batch_size, channels, L)的形状，然后将其沿着空间维度进行堆叠和转置，以得到期望的形状。
        x_hwwh2 = torch.stack([ir.view(B, -1, L), torch.transpose(ir, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # xs = torch.cat([...], dim=1): 这行代码将x_hwwh和其在空间维度上的翻转拼接在一起，形成一个新的张量xs，其形状为(batch_size, 4, 2, L)。
        xs2 = torch.cat([x_hwwh2, torch.flip(x_hwwh2, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        # 张量运算：
        # x_dbl = torch.einsum(...): 这行代码使用torch.einsum函数执行张量乘法，将xs与self.x_proj_weight相乘，并对其中的维度进行合并和变换。结果被分割成三个张量dts、Bs和Cs。
        x_dbl2 = torch.einsum("b k d l, k c d -> b k c l", xs2.view(B, K, -1, L), self.x_proj_weight2) # 张量乘法，输出[]
        dts2, Bs2, Cs2 = torch.split(x_dbl2, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        # dts = torch.einsum(...): 这行代码类似于上一行，将dts张量再次与self.dt_projs_weight相乘，并对其进行合并和变换。
        dts2 = torch.einsum("b k r l, k d r -> b k d l", dts2.view(B, K, -1, L), self.dt_projs_weight2)
        
        # 数据类型转换：
        # xs = xs.float().view(...), dts = dts.contiguous().float().view(...), Bs = Bs.float().view(...), Cs = Cs.float().view(...): 
        # 这几行代码将一些张量转换为浮点型，并重新调整其形状，是为了与后续的操作兼容。
        xs2 = xs2.float().view(B, -1, L)
        dts2 = dts2.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs2 = Bs2.float().view(B, K, -1, L)
        Cs2 = Cs2.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds2 = self.Ds2.float().view(-1)
        As2 = -torch.exp(self.A_logs2.float()).view(-1, self.d_state)
        dt_projs_bias2 = self.dt_projs_bias2.float().view(-1) # (k * d)
        # print(As)


###---------------------------------------------------------------------------------------------------------------------------------####



# RGB
###---------------------------------------------------------------------------------------------------------------------------------####
        
        # out_y = self.selective_scan(...): 这行代码调用了一个自定义函数selective_scan，传递了一系列张量作为参数，并返回一个张量out_y。  dts1*Just+dts2+dts3 dts1*Just+dts3
        out_y1 = self.selective_scan1(
            xs1, dts1,
            As1, Bs1, Cs1+Cs2, Ds1, z=None,
            delta_bias=dt_projs_bias1,

            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y1.dtype == torch.float

        # inv_y = ..., wh_y = ..., invwh_y = ...: 这几行代码对out_y进行了一些处理，将其切片、翻转和重塑为期望的形状，并将结果分别赋给了inv_y、wh_y和invwh_y。
        inv_y1 = torch.flip(out_y1[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y1 = torch.transpose(out_y1[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y1 = torch.transpose(inv_y1[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        rgb_shu = (out_y1[:, 0], inv_y1[:, 0], wh_y1, invwh_y1)
####        
        # return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y: 这行代码返回了四个张量作为函数的输出。
###---------------------------------------------------------------------------------------------------------------------------------####

# IR
###---------------------------------------------------------------------------------------------------------------------------------####

        # out_y = self.selective_scan(...): 这行代码调用了一个自定义函数selective_scan，传递了一系列张量作为参数，并返回一个张量out_y。
        # VSS2
        out_y2 = self.selective_scan2(
            xs2, dts2,
            As2, Bs2, Cs1+Cs2, Ds2, z=None,
            delta_bias=dt_projs_bias2,

            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y2.dtype == torch.float

        # inv_y = ..., wh_y = ..., invwh_y = ...: 这几行代码对out_y进行了一些处理，将其切片、翻转和重塑为期望的形状，并将结果分别赋给了inv_y、wh_y和invwh_y。
        inv_y2 = torch.flip(out_y2[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y2 = torch.transpose(out_y2[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y2 = torch.transpose(inv_y2[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
####        
        # return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y: 这行代码返回了四个张量作为函数的输出。
        ir_shu = (out_y2[:, 0], inv_y2[:, 0], wh_y2, invwh_y2)
###---------------------------------------------------------------------------------------------------------------------------------###

###---------------------------------------------------------------------------------------------------------------------------------####

        result = (rgb_shu, ir_shu)

        return result

        # return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    

    # 然后，在forward中，这四个输出通过特定的变换和组合，最后通过归一化、激活函数和最后的out_proj层，生成最终的输出。
    def forward(self, x: torch.Tensor, Just,**kwargs):

###---------------------------------------------------------------------------------------------------------------------------------####
        B, H, W, C = x.shape
        rgb1 = x[:,...,:C//2]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir1 = x[:,...,C//2:(C//2)*2]   # ir_fea (tensor): dim:(B, C, H, W) 
            
# RGB               
###---------------------------------------------------------------------------------------------------------------------------------####
        # rgb1 = rgb.permute(0,2,3,1)
        B1, H1, W1, C1 = rgb1.shape
        
        rgb1z1 = self.in_proj1(rgb1) # Linear
        rgb1, z1 = rgb1z1.chunk(2, dim=-1)

        rgb1 = rgb1.permute(0, 3, 1, 2).contiguous()
        rgb1 = self.act(self.conv2d1(rgb1))
###---------------------------------------------------------------------------------------------------------------------------------####
        # ir1 = ir.permute(0,2,3,1)
        B2, H2, W2, C2 = ir1.shape

        ir1z2 = self.in_proj2(ir1)
        ir1, z2 = ir1z2.chunk(2, dim=-1)

        ir1 = ir1.permute(0, 3, 1, 2).contiguous()
        ir1 = self.act(self.conv2d2(ir1))
###---------------------------------------------------------------------------------------------------------------------------------####
        fea = (rgb1, ir1, Just)

        rgb_fea ,ir_fea = self.forward_core(fea)
     

        rgb_y1, rgb_y2, rgb_y3, rgb_y4 = rgb_fea
        ir_y1, ir_y2, ir_y3, ir_y4 = ir_fea
        # r1_y1, r1_y2, r1_y3, r1_y4 = r1_fea
        
        assert rgb_y1.dtype == torch.float32
        y_rgb1 = rgb_y1 + rgb_y2 + rgb_y3 + rgb_y4

        assert ir_y1.dtype == torch.float32
        y_ir1 = ir_y1 + ir_y2 + ir_y3 + ir_y4

    

        y_rgb = y_rgb1 
        y_ir = y_ir1


        y_rgb = torch.transpose(y_rgb, dim0=1, dim1=2).contiguous().view(B1, H1, W1, -1)
        y_rgb = self.out_norm1(y_rgb)
        y_rgb = y_rgb * F.silu(z1)
        rgb_out = self.out_proj1(y_rgb)
        if self.dropout1 is not None:
            out = self.dropout1(out)
        # rgb_out = rgb_out.permute(0,3,1,2)



        y_ir = torch.transpose(y_ir, dim0=1, dim1=2).contiguous().view(B2, H2, W2, -1)
        y_ir = self.out_norm2(y_ir)
        y_ir = y_ir * F.silu(z2)
        ir_out = self.out_proj2(y_ir)
        if self.dropout2 is not None:
            out = self.dropout2(out)
        # ir_out = ir_out.permute(0,3,1,2)


        out = torch.concat([rgb_out,ir_out] , dim = -1)

        # return rgb_out, ir_out
        return out





class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    

class MF(nn.Module):
    def __init__(self, c1, c2, reduction=16):
        super(MF, self).__init__()
        self.mask_map_r = nn.Conv2d(c1//2, 1, 1, 1, 0, bias=True)
        self.mask_map_i = nn.Conv2d(c1//2, 1, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.bottleneck1 = nn.Conv2d(c1//2, c2//2, 3, 1, 1, bias=False)
        self.bottleneck2 = nn.Conv2d(c1//2, c2//2, 3, 1, 1, bias=False)
        self.se = SE_Block(c2, reduction)

    def forward(self, x):
        x_left_ori,x_right_ori = x[:, :3, :, :], x[:, 3:, :, :]
        x_left = x_left_ori * 0.5
        x_right = x_right_ori * 0.5

        x_mask_left = torch.mul(self.mask_map_r(x_left), x_left)
        x_mask_right = torch.mul(self.mask_map_i(x_right), x_right)

        out_IR = self.bottleneck1(x_mask_right + x_right_ori)
        out_RGB = self.bottleneck2(x_mask_left + x_left_ori)  # RGB
        out = self.se(torch.cat([out_RGB, out_IR], 1))

        return out

class IN(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
    
class IN1(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = x[2] # IR
        x2 = x[3] # RGB
        return x1,x2

class IN2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        R = x[2]
        x1 = x[3] # IR
        x2 = x[4] # RGB
        return x1,x2,R

class IN3(nn.Module):
    def __init__(self, index):
        super().__init__()
        # self.index = index

    def forward(self, x):
        R = x[2]
        return R

        # if self.index == 0:
        #     rgb = x[0] # RGB
        #     return rgb
        # elif self.index == 1:
        #     ir = x[1] # IR
        #     return ir

class IN4(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        ir = x[1] # IR
        return ir


class Multiin(nn.Module):  # stereo attention block
    def __init__(self, out=1):
        super().__init__()
        self.out = out

    def forward(self, x):
        x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        if self.out == 1:
            x = x1
        else:
            x = x2
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        # 写法二,亦可使用顺序容器
        # self.sharedMLP = nn.Sequential(
        # nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
        # nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out
        

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x1 = torch.mean(x)
        x2 = torch.max(x)
        x= x1+x2
        x = self.sigmoid(x)
        return x
    
class Concat3(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, ratio=16, kernel_size=7,dimension=1):
        super().__init__()
        self.d = dimension#沿着哪个维度进行拼接
        self.spatial_attention = SpatialAttention(7)
        self.channel_attention = ChannelAttention(c1, ratio)
    def forward(self, x):
        weight1 = self.spatial_attention(x[1]) # ir
        weight2 = self.spatial_attention(x[0]) # rgb
        ir_weight = (weight1/weight2)
        x[0]=ir_weight*x[0]
        x[1]=x[1]*(2-ir_weight)
        x = torch.cat((x[0],x[1]), self.d)
        X=self.channel_attention(x)
        return x

class Add(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        return torch.add(x[0], x[1])

class Add5(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add5, self).__init__()
        self.arg = arg

    def forward(self, x):
        return torch.add(x[0], x[1])

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(Conv(c1, c2, k=7, s=2, p=3, act=True),
                                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)
    

class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):

        if type(x[0]) == tuple :
            if type(x[1]) == list :
                if self.index == 0:
                    return torch.add(x[0][0], x[1][0][0])
                elif self.index == 1:
                    return torch.add(x[0][1], x[1][0][1])
            elif type(x[1]) == tuple :
                if self.index == 0:
                    return torch.add(x[0][0], x[1][0])
                elif self.index == 1:
                    return torch.add(x[0][1], x[1][1])
        
        elif type(x[0]) == list :
            if type(x[1]) == list :
                if self.index == 0:
                    return torch.add(x[0][0], x[1][0][0])
                elif self.index == 1:
                    return torch.add(x[0][0], x[1][1][1])
            elif type(x[1]) == tuple :
                if self.index == 0:
                    return torch.add(x[0][0], x[1][0])
                elif self.index == 1:
                    return torch.add(x[0][0], x[1][1])

        else:
            if self.index == 0:
                return torch.add(x[0], x[1][0])
            elif self.index == 1:
                return torch.add(x[0], x[1][1])
           


###！！！双向对齐，相互偏移！！！ 可见光先为参考帧 ###
class mutual_alignment(pl.LightningModule):
    def __init__(self, dim=64, memory=False, stride=1, type='group_conv'):
        
        super(mutual_alignment, self).__init__()
        
        act = nn.GELU()
        bias = False

        kernel_size = 3
        padding = kernel_size//2
        deform_groups = 8
        out_channels = deform_groups * 3 * (kernel_size**2)

        self.offset_conv_1 = nn.Conv2d(dim, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.deform_1 = DeformConv2d(dim, dim, kernel_size, padding = 2, groups = deform_groups, dilation=2)            
        # self.back_projection_1 = ref_back_projection(dim, stride=1)


        self.offset_conv_2 = nn.Conv2d(dim, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.deform_2 = DeformConv2d(dim, dim, kernel_size, padding = 2, groups = deform_groups, dilation=2)            
        # self.back_projection_2 = ref_back_projection(dim, stride=1)
        
        self.bottleneck_1 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
        self.bottleneck_2 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
        
        if memory==True:
            self.bottleneck_o = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
            
    def offset_gen(self, x):
        
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        
        return offset, mask
        
    def forward(self, x, prev_offset_feat=None):
        
        B, f, H, W = x[0].size()  # 8 128 80 80
        x1=x[0].unsqueeze(0)  # 1 8 128 80 80  RGB
        x2=x[1].unsqueeze(0)  # 1 8 128 80 80  IR
        x = torch.cat([x1, x2], dim=0) #2 8 128 80 80
    

        # 可见光作为参考帧
        ref_ir = x[1]  # 8 128 80 80 
        # ref = torch.repeat_interleave(ref, B, dim=0)


        offset_feat_rgb = self.bottleneck_1(torch.cat([ref_ir, x[0]], dim=1))  # 8 128 80 80

        # if not prev_offset_feat==None:
        #     offset_feat_ir = self.bottleneck_o(torch.cat([prev_offset_feat, offset_feat_ir], dim=1))

        offset_rgb, mask_rgb = self.offset_gen(self.offset_conv_1(offset_feat_rgb)) 

        aligned_feat_rgb = self.deform_1(x[0], offset_rgb, mask_rgb)
        # aligned_feat_rgb = torch.cat([x[1].unsqueeze(0),aligned_feat_rgb.unsqueeze(0)], dim=0)

        # aligned_feat_rgb = self.back_projection_1(aligned_feat_rgb)



        # 红外作为参考帧
        # ref_ir = x[1]
        ref_rgb = aligned_feat_rgb

        offset_feat_ir = self.bottleneck_2(torch.cat([ref_rgb, x[1]], dim=1))  # 8 128 80 80


        # if not prev_offset_feat==None:
        #     offset_feat_rgb = self.bottleneck_o(torch.cat([prev_offset_feat, offset_feat_rgb], dim=1))

        offset_ir, mask_ir = self.offset_gen(self.offset_conv_2(offset_feat_ir)) 

        aligned_feat_ir = self.deform_2(x[1], offset_rgb, mask_rgb)
        # aligned_feat_ir = torch.cat([x[0].unsqueeze(0),aligned_feat_ir.unsqueeze(0)], dim=0)

        # aligned_feat_ir = self.back_projection_2(aligned_feat_ir)




        #  输出一个元组
        feature_out = (aligned_feat_rgb, aligned_feat_ir)
        
        
        return feature_out
        # return aligned_feat_ir, aligned_feat_rgb  #, offset_feat  # 8 128 80 80

###！！！双向对齐，相互偏移！！！  可见光先为参考帧  ###
class mutual_alignment1(pl.LightningModule):
    def __init__(self, dim=64, stride=1, type='group_conv'):
        
        super(mutual_alignment1, self).__init__()
        
        act = nn.GELU()
        bias = False

        kernel_size = 3
        padding = kernel_size//2
        deform_groups = 8
        out_channels = deform_groups * 3 * (kernel_size**2)

        self.offset_conv_1 = nn.Conv2d(dim, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.deform_1 = DeformConv2d(dim, dim, kernel_size, padding = 2, groups = deform_groups, dilation=2)            
        # self.back_projection_1 = ref_back_projection(dim, stride=1)


        self.offset_conv_2 = nn.Conv2d(dim, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.deform_2 = DeformConv2d(dim, dim, kernel_size, padding = 2, groups = deform_groups, dilation=2)            
        # self.back_projection_2 = ref_back_projection(dim, stride=1)

        self.down1 = nn.Conv2d(dim//2, dim, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(dim//2, dim, 3, stride=2, padding=1)
        
        self.bottleneck_1 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
        self.bottleneck_2 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
        
    
        self.bottleneck_o1 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
        self.bottleneck_o2 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
            
    def offset_gen(self, x):
        
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        
        return offset, mask
        
    def forward(self, x):
        
        B, f, H, W = x[0].size()  # 8 128 80 80
        x1=x[0]  # 1 8 128 80 80
        x2=x[1]  # 1 8 128 80 80
        # x = torch.cat([x1, x2], dim=0) #2 8 128 80 80

        if len(x) == 3 :
            if type(x[2]) == tuple :
                prev_offset_feat_ir = x[2][0]
                prev_offset_feat_ir = self.down1(prev_offset_feat_ir)

                prev_offset_feat_rgb = x[2][1]
                prev_offset_feat_rgb = self.down2(prev_offset_feat_rgb)

            else :
                prev_offset_feat_ir = None
                prev_offset_feat_rgb = None
        else :
                prev_offset_feat_ir = None
                prev_offset_feat_rgb = None

        # 可见光作为参考帧
        ref_rgb = x1  # 8 128 80 80 
        # ref = torch.repeat_interleave(ref, B, dim=0)


        offset_feat_ir = self.bottleneck_1(torch.cat([ref_rgb, x2], dim=1))  # 8 128 80 80

        if not prev_offset_feat_ir==None:
            offset_feat_ir = self.bottleneck_o1(torch.cat([prev_offset_feat_ir, offset_feat_ir], dim=1))

        offset_ir, mask_ir = self.offset_gen(self.offset_conv_1(offset_feat_ir)) 

        aligned_feat_ir = self.deform_1(x2, offset_ir, mask_ir)
        # aligned_feat_ir = torch.cat([x[0].unsqueeze(0),aligned_feat_ir.unsqueeze(0)], dim=0)

        # aligned_feat_ir = self.back_projection_1(aligned_feat_ir)



        # 红外作为参考帧
        # ref_ir = x[1]
        ref_ir = aligned_feat_ir

        offset_feat_rgb = self.bottleneck_2(torch.cat([ref_ir, x1], dim=1))  # 8 128 80 80


        if not prev_offset_feat_rgb==None:
            offset_feat_rgb = self.bottleneck_o2(torch.cat([prev_offset_feat_rgb, offset_feat_rgb], dim=1))

        offset_rgb, mask_rgb = self.offset_gen(self.offset_conv_2(offset_feat_rgb)) 

        aligned_feat_rgb = self.deform_2(x1, offset_rgb, mask_rgb)
        # aligned_feat_rgb = torch.cat([x[1].unsqueeze(0),aligned_feat_rgb.unsqueeze(0)], dim=0)

        # aligned_feat_rgb = self.back_projection_2(aligned_feat_rgb)




        #  输出一个元组
        feature_out = (aligned_feat_rgb, aligned_feat_ir, offset_feat_ir, offset_feat_rgb)
        
        
        return feature_out
        # return aligned_feat_ir, aligned_feat_rgb  #, offset_feat  # 8 128 80 80


###！！！双向对齐，相互偏移！！！  红外先为参考帧  ###
class mutual_alignment2(pl.LightningModule):
    def __init__(self, dim=64, stride=1, type='group_conv'):
        
        super(mutual_alignment2, self).__init__()
        
        act = nn.GELU()
        bias = False

        kernel_size = 3
        padding = kernel_size//2
        deform_groups = 8
        out_channels = deform_groups * 3 * (kernel_size**2)

        self.offset_conv_1 = nn.Conv2d(dim, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.deform_1 = DeformConv2d(dim, dim, kernel_size, padding = 2, groups = deform_groups, dilation=2)            
        # self.back_projection_1 = ref_back_projection(dim, stride=1)


        self.offset_conv_2 = nn.Conv2d(dim, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.deform_2 = DeformConv2d(dim, dim, kernel_size, padding = 2, groups = deform_groups, dilation=2)            
        # self.back_projection_2 = ref_back_projection(dim, stride=1)

        self.down1 = nn.Conv2d(dim//2, dim, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(dim//2, dim, 3, stride=2, padding=1)
        
        self.bottleneck_1 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
        self.bottleneck_2 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
        
    
        self.bottleneck_o1 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
        self.bottleneck_o2 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
            
    def offset_gen(self, x):
        
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        
        return offset, mask
        
    def forward(self, x):
        
        B, f, H, W = x[0].size()  # 8 128 80 80
        x1=x[0]  # 1 8 128 80 80
        x2=x[1]  # 1 8 128 80 80
        # x = torch.cat([x1, x2], dim=0) #2 8 128 80 80

        if len(x) == 3 :
            if type(x[2]) == tuple :
                prev_offset_feat_ir = x[2][0]
                prev_offset_feat_ir = self.down1(prev_offset_feat_ir)

                prev_offset_feat_rgb = x[2][1]
                prev_offset_feat_rgb = self.down2(prev_offset_feat_rgb)

            else :
                prev_offset_feat_ir = None
                prev_offset_feat_rgb = None
        else :
                prev_offset_feat_ir = None
                prev_offset_feat_rgb = None

        # 红外作为参考帧
        ref_ir = x2  # 8 128 80 80 
        # ref = torch.repeat_interleave(ref, B, dim=0)


        offset_feat_rgb = self.bottleneck_1(torch.cat([ref_ir, x1], dim=1))  # 8 128 80 80

        if not prev_offset_feat_rgb==None:
            offset_feat_rgb = self.bottleneck_o1(torch.cat([prev_offset_feat_rgb, offset_feat_rgb], dim=1))

        offset_rgb, mask_rgb = self.offset_gen(self.offset_conv_1(offset_feat_rgb)) 

        aligned_feat_rgb = self.deform_1(x1, offset_rgb, mask_rgb)
        # aligned_feat_ir = torch.cat([x[0].unsqueeze(0),aligned_feat_ir.unsqueeze(0)], dim=0)

        # aligned_feat_ir = self.back_projection_1(aligned_feat_ir)



        # 可见光作为参考帧
        # ref_ir = x[1]
        ref_rgb = aligned_feat_rgb

        offset_feat_ir = self.bottleneck_2(torch.cat([ref_rgb, x2], dim=1))  # 8 128 80 80


        if not prev_offset_feat_ir==None:
            offset_feat_ir = self.bottleneck_o2(torch.cat([prev_offset_feat_ir, offset_feat_ir], dim=1))

        offset_ir, mask_ir = self.offset_gen(self.offset_conv_2(offset_feat_ir)) 

        aligned_feat_ir = self.deform_2(x2, offset_ir, mask_ir)
        # aligned_feat_rgb = torch.cat([x[1].unsqueeze(0),aligned_feat_rgb.unsqueeze(0)], dim=0)

        # aligned_feat_rgb = self.back_projection_2(aligned_feat_rgb)




        #  输出一个元组
        feature_out = (aligned_feat_rgb, aligned_feat_ir, offset_feat_ir, offset_feat_rgb)
        
        
        return feature_out
        # return aligned_feat_ir, aligned_feat_rgb  #, offset_feat  # 8 128 80 80

###！！！双向对齐，相互偏移！！！  可见光先为参考帧  ###
class mutual_align1(pl.LightningModule):
    def __init__(self, dim=64, stride=1, type='group_conv'):
        
        super(mutual_align1, self).__init__()
        
        act = nn.GELU()
        bias = False

        kernel_size = 3
        padding = kernel_size//2
        deform_groups = 8
        out_channels = deform_groups * 3 * (kernel_size**2)

        self.offset_conv_1 = nn.Conv2d(dim, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.deform_1 = DeformConv2d(dim, dim, kernel_size, padding = 2, groups = deform_groups, dilation=2)            
        # self.back_projection_1 = ref_back_projection(dim, stride=1)


        self.offset_conv_2 = nn.Conv2d(dim, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.deform_2 = DeformConv2d(dim, dim, kernel_size, padding = 2, groups = deform_groups, dilation=2)            
        # self.back_projection_2 = ref_back_projection(dim, stride=1)

        self.offset_conv_3 = nn.Conv2d(dim, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.deform_3 = DeformConv2d(dim, dim, kernel_size, padding = 2, groups = deform_groups, dilation=2)            
        # self.back_projection_3 = ref_back_projection(dim, stride=1)

        self.down1 = nn.Conv2d(dim//2, dim, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(dim//2, dim, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(dim//2, dim, 3, stride=2, padding=1)
        
        self.bottleneck_1 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
        self.bottleneck_2 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
        self.bottleneck_3 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)

        
    
        self.bottleneck_o1 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
        self.bottleneck_o2 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
        self.bottleneck_o3 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)

        # 用于调整通道数的卷积层
        self.conv1 = nn.Conv2d(3, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim//2, dim, kernel_size=1)
            
    def offset_gen(self, x):
        
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        
        return offset, mask
        
    def forward(self, x):
        
        B, f, H, W = x[0].size()  # 8 128 80 80
        if len(x) == 3 :
            x_rgb=x[0]  # 1 8 128 80 80 
            x_ir=x[1]  # 1 8 128 80 80
            x_r=x[2][2]
        else :
            if len(x[3]) == 3 :
                x_rgb=x[0]  # 1 8 128 80 80 
                x_ir=x[1]  # 1 8 128 80 80
                x_r=x[3][2]
            else:
                x_rgb=x[0]  # 1 8 128 80 80 
                x_ir=x[1]  # 1 8 128 80 80
                x_r=x[2][2]
        B1, f2, H3, W4 = x_r.size()
        if f2==3:
            x_r = self.conv1(x_r)  # 调整通道数到128
        else:
            x_r = self.conv2(x_r)  # 调整通道数到128
        # 使用双线性插值将空间维度调整为 (80, 80)
        x_r = F.interpolate(x_r, size=(H, W), mode='bilinear', align_corners=False)
        # L = F.interpolate(L1, size=(H, W), mode='bilinear', align_corners=False)

        # if len(x[2]) == 3 :
        if len(x[2]) == 10 :
            prev_offset_feat_ir = x[2][0]
            prev_offset_feat_ir = self.down1(prev_offset_feat_ir)

            prev_offset_feat_r = x[2][1]
            prev_offset_feat_r = self.down2(prev_offset_feat_r)

        else :
            prev_offset_feat_ir = None
            prev_offset_feat_r = None
        # else :
        #         prev_offset_feat_ir = None
        #         prev_offset_feat_r = None

        # R作为参考帧，对齐红外
        ref_r = x_r  # 8 128 80 80 
        # ref = torch.repeat_interleave(ref, B, dim=0)


        offset_feat_ir = self.bottleneck_1(torch.cat([ref_r, x_ir], dim=1))  # 8 128 80 80

        if not prev_offset_feat_ir==None:
            offset_feat_ir = self.bottleneck_o1(torch.cat([prev_offset_feat_ir, offset_feat_ir], dim=1))

        offset_ir, mask_ir = self.offset_gen(self.offset_conv_1(offset_feat_ir)) 

        aligned_feat_ir = self.deform_1(x_ir, offset_ir, mask_ir)
        # aligned_feat_ir = torch.cat([x[0].unsqueeze(0),aligned_feat_ir.unsqueeze(0)], dim=0)

        # aligned_feat_ir = self.back_projection_1(aligned_feat_ir)



        
        # aligned_feat_rgb = torch.cat([x[1].unsqueeze(0),aligned_feat_rgb.unsqueeze(0)], dim=0)

        # aligned_feat_rgb = self.back_projection_2(aligned_feat_rgb)

        # 红外作为参考帧，对齐R
        # ref_ir = x[1]
        ref_ir = aligned_feat_ir

        offset_feat_r = self.bottleneck_3(torch.cat([ref_ir, x_r], dim=1))  # 8 128 80 80


        if not prev_offset_feat_r==None:
            offset_feat_r = self.bottleneck_o3(torch.cat([prev_offset_feat_r, offset_feat_r], dim=1))

        offset_r, mask_r = self.offset_gen(self.offset_conv_3(offset_feat_r)) 

        aligned_feat_r = self.deform_3(x_r, offset_r, mask_r)
        # aligned_feat_rgb = torch.cat([x[1].unsqueeze(0),aligned_feat_rgb.unsqueeze(0)], dim=0)

        # aligned_feat_rgb = self.back_projection_2(aligned_feat_rgb)
        # 红外作为参考帧，对齐可见光
        # ref_ir = x[1]
        # ref_ir = aligned_feat_ir

        # offset_feat_rgb = self.bottleneck_2(torch.cat([ref_ir, x_rgb], dim=1))  # 8 128 80 80


        # if not prev_offset_feat_r==None:
        #     offset_feat_rgb = self.bottleneck_o2(torch.cat([prev_offset_feat_r, offset_feat_rgb], dim=1))

        # offset_rgb, mask_rgb = self.offset_gen(self.offset_conv_2(offset_feat_rgb)) 

        aligned_feat_rgb = self.deform_3(x_rgb, offset_r, mask_r)
        # aligned_feat_rgb = self.deform_2(x_rgb, offset_rgb, mask_rgb)

        aligned_feat_rgb = aligned_feat_rgb + aligned_feat_r


        #  输出一个元组
        feature_out = (aligned_feat_rgb, aligned_feat_ir, aligned_feat_r, offset_feat_ir, offset_feat_r)
        
        
        return feature_out
        # return aligned_feat_ir, aligned_feat_rgb  #, offset_feat  # 8 128 80 80



###！！！双向对齐，相互偏移！！！  可见光先为参考帧  ###
class mutual_align2(pl.LightningModule):
    def __init__(self, dim=64, stride=1, type='group_conv'):
        
        super(mutual_align2, self).__init__()
        
        act = nn.GELU()
        bias = False

        kernel_size = 3
        padding = kernel_size//2
        deform_groups = 8
        out_channels = deform_groups * 3 * (kernel_size**2)

        self.offset_conv_1 = nn.Conv2d(dim, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.deform_1 = DeformConv2d(dim, dim, kernel_size, padding = 2, groups = deform_groups, dilation=2)            
        # self.back_projection_1 = ref_back_projection(dim, stride=1)


        self.offset_conv_2 = nn.Conv2d(dim, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.deform_2 = DeformConv2d(dim, dim, kernel_size, padding = 2, groups = deform_groups, dilation=2)            
        # self.back_projection_2 = ref_back_projection(dim, stride=1)

        self.offset_conv_3 = nn.Conv2d(dim, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.deform_3 = DeformConv2d(dim, dim, kernel_size, padding = 2, groups = deform_groups, dilation=2)            
        # self.back_projection_3 = ref_back_projection(dim, stride=1)

        self.down1 = nn.Conv2d(dim//2, dim, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(dim//2, dim, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(dim//2, dim, 3, stride=2, padding=1)
        
        self.bottleneck_1 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
        self.bottleneck_2 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
        self.bottleneck_3 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)

        
    
        self.bottleneck_o1 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
        self.bottleneck_o2 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)
        self.bottleneck_o3 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias = bias), act)

        # 用于调整通道数的卷积层
        self.conv1 = nn.Conv2d(3, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim//2, dim, kernel_size=1)
            
    def offset_gen(self, x):
        
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        
        return offset, mask
        
    def forward(self, x):
        
        B, f, H, W = x[0].size()  # 8 128 80 80
        x_rgb=x[0]  # 1 8 128 80 80 
        x_ir=x[1]  # 1 8 128 80 80
        
        if len(x) == 3 :
            prev_offset_feat_ir = x[2][0]
            prev_offset_feat_ir = self.down1(prev_offset_feat_ir)

            prev_offset_feat_r = x[2][1]
            prev_offset_feat_r = self.down2(prev_offset_feat_r)

        else :
            prev_offset_feat_ir = None
            prev_offset_feat_r = None
        # else :
        #         prev_offset_feat_ir = None
        #         prev_offset_feat_r = None

        # R作为参考帧，对齐红外
        ref_rgb = x_rgb  # 8 128 80 80 
        # ref = torch.repeat_interleave(ref, B, dim=0)


        offset_feat_ir = self.bottleneck_1(torch.cat([ref_rgb, x_ir], dim=1))  # 8 128 80 80

        if not prev_offset_feat_ir==None:
            offset_feat_ir = self.bottleneck_o1(torch.cat([prev_offset_feat_ir, offset_feat_ir], dim=1))

        offset_ir, mask_ir = self.offset_gen(self.offset_conv_1(offset_feat_ir)) 

        aligned_feat_ir = self.deform_1(x_ir, offset_ir, mask_ir)
        # aligned_feat_ir = torch.cat([x[0].unsqueeze(0),aligned_feat_ir.unsqueeze(0)], dim=0)

        # aligned_feat_ir = self.back_projection_1(aligned_feat_ir)



        
        # aligned_feat_rgb = torch.cat([x[1].unsqueeze(0),aligned_feat_rgb.unsqueeze(0)], dim=0)

        # aligned_feat_rgb = self.back_projection_2(aligned_feat_rgb)

        # 红外作为参考帧，对齐R
        # ref_ir = x[1]
        ref_ir = aligned_feat_ir

        offset_feat_r = self.bottleneck_3(torch.cat([ref_ir, x_rgb], dim=1))  # 8 128 80 80


        if not prev_offset_feat_r==None:
            offset_feat_r = self.bottleneck_o3(torch.cat([prev_offset_feat_r, offset_feat_r], dim=1))

        offset_r, mask_r = self.offset_gen(self.offset_conv_3(offset_feat_r)) 

        aligned_feat_rgb = self.deform_3(x_rgb, offset_r, mask_r)
        # aligned_feat_rgb = torch.cat([x[1].unsqueeze(0),aligned_feat_rgb.unsqueeze(0)], dim=0)

        # aligned_feat_rgb = self.back_projection_2(aligned_feat_rgb)
        # 红外作为参考帧，对齐可见光
        # ref_ir = x[1]
        # ref_ir = aligned_feat_ir

        # offset_feat_rgb = self.bottleneck_2(torch.cat([ref_ir, x_rgb], dim=1))  # 8 128 80 80


        # if not prev_offset_feat_r==None:
        #     offset_feat_rgb = self.bottleneck_o2(torch.cat([prev_offset_feat_r, offset_feat_rgb], dim=1))

        # offset_rgb, mask_rgb = self.offset_gen(self.offset_conv_2(offset_feat_rgb)) 

        # aligned_feat_rgb = self.deform_2(x_rgb, offset_r, mask_r)
        # aligned_feat_rgb = self.deform_2(x_rgb, offset_rgb, mask_rgb)

        # aligned_feat_rgb = aligned_feat_rgb + aligned_feat_r


        #  输出一个元组
        feature_out = (aligned_feat_rgb, aligned_feat_ir, offset_feat_ir, offset_feat_r)
        
        
        return feature_out
        # return aligned_feat_ir, aligned_feat_rgb  #, offset_feat  # 8 128 80 80
    


class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out

class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """

        if type(x[0]) == tuple :
            rgb_fea = x[0][0]  # rgb_fea (tensor): dim:(B, C, H, W)
            ir_fea = x[0][1]   # ir_fea (tensor): dim:(B, C, H, W) 
            
        
        else:
            rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
            ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
            

        # ##########双向对齐#############
        # rgb_fea = x[0][0]  # rgb_fea (tensor): dim:(B, C, H, W)
        # ir_fea = x[0][1]   # ir_fea (tensor): dim:(B, C, H, W)

        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        return rgb_fea_out, ir_fea_out
    



