from __future__ import print_function, division
import os
import torchvision
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
from math import log10, pi


class ResUNet(torch.nn.Module):
    def __init__(self, in_chans = 6, out_chans = 3, disp_clip = 1.0):
        super(ResUNet, self).__init__()
        self.disp_clip = disp_clip
        
        self.aspp1 = ASPP(in_chans, 32)
        self.conv1 = ConvLayer(32, 32, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.aspp2 = ASPP(32, 32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.aspp3 = ASPP(64, 64)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.aspp4 = ASPP(128, 128)
        
        self.res1 = []
        self.res2 = []
        self.res3 = []
        for ii in range(4):
            self.res1.append(ResidualBlock(32))
        for ii in range(8):
            self.res2.append(ResidualBlock(64))
        for ii in range(16):
            self.res3.append(ResidualBlock(128))
        self.res1 = nn.Sequential(*self.res1)
        self.res2 = nn.Sequential(*self.res2)
        self.res3 = nn.Sequential(*self.res3)
       
        self.deconv1 = UpsampleConvLayer(128*2, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64*2, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32*2, out_chans, kernel_size=9, stride=1)

        self.relu = torch.nn.ReLU()
    
    def get_pattern_raw(self, X):
        B, C, H, W = X.shape
        pattern_raw = torch.zeros(B, 1, H*2, W*2).to(X.device)
        pattern_raw[:,0,::2,::2] = X[:,-4,:,:]
        pattern_raw[:,0,1::2,::2] = X[:,-3,:,:]
        pattern_raw[:,0,1::2,1::2] = X[:,-2,:,:]
        pattern_raw[:,0,::2,1::2] = X[:,-1,:,:]
        pattern_raw.requires_grad_(requires_grad = False)    
        return pattern_raw
    
    def get_pattern_color(self, pattern_raw):
        return torch.cat((pattern_raw[:,:1,::2,::2], \
                   pattern_raw[:,:1,1::2,::2], \
                   pattern_raw[:,:1,1::2,1::2], \
                   pattern_raw[:,:1,::2,1::2]), dim = 1)
        
    def get_pattern_shift(self, X, sx, sy):
        pattern_raw = self.get_pattern_raw(X)
        theta = torch.Tensor([[1, 0, 0], [0, 1, 0]])
        theta = theta.repeat(pattern_raw.size()[0], 1, 1)
        grid = F.affine_grid(theta, pattern_raw.size()).to(X.device)
        grid.requires_grad_(requires_grad = False)
        sx = torch.repeat_interleave(torch.repeat_interleave(sx, 2, dim = 1), 2, dim = 2)
        sy = torch.repeat_interleave(torch.repeat_interleave(sy, 2, dim = 1), 2, dim = 2)
        sx_norm = sx/pattern_raw.shape[2]
        sy_norm = sy/pattern_raw.shape[3]
        grid -= torch.cat((sx_norm.unsqueeze(3), sy_norm.unsqueeze(3)), dim = 3)
        pattern_raw_shift = F.grid_sample(pattern_raw, grid)
        return self.get_pattern_color(pattern_raw_shift)
    
    def forward(self, X):
        o1 = self.relu(self.conv1(self.aspp1(X)))
        o2 = self.relu(self.conv2(self.aspp2(o1)))
        o3 = self.relu(self.conv3(self.aspp3(o2)))
        o1 = self.res1(o1)
        o2 = self.res2(o2)
        y = self.res3(self.aspp4(o3))
        in1 = torch.cat( (y, o3), 1 )
        y = self.relu(self.deconv1(in1))
        in2 = torch.cat( (y, o2), 1 )
        y = self.relu(self.deconv2(in2))
        in3 = torch.cat( (y, o1), 1 )
        y = self.deconv3(in3)
        
        sx = torch.nn.Tanh()(y[:,3,:,:])*self.disp_clip
        sy = torch.nn.Tanh()(y[:,4,:,:])*self.disp_clip
        pattern_shift = self.get_pattern_shift(X, sx, sy)
        y = torch.cat((y[:,:3,:,:], pattern_shift, \
                       sx.unsqueeze(1), sy.unsqueeze(1)), dim = 1)
        
        return y

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(torch.nn.Module):
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return out

class ConvLayer_atrous(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(ConvLayer_atrous, self).__init__()
        self.atrous_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.atrous_conv(x))
        return x
    
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        dilations = [1, 3, 6, 12]
        self.aspp1 = ConvLayer_atrous(in_channels, out_channels, 3, padding=1, dilation=dilations[0])
        self.aspp2 = ConvLayer_atrous(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ConvLayer_atrous(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ConvLayer_atrous(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(out_channels*5, out_channels, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.relu(self.conv1(x))
        return x
    
    
class UpsampleConvLayer(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out