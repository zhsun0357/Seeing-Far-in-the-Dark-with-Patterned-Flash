import numpy as np
import torch
import torch.utils.data as data
import imageio
import sys
import scipy.signal as signal
import torch.nn.functional as F
import os
from PIL import Image
import scipy.ndimage
sys.path.append('data/')
from IO import *
import pdb

def mod_flash(H, W):
    """
    modulate flash image, useful in flash/no-flash simulations
    """
    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
    gridX = gridX.astype(np.float32)
    gridY = gridY.astype(np.float32)

    x_ = np.random.uniform(0, float(W))
    y_ = np.random.uniform(0, float(H))
    period = np.random.uniform(100.0, 500.0)
    low = np.random.uniform(0.2, 1.0)
    high = np.random.uniform(low + 0.1, 1.0)
    amp = (high-low)/2
    mod_pattern = amp * np.sin(2*np.pi/period*( (gridX-x_)**2 + (gridY-y_)**2 )**0.5 ) + low + amp

    return mod_pattern


def stn(pattern_raw, sx, sy):
    """
    spatial transformation network for warping
    """
    assert (pattern_raw.shape == sx.shape) and (pattern_raw.shape == sy.shape)
    pattern_raw_t = torch.from_numpy(pattern_raw).unsqueeze(0).unsqueeze(1)
    theta = torch.Tensor([[1, 0, 0], [0, 1, 0]])
    theta = theta.repeat(1, 1, 1)
    grid = F.affine_grid(theta, pattern_raw_t.size())
    sx = torch.from_numpy(sx).unsqueeze(0).unsqueeze(1)
    sy = torch.from_numpy(sy).unsqueeze(0).unsqueeze(1)
    sx_norm = sx/pattern_raw_t.shape[2]
    sy_norm = sy/pattern_raw_t.shape[3]
    grid -= torch.cat((sx_norm.squeeze(1).unsqueeze(3), sy_norm.squeeze(1).unsqueeze(3)), dim = 3)
    pattern_shift_t = F.grid_sample(pattern_raw_t, grid)
    return pattern_shift_t[0,0].numpy()


def get_dshift_pattern(pattern_raw, depth, **kwargs):
    """
    warp calibrated RGGB raw pattern with scene depth
    """
    dir_type = np.random.randint(4)
    shift_kernels = torch.zeros(1,9,pattern_raw.shape[0], pattern_raw.shape[1])
    deltax = (1 - 1/depth).astype(np.float32) * np.random.uniform(0.5,kwargs['disp_clip'])
    deltay = (1 - 1/depth).astype(np.float32) * np.random.uniform(0.5,kwargs['disp_clip'])
    assert np.all(deltax >= 0) and np.all(deltax <= kwargs['disp_clip']) \
        and np.all(deltay >= 0) and np.all(deltay <= kwargs['disp_clip'])
    if dir_type == 1:
        deltax = -deltax
    if dir_type == 2:
        deltay = -deltay
    if dir_type == 3:
        deltax = -deltax
        deltay = -deltay

    pattern_raw_shift = stn(pattern_raw, deltax, deltay)
    pattern_shift = np.concatenate((pattern_raw_shift[::2,::2, np.newaxis],\
                                 pattern_raw_shift[1::2,::2, np.newaxis],\
                                 pattern_raw_shift[1::2,1::2, np.newaxis],\
                                 pattern_raw_shift[::2,1::2, np.newaxis]), axis = 2)

    return pattern_shift[20:(20+kwargs['crop_size'][0]), 20:(20+kwargs['crop_size'][1])].astype(np.float32),\
            deltax[::2,::2][20:(20+kwargs['crop_size'][0]), 20:(20+kwargs['crop_size'][1]), np.newaxis].astype(np.float32), \
            deltay[::2,::2][20:(20+kwargs['crop_size'][0]), 20:(20+kwargs['crop_size'][1]), np.newaxis].astype(np.float32)


def get_pattern_raw(pattern):
    pattern_raw = np.zeros((pattern.shape[0]*2, pattern.shape[1]*2))
    pattern_raw[::2,::2] = pattern[...,0]
    pattern_raw[1::2,::2] = pattern[...,1]
    pattern_raw[1::2,1::2] = pattern[...,2]
    pattern_raw[::2,1::2] = pattern[...,3]
    return pattern_raw


def get_simu_eval(pattern_calib, flash, depth, **kwargs):
    depth = np.repeat(np.repeat(depth, 2, axis = 0), 2, axis = 1).astype(np.float32)

    warp_kwargs = {'disp_clip': kwargs['baseline'], 'crop_size': [kwargs['crop_H'],kwargs['crop_W']]}
    pattern_raw = get_pattern_raw(pattern_calib).astype(np.float32)
    pattern_real, sx_gt, sy_gt = get_dshift_pattern(pattern_raw, depth, **warp_kwargs)
    
    pattern = pattern_real * kwargs['boost'] * kwargs['power']
    img_flash = np.copy(flash) * pattern
    sigma_map_flash = np.sqrt(img_flash * kwargs['poiss_K']/4096 + kwargs['noise']**2)
    img_flash += np.random.normal(size = img_flash.shape)*sigma_map_flash
    img_flash = np.clip(img_flash, -1.0, 1.0)
    img_flash = (img_flash * 4096).astype(np.int64).astype(np.float32)/4096
    
    img_flash = np.concatenate((img_flash/kwargs['boost']/kwargs['power'], \
                pattern_calib[20:(20+kwargs['crop_H']), 20:(20+kwargs['crop_W'])]), axis = 2)
    return img_flash
    

def get_ft3d_fd_eval(data_dir, disp_dir, **kwargs):
    """
    FT3D dataset has too much high freq details, make resolution lower
    """
    rgb_file = os.path.join(data_dir, kwargs['fname'])
    rgb = imageio.imread(rgb_file).astype(np.float32)/255.0
    rgb = np.concatenate((rgb, rgb[...,1:2]), axis = 2)
    rgb = rgb[20:(20+kwargs['crop_H']), 20:(20+kwargs['crop_W'])]
    
    disp, _ = readPFM(os.path.join(disp_dir, kwargs['fname'].replace('png', 'pfm')))
    disp = disp[:(kwargs['crop_H']+40), :(kwargs['crop_W']+40)]
    depth = np.clip(1.0/np.asarray(disp) * kwargs['depth_clip']*10, 0.0, kwargs['depth_clip']) 
    depth = (depth - np.min(depth))/(np.max(depth) - np.min(depth))*(kwargs['depth_clip'] - 1) + 1
    
    rgb = rgb/depth[20:-20,20:-20,np.newaxis]**2
    
    dbase = np.sqrt(1/kwargs['power'])
    depth += (dbase/8) - 1
    return rgb, depth
