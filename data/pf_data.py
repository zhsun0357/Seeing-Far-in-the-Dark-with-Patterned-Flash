import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
import pickle
import imageio
import sys
sys.path.append('data/')
from IO import *
import scipy.signal as signal
import torch.nn.functional as F

from PIL import Image
import scipy.ndimage
import os
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

from data_utils import stn, mod_flash, get_dshift_pattern

class DatasetPF(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """
    def __init__(self, opt):
        super(DatasetPF, self).__init__()
        print('Dataset: Denosing on AWGN with fixed sigma. Only dataroot_H is needed.')
        self.opt = opt

        """
        General params
        """
        self.n_channels = opt['n_channels']
        self.min_noise = opt['min_noise']
        self.max_noise = opt['max_noise']
        self.min_power = opt['min_power']
        self.max_power = opt['max_power']
        self.band_noise = opt['band']
        self.min_poiss_K = opt['min_poiss_K']
        self.max_poiss_K = opt['max_poiss_K']
        self.crop_size = (opt['H_size'], opt['W_size'])
        self.warp_kwargs = {'disp_clip': opt['disp_clip'], 'crop_size': self.crop_size}
        self.split = opt['split']
        self.data_dir = opt['dataroot_H']
        
        if self.split == 'train':
            with open(os.path.join(self.data_dir, 'train_split.txt'), 'r') as f:
                self.file_list = f.read().splitlines()
        elif self.split == 'val':
            with open(os.path.join(self.data_dir, 'val_small_split.txt'), 'r') as f:
                self.file_list = f.read().splitlines()
            """
            a small validation dataset with only 20 data for evaluation during training
            """
        else:
            raise NotImplementedError
        
        """
        we are only using the A subset in both FlyingThings3D TRAIN and TEST
        """  
        if self.split == 'train':
            self.data_dir = os.path.join(self.data_dir, 'TRAIN/A')
        else:
            self.data_dir = os.path.join(self.data_dir, 'TEST/A')

        self.disp_dir = self.data_dir.replace('frames_cleanpass', 'disparity')

        
        """
        load calibrated pattern and related params
        """
        self.pattern_dir = opt['pattern_dir']
        self.pattern = np.load(self.pattern_dir).astype(np.float32)
        self.pattern_raw = np.zeros((self.pattern.shape[0]*2, self.pattern.shape[1]*2))
        self.pattern_raw[::2,::2] = self.pattern[...,0]
        self.pattern_raw[1::2,::2] = self.pattern[...,1]
        self.pattern_raw[1::2,1::2] = self.pattern[...,2]
        self.pattern_raw[::2,1::2] = self.pattern[...,3]
        self.pattern_raw = self.pattern_raw.astype(np.float32)
        self.pattern_boost = opt['pattern_boost']

        
    def flip(self, img):
        if img.ndim == 2:
            img = np.flipud(img)
        else:
            for cc in range(img.shape[2]):
                img[...,cc] = np.flipud(img[...,cc])
        return img.copy()
    
    def get_ft3d_fd(self, idx):
        """
        load rgb-d data
        """
        rgb_file = os.path.join(self.data_dir, self.file_list[idx])
        rgb = imageio.imread(rgb_file).astype(np.float32)/255.0
        rgb = np.concatenate((rgb, rgb[...,1:2]), axis = 2)
        
        disp, _ = readPFM(os.path.join(self.disp_dir, self.file_list[idx].replace('png', 'pfm')))
        depth = np.clip(1.0/np.asarray(disp) * 50.0, 1.0, 10.0)
        return rgb, depth
        
        
    def __getitem__(self, idx):
        flash, depth = self.get_ft3d_fd(idx)
        
        if self.split == 'train':
            H,W,_ = flash.shape
            crop_idx_x = random.randint(0,H-self.crop_size[0]-41)
            crop_idx_y = random.randint(0,W-self.crop_size[1]-41)
            flash_crop = flash[crop_idx_x+20:crop_idx_x+20 + self.crop_size[0], \
                               crop_idx_y+20:crop_idx_y+20 + self.crop_size[1]]
            depth_crop = depth[crop_idx_x:crop_idx_x + self.pattern.shape[0], \
                               crop_idx_y:crop_idx_y + self.pattern.shape[1]]
            depth_crop = np.repeat(np.repeat(depth_crop, 2, axis = 0), 2, axis = 1)
            
            if random.uniform(0,1) > 0.5:
                flash_crop = self.flip(flash_crop)
                depth_crop = self.flip(depth_crop)
                
        elif self.split == 'val':
            flash_crop = flash[20:20+self.crop_size[0],20:20+self.crop_size[1]]
            depth_crop = depth[:self.pattern.shape[0], :self.pattern.shape[1]]
            depth_crop = np.repeat(np.repeat(depth_crop, 2, axis = 0), 2, axis = 1)
            crop_idx_x = 0
            crop_idx_y = 0            
        else:
            raise NotImplementedError()
        
        pattern_shift, deltax, deltay = get_dshift_pattern(self.pattern_raw, depth_crop, **self.warp_kwargs)
        power = 10**np.random.uniform(np.log10(self.min_power), np.log10(self.max_power))
        band_noise_level = np.random.uniform(self.band_noise*0.5, self.band_noise*1.5)
        noise_level = 10**np.random.uniform(np.log10(self.min_noise), np.log10(self.max_noise))
        poiss_K = 10**np.random.uniform(np.log10(self.min_poiss_K), np.log10(self.max_poiss_K))
        band_noise1 = np.tile(np.random.normal(size = (flash_crop.shape[0], 1, flash_crop.shape[2]))*band_noise_level, \
                             (1, flash_crop.shape[1], 1))
        band_noise2 = np.tile(np.random.normal(size = (flash_crop.shape[0], 1, flash_crop.shape[2]))*band_noise_level, \
                             (1, flash_crop.shape[1], 1))
        
        pattern = pattern_shift * self.pattern_boost * power
        img_flash_crop = np.copy(flash_crop) * pattern
        sigma_map_flash = np.sqrt(img_flash_crop * poiss_K/4096 + noise_level**2)
        img_flash_crop += np.random.normal(size = img_flash_crop.shape)*sigma_map_flash + band_noise2
        img_flash_crop = np.clip(img_flash_crop, -1.0, 1.0)
        img_flash_crop = (img_flash_crop * 4096).astype(np.int64).astype(np.float32)/4096
        
        pattern_norm = self.pattern[20:(20+self.crop_size[0]), 20:(20+self.crop_size[1])]
        img_flash_crop = np.concatenate((img_flash_crop/self.pattern_boost/power, pattern_norm), axis = 2)
        
        gt = np.concatenate((flash_crop, pattern_shift, deltax, deltay), axis = 2)
        gt = torch.from_numpy(gt.transpose(2,0,1))
        img_flash_crop = torch.from_numpy(img_flash_crop.transpose(2,0,1))
        
        H_path = '{}'.format(idx)
        L_path = H_path
        img_L = img_flash_crop
        img_H = gt
        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path, 'N': sigma_map_flash, \
                'P': pattern_shift, 'D': depth_crop}

    def __len__(self):
        return len(self.file_list)
