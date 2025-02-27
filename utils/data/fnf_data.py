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
from data.IO import *
#import scipy.signal as signal
import torch.nn.functional as F

from PIL import Image
#import scipy.ndimage
import os

os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

from data.data_utils import stn, mod_flash, get_dshift_pattern


class DatasetFnF(data.Dataset):    #继承自DATASET

    def __init__(self, opt):   #opt传入的是个字典
        super(DatasetFnF, self).__init__()
        print('Dataset: Denosing on AWGN with fixed sigma. Only dataroot_H is needed.')
        self.opt = opt

        """
        General params
        """
        self.data_dir = opt['dataroot_H']    #ok
        self.n_channels = opt['n_channels']  #ok
        self.min_refscale = opt['min_refscale']    #ok
        self.max_refscale = opt['max_refscale']    #ok
        self.min_noise = opt['min_noise']      #ok
        self.max_noise = opt['max_noise']     #ok
        self.min_power = opt['min_power']    #ok
        self.max_power = opt['max_power']    #ok
        self.band_noise = opt['band']          #ok
        self.min_poiss_K = opt['min_poiss_K']      #ok
        self.max_poiss_K = opt['max_poiss_K']     #ok
        self.crop_size = (opt['H_size'], opt['W_size'])    #ok
        self.split = opt['split']          #ok

        if self.split == 'train':
            with open(os.path.join(self.data_dir, 'train_split.txt'), 'r') as f:
                self.file_list = f.read().splitlines()        #file_list，文件对象列表
            self.warp_kwargs = {'disp_clip': opt['disp_clip'], 'crop_size': self.crop_size, 'train': True}   #stn用的参数
        elif self.split == 'val':
            with open(os.path.join(self.data_dir, 'val_small_split.txt'), 'r') as f:
                self.file_list = f.read().splitlines()
            self.warp_kwargs = {'disp_clip': opt['disp_clip'], 'crop_size': self.crop_size, 'train': False}
            """
            a small validation dataset with only 20 data for evaluation during training
            """
        else:
            raise NotImplementedError

        """
        we are only using the A subset in both FlyingThings3D TRAIN and TEST
        """
        if self.split == 'train':
            self.data_dir = os.path.join(self.data_dir, 'TRAIN/')
        else:
            self.data_dir = os.path.join(self.data_dir, 'TEST/')   #TODO 这里是VAL

        self.disp_dir = self.data_dir.replace('frames_cleanpass', 'disparity')

        """
        load calibrated pattern and related params
        """
        self.pattern_dir = opt['pattern_dir']    #ok
        self.pattern = np.load(self.pattern_dir).astype(np.float32)
        self.pattern_raw = np.zeros((self.pattern.shape[0] * 2, self.pattern.shape[1] * 2))
        self.pattern_raw[::2, ::2] = self.pattern[..., 0]
        self.pattern_raw[1::2, ::2] = self.pattern[..., 1]
        self.pattern_raw[1::2, 1::2] = self.pattern[..., 2]
        self.pattern_raw[::2, 1::2] = self.pattern[..., 3]
        self.pattern_raw = self.pattern_raw.astype(np.float32)       #self.pattern_raw 是CFA后数据
        self.pattern_boost = opt['pattern_boost']    #ok

    def flip(self, img):
        if img.ndim == 2:          #只是二维的
            img = np.flipud(img)    #在第一个维度进行倒置
        else:
            for cc in range(img.shape[2]):   #每个通道进行沿着第一维倒置
                img[..., cc] = np.flipud(img[..., cc])
        return img.copy()

    def get_ft3d_fd(self, idx):    #获取第几个图片的rgb和d,idx为int, index
        """
        load rgb-d data
        返回rgb,光照后的rgb，深度
        rgb:H*W*3
        rgb_flash:H*W*1
        depth:H*W*1 或者 3 channels
        """
        rgb_file = os.path.join(self.data_dir, self.file_list[idx])
        rgb = imageio.v2.imread(rgb_file).astype(np.float32) / 255.0
        rgb = np.concatenate((rgb, rgb[..., 1:2]), axis=2)    #rggb
        rgb_flash = rgb * mod_flash(rgb.shape[0], rgb.shape[1])  #模拟光照

        disp, _ = readPFM(os.path.join(self.disp_dir, self.file_list[idx].replace('png', 'pfm')))   #读深度图用READ PFM
        depth = np.clip(1.0 / np.asarray(disp) * 50.0, 1.0, 10.0)
        return rgb, rgb_flash, depth

    def __getitem__(self, idx):
        amb, flash, depth = self.get_ft3d_fd(idx)

        if self.split == 'train':   #引入随机性，从不同地方剪裁
            H, W, _ = flash.shape
            crop_idx_x = random.randint(0, H - self.crop_size[0] - 41)
            crop_idx_y = random.randint(0, W - self.crop_size[1] - 41)
            flash_crop = flash[crop_idx_x + 20:crop_idx_x + 20 + self.crop_size[0], \
                         crop_idx_y + 20:crop_idx_y + 20 + self.crop_size[1]]              #裁剪
            amb_crop = amb[crop_idx_x + 20:crop_idx_x + 20 + self.crop_size[0], \
                       crop_idx_y + 20:crop_idx_y + 20 + self.crop_size[1]]
            depth_crop = depth[crop_idx_x:crop_idx_x + self.pattern.shape[0], \
                         crop_idx_y:crop_idx_y + self.pattern.shape[1]]
            depth_crop = np.repeat(np.repeat(depth_crop, 2, axis=0), 2, axis=1)      #相当于大小乘以4，RAW

            if random.uniform(0, 1) > 0.5:             #有一定的几率进行翻转
                flash_crop = self.flip(flash_crop)
                amb_crop = self.flip(amb_crop)
                depth_crop = self.flip(depth_crop)

        elif self.split == 'val':   #不需要随机，从同一个地方裁剪
            flash_crop = flash[20:20 + self.crop_size[0], 20:20 + self.crop_size[1]]
            amb_crop = amb[20:20 + self.crop_size[0], 20:20 + self.crop_size[1]]
            depth_crop = depth[:self.pattern.shape[0], :self.pattern.shape[1]]
            depth_crop = np.repeat(np.repeat(depth_crop, 2, axis=0), 2, axis=1)
            crop_idx_x = 0
            crop_idx_y = 0
        else:
            raise NotImplementedError()

        pattern_shift, deltax, deltay = get_dshift_pattern(self.pattern_raw, depth_crop, **self.warp_kwargs) #深度变换
        refscale = 10 ** np.random.uniform(np.log10(self.min_refscale), np.log10(self.max_refscale))
        power = 10 ** np.random.uniform(np.log10(self.min_power), np.log10(self.max_power))
        band_noise_level = np.random.uniform(self.band_noise * 0.5, self.band_noise * 1.5)
        noise_level = 10 ** np.random.uniform(np.log10(self.min_noise), np.log10(self.max_noise))
        poiss_K = 10 ** np.random.uniform(np.log10(self.min_poiss_K), np.log10(self.max_poiss_K))
        band_noise = np.tile(np.random.normal(size=(flash_crop.shape[0], 1, flash_crop.shape[2])) * band_noise_level, \
                             (1, flash_crop.shape[1], 1))

        pattern = pattern_shift * self.pattern_boost * power
        img_pflash_crop = np.copy(flash_crop) * pattern + refscale * np.copy(amb_crop)
        sigma_map_flash = np.sqrt(img_pflash_crop * poiss_K / 4096 + noise_level ** 2)
        img_pflash_crop += np.random.normal(size=img_pflash_crop.shape) * sigma_map_flash + band_noise
        img_pflash_crop = np.clip(img_pflash_crop, -1.0, 1.0)
        img_pflash_crop = (img_pflash_crop * 4096).astype(np.int64).astype(np.float32) / 4096

        img_pflash_scale = np.clip(img_pflash_crop / refscale, -2.0, 2.0)
        pattern_norm = self.pattern[20:(20 + self.crop_size[0]), 20:(20 + self.crop_size[1])]
        img_pflash_crop = np.concatenate((img_pflash_scale, img_pflash_crop / (16 * power + refscale), pattern_norm),
                                         axis=2)

        gt = np.concatenate((amb_crop[..., :3], flash_crop[..., :3], pattern_shift, deltax, deltay,
                             np.ones_like(deltax) * refscale, np.ones_like(deltax) * power), axis=2)
        gt = torch.from_numpy(gt.transpose(2, 0, 1))
        img_pflash_crop = torch.from_numpy(img_pflash_crop.transpose(2, 0, 1))

        H_path = '{}'.format(idx)
        L_path = H_path
        img_L = img_pflash_crop
        img_H = gt
        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path, 'N': sigma_map_flash,
                'F': flash_crop, 'P': pattern_shift, 'D': depth_crop}          #8个通道

    def __len__(self):   #数据集长度，及文件列表长度
        return len(self.file_list)