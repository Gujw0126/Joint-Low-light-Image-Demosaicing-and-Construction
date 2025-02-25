import numpy as np
import torch
import torch.utils.data as data
import imageio
import sys
#import scipy.signal as signal
import torch.nn.functional as F
import os
from PIL import Image
#import scipy.ndimage
sys.path.append('data/')
from data.IO import *
import pdb


'''
模拟光照变换的函数，只在fnf_dataset中用到，模拟一个曝光，图片会变亮
输入H,W,返回（H,W,1）的np矩阵，矩阵内部元素0-1，float32，只有光照一个维度
'''
def mod_flash(H, W):
    """
    modulate flash image, useful in flash/no-flash simulations
    返回（H,W,1）的np矩阵，矩阵内部元素0-1，float32
    """
    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))  # X: 0,1...W   Y: 0,1...H
    gridX = gridX.astype(np.float32)  #数值转换
    gridY = gridY.astype(np.float32)

    x_ = np.random.uniform(0, float(W))      #在0-W之间随机取一个值
    y_ = np.random.uniform(0, float(H))      #在0-H之间随机取一个值
    period = np.random.uniform(100.0, 500.0)
    low = np.random.uniform(0.2, 1.0)
    high = np.random.uniform(low + 0.1, 1.0)
    amp = (high - low) / 2
    mod_pattern = amp * np.sin(2 * np.pi / period * ((gridX - x_) ** 2 + (gridY - y_) ** 2) ** 0.5) + low + amp

    return mod_pattern[..., np.newaxis]    #增加一维




def stn(pattern_raw, sx, sy):
    """
    spatial transformation network for warping
    为了让CNN能有扭曲不变性
    该模块在训练阶段学习如何对输入数据进行变换更有益于模型的分类，
    然后在测试阶段应用已经训练好的网络对输入数据进行执行相应的变换，从而提高模型的识别率。
    输入的sx与sy的形状与pattern_raw一致
    pattern_raw,sx,sy:numpy array
    返回处理好的np图像
    """
    assert (pattern_raw.shape == sx.shape) and (pattern_raw.shape == sy.shape)
    pattern_raw_t = torch.from_numpy(pattern_raw).unsqueeze(0).unsqueeze(1)     #前面增加两个维度
    theta = torch.tensor([[1, 0, 0], [0, 1, 0]])
    theta = theta.repeat(1, 1, 1)
    theta = theta.float()
    grid = F.affine_grid(theta, pattern_raw_t.shape)
    sx = torch.from_numpy(sx).unsqueeze(0).unsqueeze(1)      #前面增加两个维度
    sy = torch.from_numpy(sy).unsqueeze(0).unsqueeze(1)       #前面增加两个维度
    sx_norm = sx / pattern_raw_t.shape[2]
    sy_norm = sy / pattern_raw_t.shape[3]
    grid -= torch.cat((sx_norm.squeeze(1).unsqueeze(3), sy_norm.squeeze(1).unsqueeze(3)), dim=3)
    pattern_shift_t = F.grid_sample(pattern_raw_t, grid)         #根据grid对pattern_raw_t进行变换
    return pattern_shift_t[0, 0].numpy()

'''
根据深度变换图像，输入一些参数，如是否训练，剪裁
返回处理好的图像和x,y变换矩阵
'''
def get_dshift_pattern(pattern_raw, depth, **kwargs):  #kwargs 传入参数个数未知，需要命名
    """
    根据深度变换pattern图像，返回变换后的RAW图像
    warp calibrated RGGB raw pattern with scene depth
    """
    dir_type = np.random.randint(4)    #随机生成0-4
    shift_kernels = torch.zeros(1, 9, pattern_raw.shape[0], pattern_raw.shape[1])
    if kwargs['train']:
        deltax = (1 - 1 / depth).astype(np.float32) * np.random.uniform(0.5, kwargs['disp_clip'])  #0.5-3
        deltay = (1 - 1 / depth).astype(np.float32) * np.random.uniform(0.5, kwargs['disp_clip'])
    else:
        deltax = (1 - 1 / depth).astype(np.float32) * kwargs['disp_clip']
        deltay = (1 - 1 / depth).astype(np.float32) * kwargs['disp_clip']
    assert np.all(deltax >= 0) and np.all(deltax <= kwargs['disp_clip']) \
           and np.all(deltay >= 0) and np.all(deltay <= kwargs['disp_clip'])
    if kwargs['train']:        #0不做处理
        if dir_type == 1:
            deltax = -deltax
        if dir_type == 2:
            deltay = -deltay
        if dir_type == 3:
            deltax = -deltax
            deltay = -deltay

    pattern_raw_shift = stn(pattern_raw, deltax, deltay)

    pattern_shift = np.concatenate((pattern_raw_shift[::2, ::2, np.newaxis],
                                    pattern_raw_shift[1::2, ::2, np.newaxis],
                                    pattern_raw_shift[1::2, 1::2, np.newaxis],
                                    pattern_raw_shift[::2, 1::2, np.newaxis]), axis=2)     #RGGB  h,w,c,C=4


    return pattern_raw_shift[20:(20 + kwargs['crop_size'][0]), 20:(20 + kwargs['crop_size'][1])].astype(np.float32), \
           deltax[::2, ::2][20:(20 + kwargs['crop_size'][0]), 20:(20 + kwargs['crop_size'][1]), np.newaxis].astype(
               np.float32), \
           deltay[::2, ::2][20:(20 + kwargs['crop_size'][0]), 20:(20 + kwargs['crop_size'][1]), np.newaxis].astype(
               np.float32)


def get_pattern_raw(pattern): #input RGGB
    pattern_raw = np.zeros((pattern.shape[0] * 2, pattern.shape[1] * 2))
    pattern_raw[::2, ::2] = pattern[..., 0]
    pattern_raw[1::2, ::2] = pattern[..., 1]
    pattern_raw[1::2, 1::2] = pattern[..., 2]
    pattern_raw[::2, 1::2] = pattern[..., 3]
    return pattern_raw


def get_simu_eval(pattern_calib, flash, depth, **kwargs):   #根据光照给图像打光
    depth = np.repeat(np.repeat(depth, 2, axis=0), 2, axis=1).astype(np.float32)   #W*2，H*2，为了适应RAW数据

    warp_kwargs = {'disp_clip': kwargs['baseline'], 'crop_size': [kwargs['crop_H'], kwargs['crop_W']],  #参数字典
                   'train': kwargs['train']}
    pattern_raw = get_pattern_raw(pattern_calib).astype(np.float32)
    pattern_real, sx_gt, sy_gt = get_dshift_pattern(pattern_raw, depth, **warp_kwargs)    #根据深度进行变换

    pattern = pattern_real * kwargs['boost'] * kwargs['power']
    img_flash = np.copy(flash) * pattern                                        #模拟光照后的图片
    sigma_map_flash = np.sqrt(img_flash * kwargs['poiss_K'] / 4096 + kwargs['noise'] ** 2)   #噪声
    img_flash += np.random.normal(size=img_flash.shape) * sigma_map_flash   #随机读出噪声（应该是）
    img_flash = np.clip(img_flash, -1.0, 1.0)
    img_flash = (img_flash * 4096).astype(np.int64).astype(np.float32) / 4096

    img_flash = np.concatenate((img_flash / kwargs['boost'] / kwargs['power'],
                                pattern_calib[20:(20 + kwargs['crop_H']), 20:(20 + kwargs['crop_W'])]), axis=2)
    return img_flash


'''
读取+剪裁+调分辨率+归一化
'''
def get_ft3d_fd_eval(**kwargs):
    """
    FT3D dataset has too much high freq details, make resolution lower
    """
    rgb_file = os.path.join('data/ft3d_data', '{}.png'.format(kwargs['fname']))
    rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0    #换到0-1
    rgb = np.concatenate((rgb, rgb[..., 1:2]), axis=2)     #rggb
    rgb = rgb[20:(20 + kwargs['crop_H']), 20:(20 + kwargs['crop_W'])]   #剪裁

    disp, _ = readPFM(os.path.join('data/ft3d_data', '{}.pfm'.format(kwargs['fname'])))     #读深度数据
    disp = disp[:(kwargs['crop_H'] + 40), :(kwargs['crop_W'] + 40)]      #剪裁深度数据
    depth = np.clip(1.0 / np.asarray(disp) * kwargs['depth_clip'] * 10, 0.0, kwargs['depth_clip'])
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * (kwargs['depth_clip'] - 1) + 1

    rgb = rgb / depth[20:-20, 20:-20, np.newaxis] ** 2
    return rgb, depth