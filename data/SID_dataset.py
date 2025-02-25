import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import sys
import rawpy
import cv2
import imageio
import re   #正则表达式

from data.IO import*
import os

os.environ["OMP_NUM_THREADS"]="1"
torch.set_num_threads(1)

'''
SplitFiles:得到options里面的三个文件
'''
class SplitFiles:
    def __init__(self, train, val, test, data_root):   #train,val,test是比例（小数）
        self.train = train
        self.val = val
        self.test = test
        self.data_root = data_root
        self.raw_list = []

    def Read_Raw(self):
        self.raw_list = os.listdir(os.path.join(self.data_root, 'short'))

    def Split(self,save):
        self.Read_Raw()
        length = len(self.raw_list)
        list = np.arange(0,length)
        train_list =  np.random.choice(list, int(length*self.train), replace=False)
        val_list = np.random.choice((np.setdiff1d(list,train_list)), int(length*self.val), replace=False)
        test_list = np.setdiff1d(np.setdiff1d(list,train_list), val_list)
        with open(os.path.join(save,'train.txt'),'w') as train_file:
            for i in range(len(train_list)):
                train_file.write(self.raw_list[train_list[i]]+'\n')

        with open(os.path.join(save,'test.txt'),'w') as test_file:
            for i in range(len(test_list)):
                test_file.write(self.raw_list[test_list[i]]+'\n')

        with open(os.path.join(save,'val.txt'),'w') as val_file:
            for i in range(len(val_list)):
                val_file.write(self.raw_list[val_list[i]]+'\n')




class DatasetSID(data.Dataset):
    def __init__(self,opt):
        super(DatasetSID,self).__init__()
        print("Dataset:SID,sony")
        self.opt = opt
        self.data_dir = opt['dataroot_H']
        self.n_channels = opt['n_channels']
        self.crop_size = (opt['patch_size'])
        self.split = opt['split']
        self.split_dir = opt['split_dir']
        self.long_dir = os.path.join(self.data_dir,'long_rgb')
        self.short_dir = os.path.join(self.data_dir,'short')
        self.long_list = os.listdir(self.long_dir)
        self.file_list = []

        if self.split == 'train':
            with open(os.path.join(self.split_dir,'train.txt'),'r') as f:
                self.file_list = f.read().splitlines()
        elif self.split == 'val':  # 读取测试集
            with open(os.path.join(self.split_dir, 'val.txt'), 'r') as f:
                self.file_list = f.read().splitlines()



    def __getitem__(self, item):  #调黑电平
        gt_id = self.file_list[item][0:5]
        flag = False
        for name in self.long_list:   #找到对应的rgb文件
            if name[0:5] == gt_id:
                gt_name = name
                flag = True
                break
        if flag == False:
            raise FileNotFoundError

        gt_exposure = int(gt_name[9:11])
        gt = imageio.imread(os.path.join(self.long_dir, gt_name)).astype(np.float32)
        raw_exposure = float(self.file_list[item][9:-5])
        ratio = min(int(gt_exposure/raw_exposure), 300)
        with rawpy.imread(os.path.join(self.short_dir, self.file_list[item])) as raw:
            im = raw.raw_image_visible.astype(np.float32)
            im = ratio*np.maximum(im - 512, 0) / (raw.camera_white_level_per_channel[0] - 512)

        H = im.shape[0]
        W = im.shape[1]
        xx = np.random.randint(0, H-self.crop_size)
        yy = np.random.randint(0, W-self.crop_size)
        input_patch = im[xx:xx+self.crop_size,yy:yy+self.crop_size]
        gt_patch = gt[xx:xx+self.crop_size,yy:yy+self.crop_size]/255.0

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)

        return {'L': input_patch, 'H': gt_patch}

    def __len__(self):
        return len(self.file_list)


def test():
    opt={'dataroot_H':'G:/SID/data/Sony','n_channels':8,'patch_size':512, 'split':'test','split_dir':'E:/gjw_model/split_files'}
    dataset = DatasetSID(opt)
    for i in range(len(dataset)):
        a = dataset[i]

if __name__ =='__main__':
    test()