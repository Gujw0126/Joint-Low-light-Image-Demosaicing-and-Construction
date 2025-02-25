from __future__ import print_function, division
import os
import torchvision
import torch
#from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import math


class ResUNet(torch.nn.Module):        #和论文中的插图一样
    def __init__(self, in_chans=1, out_chans=3):    #out_chans恢复的RGB图像
        super(ResUNet, self).__init__()
        self.out_chans = out_chans
        self.fre = Frequency_Selection(1, 3)
        self.aspp1 = ASPP(3, 32)     #R1
        self.conv1 = ConvLayer(32, 32, kernel_size=3, stride=1)  #R2
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)     #归一化
        self.aspp2 = ASPP(32, 32)                     #R3
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)   #R4
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.aspp3 = ASPP(64, 64)        #R5
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)   #R6
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.aspp4 = ASPP(128, 128)   #R7
        self.res1 = []
        self.res2 = []
        self.res3 = []
        for ii in range(4):
            self.res1.append(ResidualBlock(32))
        for ii in range(8):
            self.res2.append(ResidualBlock(64))
        for ii in range(16):
            self.res3.append(ResidualBlock(128))
        self.res1 = nn.Sequential(*self.res1)           #res1=residualblock*4  Y1
        self.res2 = nn.Sequential(*self.res2)           #res2=residualblock*8  Y2
        self.res3 = nn.Sequential(*self.res3)           #res3=residualblock*16  Y3

        self.deconv1 = UpsampleConvLayer(128 * 2, 64, kernel_size=3, stride=1, upsample=2)  #R9
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64 * 2, 32, kernel_size=3, stride=1, upsample=2)   #R11
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32 * 2, self.out_chans, kernel_size=9, stride=1)    #R13
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = torch.nn.ReLU()


    def get_pattern_raw(self, X):    #解释见word，是个从rggb图形中恢复raw图形的方法（bayer color filter）
        B, C, H, W = X.shape
        pattern_raw = torch.zeros(B, 1, H * 2, W * 2).to(X.device)     #图形面积扩大四倍，之前图的4个通道展成一个大的
        pattern_raw[:, 0, ::2, ::2] = X[:, -4, :, :]      #G
        pattern_raw[:, 0, 1::2, ::2] = X[:, -3, :, :]     #B
        pattern_raw[:, 0, 1::2, 1::2] = X[:, -2, :, :]       #：：是步长 G
        pattern_raw[:, 0, ::2, 1::2] = X[:, -1, :, :]     #R
        pattern_raw.requires_grad_(requires_grad=False)
        return pattern_raw


    def get_pattern_color(self, pattern_raw):   #上面函数的逆向运算
        return torch.cat((pattern_raw[:, :1, ::2, ::2],
                          pattern_raw[:, :1, 1::2, ::2],
                          pattern_raw[:, :1, 1::2, 1::2],
                          pattern_raw[:, :1, ::2, 1::2]), dim=1)


    def get_pattern_shift(self, X, sx, sy):
        pattern_raw = self.get_pattern_raw(X)               #变换前先取RAW格式
        theta = torch.tensor([[1, 0, 0], [0, 1, 0]])         #没有做什么变换
        theta = theta.repeat(pattern_raw.size()[0], 1, 1)      #每个batch重复一次
        theta = theta.float()
        grid = F.affine_grid(theta, pattern_raw.size()).to(X.device)
        grid.requires_grad_(requires_grad=False)
        sx = torch.repeat_interleave(torch.repeat_interleave(sx, 2, dim=1), 2, dim=2)
        sy = torch.repeat_interleave(torch.repeat_interleave(sy, 2, dim=1), 2, dim=2)
        sx_norm = sx / pattern_raw.shape[2]
        sy_norm = sy / pattern_raw.shape[3]
        grid -= torch.cat((sx_norm.unsqueeze(3), sy_norm.unsqueeze(3)), dim=3)
        pattern_raw_shift = F.grid_sample(pattern_raw, grid)
        return self.get_pattern_color(pattern_raw_shift)

    def forward(self, X):
        in1 = self.fre(X)
        in1 = Transform_to_RGB(in1)
        o1 = self.relu(self.conv1(self.aspp1(in1)))  #G3
        o2 = self.relu(self.conv2(self.aspp2(o1)))  #G5
        o3 = self.relu(self.conv3(self.aspp3(o2)))  #G7
        o1 = self.res1(o1)              #Y1  o1,o2,y是三个复合层跑出来的结果，计算得到三个残差块
        o2 = self.res2(o2)              #Y2
        y = self.res3(self.aspp4(o3))   #Y3
        in1 = torch.cat((y, o3), 1)     #G9
        y = self.relu(self.deconv1(in1))  #R9 得到G10
        in2 = torch.cat((y, o2), 1)       #G11
        y = self.relu(self.deconv2(in2))  #R11得到G12
        in3 = torch.cat((y, o1), 1)       #G13
        y = self.deconv3(in3)               #神经网络都跑完的结果 G14
        #TODO 这个函数后面几行没看
        #y=self.maxpool(y)
        #y = torch.cat((Transform_to_RGB(y[:, 0:3, :, :]), y[:,self.out_chans-2, :, :].unsqueeze(1),
                       #y[:, self.out_chans - 1, :, :].unsqueeze(1)), dim=1)
        #sx = torch.nn.Tanh()(y[:, self.out_chans - 2, :, :]) * self.disp_clip
        #sy = torch.nn.Tanh()(y[:, self.out_chans - 1, :, :]) * self.disp_clip
        #pattern_shift = self.get_pattern_shift(X, sx, sy)

        #y = torch.cat((y[:, :self.out_chans - 2, :, :], pattern_shift,
         #              sx.unsqueeze(1), sy.unsqueeze(1)), dim=1)

        return y


class ConvLayer(torch.nn.Module):     #same卷积
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2        #因为kernel_size绝大部分是奇数，这样镜像填充后做卷积，图片大小不变
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


class ConvLayer_atrous(nn.Module):     #一层conv,一层relu，比普通的conv加了一个扩大
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(ConvLayer_atrous, self).__init__()
        self.atrous_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.atrous_conv(x))
        return x


class ASPP(nn.Module):   #ASPP是一种池化层
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        dilations = [1, 3, 6, 12]
        self.aspp1 = ConvLayer_atrous(in_channels, out_channels, 3, padding=1, dilation=dilations[0])
        self.aspp2 = ConvLayer_atrous(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1])     #棋盘样式的卷积（空洞卷积）
        self.aspp3 = ConvLayer_atrous(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ConvLayer_atrous(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),             #使用1*1的卷积核调整channel
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
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
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)        #插值，输出是输入的2倍
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

class Frequency_Selection(nn.Module):
    def __init__(self,in_channel=1, out_channel=3):
        super(Frequency_Selection, self).__init__()
        self.alpha1_filter = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3)   #same
        self.alpha2_filter = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3)
        self.beta_filter = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        '''
        :param x: CFA data, channel=2 shape: batch, 1, h,w
        :param c_alpha1: carrier of alpha1, type: list[A,w1,w2]
        :return:  alpha,luma, beta
        '''
        H = x.shape[0]
        W = x.shape[1]
        c1 = torch.ones(1, 1, H, W)
        for h in range(H):
            for w in range(W):
                c1[:, :, h, w] = 2 * c1[:, :, h, w] * math.cos(math.pi * h)

        c2 = torch.ones(1, 1, H, W)
        for h in range(H):
            for w in range(W):
                c2[:, :, h, w] = 2 * c2[:, :, h, w] * math.cos(math.pi * w)

        c3 = torch.ones(1, 1, H, W)
        for h in range(H):
            for w in range(W):
                c3[:, :, h, w] = 2 * c3[:, :, h, w] * math.cos(math.pi * h + math.pi * w)

        carrier1 = c1.to(x.device)
        carrier2 = c2.to(x.device)
        carrier3 = c3.to(x.device)
        x_alpha1 = x * carrier1
        x_alpha2 = x * carrier2
        x_beta = x * carrier3

        x_filter1 = self.alpha1_filter(x_alpha1)
        x_filter2 = self.alpha2_filter(x_alpha2)
        x_filter_beta = self.beta_filter(x_beta)

        x_alpha = x_filter1+x_filter2
        x_luma = x - x_filter1 * carrier1 -x_filter2 * carrier2 - x_filter_beta*carrier3

        return torch.cat((x_alpha, x_luma, x_beta), dim=1)


def Transform_to_RGB(input):   #4维：batch,channel=3, h,w
    alpha = input[:, 0, :, :].unsqueeze(1)
    luma = input[:, 1, :, :].unsqueeze(1)
    beta = input[:, 2, :, :].unsqueeze(1)
    r = luma - 2*alpha + beta
    g = luma - beta
    b = luma + 2*alpha + beta
    rgb = torch.cat((r, g, b), dim=1)
    return rgb

def model_test():
    a = torch.randn(1, 1, 512, 512)
    model = ResUNet(1, 3)
    b = model(a)
    print(a.shape)
    print(b.shape)

if __name__ =='__main__':
    model_test()