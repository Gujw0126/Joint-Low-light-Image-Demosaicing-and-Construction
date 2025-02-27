from typing import Any, Union

import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torch import autograd as autograd
#from patterned_models.loss_ssim import SSIMLoss
import math
import pdb
import numpy as np


# --------------------------------------------
# imageLoss 里的Perceptual loss，先VGG变换再平方
# --------------------------------------------
class VGGFeatureExtractor(nn.Module):         #目的是得到vgg网络某些层的特征，并存在列表里
    def __init__(self, feature_layer=[2, 7, 16, 25, 34], use_input_norm=True, use_range_norm=False):  #选择归一化方式，默认为input_norm
        super(VGGFeatureExtractor, self).__init__()
        '''
        use_input_norm: If True, x: [0, 1] --> (x - mean) / std
        use_range_norm: If True, x: [0, 1] --> x: [-1, 1]
        '''
        model = torchvision.models.vgg19(pretrained=True)      #使用预先训练好的vgg19
        self.use_input_norm = use_input_norm
        self.use_range_norm = use_range_norm
        if self.use_input_norm:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.list_outputs = isinstance(feature_layer, list)      #isinstance函数，判断feature_layer是不是list类型的
        if self.list_outputs:                                    #假设feature_layer是list类型
            self.features = nn.Sequential()
            feature_layer = [-1] + feature_layer            #feature_layer+[-1] = [-1, 2, 7, 16, 25, 34]
            for i in range(len(feature_layer) - 1):
                self.features.add_module('child' + str(i), nn.Sequential(
                    *list(model.features.children())[(feature_layer[i] + 1):(feature_layer[i + 1] + 1)]))   #*用于迭代取出内容, 取出0-2，3-7，8-15等层

        else:
            self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])

        print(self.features)

        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_range_norm:
            x = (x + 1.0) / 2.0
        if self.use_input_norm:
            x = (x - self.mean) / self.std                 #两种可选的归一化
        if self.list_outputs:
            output = []
            for child_model in self.features.children():
                x = child_model(x)
                output.append(x.clone())                     #output是5个children的结果
            return output
        else:
            return self.features(x)


class PerceptualLoss(nn.Module):
    """VGG Perceptual loss
    """

    def __init__(self, feature_layer=[2, 7, 16, 25, 34], weights=[0.1, 0.1, 1.0, 1.0, 1.0], lossfn_type='l1',    #weights是每一子层误差的权重
                 use_input_norm=True, use_range_norm=False):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGFeatureExtractor(feature_layer=feature_layer, use_input_norm=use_input_norm,
                                       use_range_norm=use_range_norm)         #vgg一些层的结果（高层特征）
        self.lossfn_type = lossfn_type
        self.weights = weights
        if self.lossfn_type == 'l1':
            self.lossfn = nn.L1Loss()
        else:
            self.lossfn = nn.MSELoss()            #均方损失
        print(f'feature_layer: {feature_layer}  with weights: {weights}')

    def forward(self, x, gt):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x_vgg, gt_vgg = self.vgg(x), self.vgg(gt.detach())         #待处理的图片与真实图片的vgg特征
        loss = 0.0
        if isinstance(x_vgg, list):
            n = len(x_vgg)
            for i in range(n):
                loss += self.weights[i] * self.lossfn(x_vgg[i], gt_vgg[i])
        else:
            loss += self.lossfn(x_vgg, gt_vgg.detach())
        return loss

'''
好像也没用到
# --------------------------------------------
# GAN loss: gan, ragan
# --------------------------------------------
class GANLoss(nn.Module):           #只是根据gan_type使用pytroch提供的几种损失函数计算，不是网络结构
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':    #这一堆判断是根据传入的gan_type确定self.loss用哪个
            self.loss = nn.BCEWithLogitsLoss()         #sigmoid+Binary CrossEntropy
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()                  #平均平方误差损失
        elif self.gan_type == 'wgan':
            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()       #均值

            self.loss = wgan_loss
        elif self.gan_type == 'softplusgan':
            def softplusgan_loss(input, target):
                # target is boolean
                return F.softplus(-input).mean() if target else F.softplus(input).mean()

            self.loss = softplusgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))     #报错

    def get_target_label(self, input, target_is_real):
        if self.gan_type in ['wgan', 'softplusgan']:
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):                           #计算损失函数
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss
'''

# --------------------------------------------
# TV loss
# --------------------------------------------
class TVLoss(nn.Module):           #可以起到平滑图像，去除鬼影，消除噪声的作用
    def __init__(self, tv_loss_weight=1):
        """
        Total variation loss
        https://github.com/jxgu1016/Total_Variation_Loss.pytorch
        Args:
            tv_loss_weight (int):
        """
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):    #x:[batch, channel, hight, weight]
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])      #count_h=c*(h-1)*w
        count_w = self.tensor_size(x[:, :, :, 1:])      #count_w=c*h*(w-1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()   #下减上错位一行
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


# --------------------------------------------
# Charbonnier loss
# --------------------------------------------
class CharbonnierLoss(nn.Module): #一种近似l1的损失函数，但比l1好
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss * 10


# --------------------------------------------
# Gradient loss
# --------------------------------------------
class GradLoss(nn.Module):    #梯度损失
    def __init__(self):
        super(GradLoss, self).__init__()
        pass

    def get_grad(self, img):
        return torch.cat(((img[:, :, 1:, :-1] - img[:, :, :-1, :-1]),
                             (img[:, :, :-1, 1:] - img[:, :, :-1, :-1])), dim=1)  #下一行减上一行，下一列减上一列，离散量的梯度， 在c上连接(6个channel)

    def forward(self, x, y):
        return torch.nn.L1Loss()(self.get_grad(x), self.get_grad(y))       #建立的同时调用


# --------------------------------------------
# Patterned flash reconstruction loss,总的损失函数
# --------------------------------------------
class SID_Loss(nn.Module):
    # Loss used in MFFNet training

    def __init__(self, lam_vgg, lam_edge):  #lam表示lamda, 是各个loss前面的系数
        super(SID_Loss, self).__init__()
        self.lam_vgg = lam_vgg
        self.lam_edge = lam_edge
        self.perceploss = PerceptualLoss()
        self.gradloss = GradLoss()

    def forward(self, x, y):      #x是预测值，y是真实值
        img = x
        img_gt = y
        '''
        loss前三项对应论文中的Limage
        recp 对应Lphoto
        dloss 对应Ldepth
        '''
        loss = torch.nn.MSELoss()(img, img_gt[:, :3, :, :]) + self.lam_vgg * self.perceploss(img, img_gt[:, :3, :, :]) + \
               self.lam_edge * self.gradloss(img, img_gt[:, :3, :, :])
        return loss

"""
FnFLoss_edge_dloss是训练要用的损失函数
"""
# --------------------------------------------
# Patterned flash/no-flash reconstruction loss
# --------------------------------------------
class FnFLoss_edge_dloss(nn.Module):
    # Loss used in MFFNet training

    def __init__(self, lam_vgg, lam_edge, lam_recp, lam_dloss, pattern_boost):
        super(FnFLoss_edge_dloss, self).__init__()
        self.lam_vgg = lam_vgg
        self.lam_edge = lam_edge
        self.lam_recp = lam_recp
        self.lam_dloss = lam_dloss
        self.pattern_boost = pattern_boost
        self.perceploss = PerceptualLoss()
        self.gradloss = GradLoss()

    def forward(self, x, y):
        amb = x[:, :3, :, :]
        amb_gt = y[:, :3, :, :]
        flash = x[:, 3:6, :, :]        #经过flash处理，有光照
        flash_gt = y[:, 3:6, :, :]
        recp = x[:, 6:10, :, :]
        recp_gt = y[:, 6:10, :, :]

        amb_rggb = torch.cat((amb, amb[:, 1:2, :, :]), dim=1)
        flash_rggb = torch.cat((flash, flash[:, 1:2, :, :]), dim=1)
        amb_rggb_gt = torch.cat((amb_gt, amb_gt[:, 1:2, :, :]), dim=1)
        flash_rggb_gt = torch.cat((flash_gt, flash_gt[:, 1:2, :, :]), dim=1)

        d = x[:, 10:12, :, :]
        d_gt = y[:, 10:12, :, :]
        refscale = y[:, 12, :, :]
        power = y[:, 13, :, :]

        loss = torch.nn.MSELoss()(amb, amb_gt) + self.lam_vgg * self.perceploss(amb, amb_gt) + \
               self.lam_edge * self.gradloss(amb, amb_gt) + \
               self.lam_recp * torch.nn.MSELoss()(flash_rggb * recp * self.pattern_boost + amb_rggb * refscale / power, \
                                                  flash_rggb_gt * recp_gt * self.pattern_boost + amb_rggb_gt * refscale / power) + \
               self.lam_dloss * torch.nn.MSELoss()(d, d_gt)
        # self.lam_recp * torch.sqrt(torch.mean((img_rggb*recp - img_gt*recp_gt)**2))
        return loss

'''
一下三个函数暂时没有用

def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.
        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (
            path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.
    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.
    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha: object = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return
'''



