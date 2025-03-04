# Joint-Low-light-Image-Demosaicing-and-Construction
## 1.问题描述  

 由于低光照环境下CMOS相机拍摄时,信号强度弱，噪声占更大的比重，低光照下的图像质量常常无法令人满意。 
 因相机中 Bayer 彩色滤波阵列的使用，低光照下彩色图像的降噪重建还涉及去马赛克的问题，降噪和重建更为困难。于是提出联合弱光下去马赛克和重建模型。
## 2.去马赛克技术

为降低成本，CMOS相机只使用一个 CMOS传感器，去同时捕捉RGB三原色。入射光中的RGB光分离则由彩色滤波矩阵完成。彩色滤波矩阵的使
用，使摄像头获取的信息更少，在每个像素上仅保留了一种颜色的光照强度。至于像素上其他颜
色的信息由后续的图像处理技术进行插值。CFA的使用使彩色CMOS的结构得到了很大程度上的简化，降低了它的价格。

![image_error](https://github.com/Gujw0126/Joint-Low-light-Image-Demosaicing-and-Construction/blob/main/resource/cfa_camera.png)  

以含有RGGB的色彩滤波矩阵为例，CMOS相机在进行像素插值前得到的RAW图像，如下图所示。    

![image_error](https://github.com/Gujw0126/Joint-Low-light-Image-Demosaicing-and-Construction/blob/main/resource/RGGB.png)    

## 3.频率筛选模块  

由于RGGB图像可以看作是α，β，luma三种信号的叠加，且这三种信号和RGB是线性的关系，从RGGB图像中分离这三种信号即可获得RGB图像。因此设计频率筛选模块来筛选
α，β，luma信号。    
![image_error](https://github.com/Gujw0126/Joint-Low-light-Image-Demosaicing-and-Construction/blob/main/resource/fourier.png)   

![image_error](https://github.com/Gujw0126/Joint-Low-light-Image-Demosaicing-and-Construction/blob/main/resource/frequency_selection.png)  

## 4.整体网络架构
在获得了弱光下的RGB信号后，使用带空洞卷积的UNET结构对图像进行重建，整体网络架构图如下图所示。  

![image_error](https://github.com/Gujw0126/Joint-Low-light-Image-Demosaicing-and-Construction/blob/main/resource/network.png)

## 5.结果
重建结果如下图所示  

![image_error](https://github.com/Gujw0126/Joint-Low-light-Image-Demosaicing-and-Construction/blob/main/resource/result.png)  

PSNR和SSIM结果  
![image_error](https://github.com/Gujw0126/Joint-Low-light-Image-Demosaicing-and-Construction/blob/main/resource/psnr.png)