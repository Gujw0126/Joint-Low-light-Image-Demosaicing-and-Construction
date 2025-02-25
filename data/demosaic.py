import numpy as np
import rawpy
import os
import imageio

class get_raw:
    def __init__(self,data_root):
        self.data_root = data_root
        self.file_list = os.listdir(data_root)

    def convert_to_RGB(self):
        new_root = self.data_root.replace('long','long_rgb')
        for i in range(len(self.file_list)):
            with rawpy.imread(os.path.join(self.data_root, self.file_list[i])) as raw:
                im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
                new_name = self.file_list[i].replace('ARW', 'png')
                imageio.imsave(os.path.join(new_root, new_name), im)
'''

    def __getitem__(self, item):
        with rawpy.imread(os.path.join(self.data_root,self.file_list[item])) as raw:
            im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
            imageio.imsave('try.png', im*255)
            
            raw_visible = raw.raw_image_visible
            raw_visible = (raw_visible-512.0)/(15360-512)    #归一化
            red = np.zeros((raw_visible.shape[0], raw_visible.shape[1]), dtype=float)
            blue = red.copy()
            red[0::2, 0::2] = raw_visible[0::2, 0::2]
            blue[1::2, 1::2] = raw_visible[1::2, 1::2]
            green = raw_visible - red - blue
        return red, green, blue


    def __len__(self):
        return len(self.file_list)
'''




def HLI(color,x):
    '''
    :param color: R:输入是R通道  G：输入是G   B:输入是b
    :param x: 待扩展
    :return: 行插值后的矩阵
    '''
    result = x.copy()
    if (color=='R'):
        for i in range(1,x.shape[1]-1, 2) :
            result[:,i] = (x[:,i-1]+x[:,i+1])/2
        result[:,x.shape[1]-1] = x[:,x.shape[1]-2]
    elif (color=='GR'):
        result = np.zeros((x.shape[0],x.shape[1]),dtype=float)
        result[0::2,0] = x[0::2,1]
        for i in range(2,x.shape[1],2):
            result[0::2,i] = (x[0::2,i-1]+x[0::2,i+1])
    elif(color=='GB'):
        result = np.zeros((x.shape[0], x.shape[1]), dtype=float)
        for i in range(1,x.shape[1]-1, 2) :
            result[1::2,i] = (x[1::2,i-1]+x[1::2,i+1])/2
        result[1::2,x.shape[1]-1] = x[1::2,x.shape[1]-2]
    elif (color=='B'):
        result[:,0] = x[:,1]
        for i in range(2,x.shape[1],2):
            result[:,i] = (x[:,i-1]+x[:,i+1])
    else:
        raise NotImplemented





class H_RI:
    pass


class H_MLRI:
    pass


class V_MLRI:
    pass


class V_MLRI:
    pass
'''
class ARI:
    def __init__(self):
        self.hori_RI =
        self.hori_MLRI =
        self.ver_RI =
        self.ver_MLRI =
'''


def test():
    dataset = get_raw('G:/SID/data/Sony/long')
    dataset.convert_to_RGB()

if __name__ =='__main__':
    test()