#'''
import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from Options.inferenceOptions import InferenceOptions
from torch.autograd import Variable
from torch.utils.data import DataLoader
from DataLoader.transform import MyTransform
from DataLoader.dataset import MyDataSet
from NetWorks import getModel, loadParameter
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize

from NetWorks.flow_generator import FlowGenerator
from utils.event_packagers import *
import imageio
import random

from utils.util import ensure_dir, flow2bgr_np

def read_h5_img(hdf_path,img_idx):
    """
    Read img from HDF5 file. Return img
    """
    f = h5py.File(hdf_path, 'r')
    #f['images/image{:09d}'.format(img_idx)]
    return f['images/image{:09d}'.format(img_idx)]
    
def read_h5_flo(hdf_path,img_idx):
    """
    Read img from HDF5 file. Return img
    """
    f = h5py.File(hdf_path, 'r')
    #f['flow/flow{:09d}'.format(img_idx)]
    return f['flow/flow{:09d}'.format(img_idx)]
    
def resample(img, flow3):
        t_width, t_height = img.shape[3], img.shape[2]
        xx, yy = torch.meshgrid(torch.arange(t_width), torch.arange(t_height))  # xx, yy -> WxH
        #xx, yy = xx.to(image0.device), yy.to(image0.device)
        xx = xx.to(img.device)
        yy = yy.to(img.device)
        xx.transpose_(0, 1)
        yy.transpose_(0, 1)
        xx, yy = xx.float(), yy.float()

        flow01_x = flow3[:, 0, :, :]  # N x H x W
        flow01_y = flow3[:, 1, :, :]  # N x H x W

        warping_grid_x = xx + flow01_x  # N x H x W
        warping_grid_y = yy + flow01_y  # N x H x W

        # normalize warping grid to [-1,1]
        warping_grid_x = (2 * warping_grid_x / (t_width - 1)) - 1
        warping_grid_y = (2 * warping_grid_y / (t_height - 1)) - 1

        warping_grid = torch.stack(
            [warping_grid_x, warping_grid_y], dim=3)  # 1 x H x W x 2

        #return warping_grid
        return F.grid_sample(img, warping_grid)

def main(result_path, event_file):
    
    num_data = 300
    len_seq = 2000
    delta_time = 0.005
    delta_num = 4

    for index in range(num_data):
        print(index)
        img = read_h5_img(event_file,index)
        flow = read_h5_flo(event_file,index)
        img = torch.from_numpy(np.array(img).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        flow = torch.from_numpy(np.array(flow).astype(np.float32)).unsqueeze(0)
        if index == 0:
            img2 = img
        print(flow)
        imageio.imwrite(os.path.join(result_path,'Img',str(index).zfill(5)+'.png'), (img[0].cpu().data.numpy()).transpose(1,2,0).astype(np.uint8))
        imageio.imwrite(os.path.join(result_path,'Img2',str(index).zfill(5)+'.png'), (img2[0].cpu().data.numpy()).transpose(1,2,0).astype(np.uint8))
        img2 = resample(img2, -flow*delta_time*delta_num)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    result_path = '/media/localdisk1/usr/gdx/project/20200721_ED/event_cnn_minimal/test/reconstruction1/'
    #event_file = '/media/localdisk1/usr/gdx/project/20200721_ED/ED/Results/val/000000552775.h5'
    event_file = '/media/localdisk1/usr/gdx/project/20200721_ED/event_cnn_minimal/test/h5_events/000000000_out.h5'
    main(result_path, event_file)
#'''
'''
import torch
a = torch.tensor([[[0.2,0.2],[0.2,0.2]]],requires_grad=True, dtype=torch.float32)
e = torch.tensor([[[1.1,1.2],[1.1,1.2]],[[5.1,6.2],[7.1,8.2]]], requires_grad=False, dtype=torch.float32)
#b = a.detach()
b = a.clone().detach()
c = (e //b*b)
#*c.frac()*c.frac()+c.trunc()
#d = c.trunc()
f = c/a
d = f[:,1:2]-f[:,0:1]
g = d-((d>0)*(e[:,1:2]>0)).float()
#print(g)
g.mean().backward()
print(a.grad, g)
#'''
'''
import numpy as np
import os

file_txt = './Data/VTE/event.txt'
file_name_list = []

pos_m = []
pos_s = []

neg_m = []
neg_s = []


with open(file_txt, 'r') as f:
    for line in f:
        file_name_list.append(line.strip())
for file_name in file_name_list:
    print(file_name)
    map = np.load(file_name)
    pos_m.append(map[0].mean())
    pos_s.append(map[0].std())
    neg_m.append(map[1].mean())
    neg_s.append(map[1].std())
print(np.mean(pos_m),np.mean(pos_s), np.mean(neg_m), np.mean(neg_s))
'''