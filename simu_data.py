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

def channel2to3(output2):
    '''
    output2 [N,2,H,W] [0,1]
    output3 [3,H,W]   [0,255]
    '''
    output3 = np.zeros((3,output2.shape[2], output2.shape[3]))
    output3[0,:] = output2[0,0,:]
    output3[2,:] = output2[0,1,:]
    #output3[0,:] = output2[0,0,:]
    #output3[2,:] = -output2[0,0,:]
    #output3[output3<0]=0
    
    #regularization = math.log((self.args.log_eps+255)/self.args.log_eps)/self.args.minimum_threshold
    output3[output3>0.1] = output3[output3>0.1]+125
    output3[output3>255]=255
    #max_value = output3.max()
    
    #return np.trunc(output3/max_value*255)
    return np.trunc(output3)

def readconfig(file):
    index = 0
    info = {'size': (0,0), 'tmax': 0, 'background_path': '', 'background_params': [], 'foreground_path': [], 'foreground_params': []}
    with open(file, 'r') as f:
        for line in f:
            tmp = line.strip().split(' ')
            if index == 0:
                info['size'] = (int(tmp[0]), int(tmp[1]))
                info['tmax'] = float(tmp[2])
            elif index == 1:
                info['background_path'] = tmp[0]
                info['background_params'] = [float(tmp[i+1]) for i in range(12)]
            else:
                info['foreground_path'].append(tmp[0])
                info['foreground_params'].append([float(tmp[i+1]) for i in range(12)])
            index += 1
        f.close()
    return info

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    desPath = '/media/localdisk1/usr/gdx/project/20200721_ED/ED/Data/NEW/simu'
    if not os.path.exists(desPath):
        os.mkdir(desPath)
        
    #images_list = []
    #with open(args.data_images, 'r') as f:
    #    for line in f:
    #        images_list.append(line.strip())
    #random.shuffle(images_list)
    
    scene_path = '/media/localdisk1/usr/gdx/project/20200721_ED/esim_config_generator/data/configs2'
    scene_list = os.listdir(scene_path)
    scene_list = [os.path.join(scene_path, scenefile) for scenefile in scene_list if '_autoscene' in scenefile]
    random.shuffle(scene_list)
    
    epochTime = []

    num_data = 20
    #len_seq = 2000
    #delta_time = 0.005
    #delta_num = 5
    
    for step, scene in enumerate(scene_list):
        info = readconfig(scene_list[step])
        #info = readconfig('/media/localdisk1/usr/gdx/project/20200721_ED/esim_config_generator/data/configs2/000000280_autoscene.txt')
        #if len(info['foreground_path']) > 5:
        flow_generator = FlowGenerator(args, info)
        for index in range(num_data):
            startTime = time.time()
            #flow_generator.setIter(index*len_seq/num_data)
            img = torch.zeros((1,11,info['size'][0],info['size'][1])).to(device)
            for i in range(0,11):
                image, flow, time_stamp = flow_generator.getImg()
                #imageio.imwrite(os.path.join(desPath,'Img',str(index*10+i).zfill(5)+'.png'), ((image[0].cpu().data.numpy()).transpose(1,2,0).astype(np.uint8)))
                img[:,i:i+1]=image
            np.save(os.path.join(desPath, scene_list[step].split('/')[-1][:-14]+'_'+str(index)+'.npy'), img.cpu().data.numpy())
            epochTime.append(time.time() - startTime)
            print('INFO %s: %d[%d/%d] fps-1: %.4fs'%(
                scene_list[step].split('/')[-1][:-14], (step+1), index, num_data, epochTime[-1]))
                #print(num_pos,num_neg,img_cnt, flow_cnt)
                #break
        #print(len(info['foreground_path']))
        #break
        
    averageEpochTime = sum(epochTime) / len(epochTime)
    print('INFO num: %d fps-1: %.4fs'%(num_data, averageEpochTime))
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = InferenceOptions().parse()
    main(parser)


