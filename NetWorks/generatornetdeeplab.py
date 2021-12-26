import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

from torch.utils import model_zoo
from torchvision import models
from .resample2d_package.resample2d import Resample2d

import imageio
from sys import version_info

import torch
import torchvision
import numpy as np
from torch import nn
import torch.nn.functional as F
from sys import version_info

from .resnet_model import *
from .deeplabv3plus import deeplabV3Plus

class EvSegNet(nn.Module):
    def __init__(self, args):
        if(version_info.major == 2):
            super(EvSegNet,self).__init__()
        else:
            super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.minimum_threshold = args.minimum_threshold

        self.debug_mode = args.debug
        self.resizeShape = args.size
        self.args = args

        self.d3p = deeplabV3Plus()

    def ln_map(self, map):
        new_map = map.clone()
        new_map[map < self.args.log_threshold] = map[map < self.args.log_threshold]/self.args.log_threshold*math.log(self.args.log_threshold)
        new_map[map >= self.args.log_threshold] = torch.log(map[map >= self.args.log_threshold])
        #new_map = torch.log(self.args.log_eps + map/255.0)
        return new_map
        
    def inferenceBefore(self, images, deltaTime):
        x = F.interpolate(images, self.args.size, mode='bilinear')
        
        normal_mean, normal_std = self.d3p(x)

        normal_sample_pos = torch.randn(normal_mean.size()).to(self.device)
        normal_sample_neg = torch.randn(normal_mean.size()).to(self.device)
        threshold_C_pos_origin = normal_sample_pos * normal_std + normal_mean
        threshold_C_neg_origin = normal_sample_neg * normal_std + normal_mean
        self.threshold_C_pos = F.relu(threshold_C_pos_origin - self.minimum_threshold) + self.minimum_threshold
        self.threshold_C_neg = F.relu(threshold_C_neg_origin - self.minimum_threshold) + self.minimum_threshold

        self.deltaTime = deltaTime
        self.cutoff_hz = 30
        self.shot_noise_rate_hz = 0.1
        
        images_origin = (images*0.226+0.449)*255
        self.irradiance_maps_0 = self.ln_map(images_origin)
        #random_sample = (torch.rand(normal_mean.size()).to(self.device)*self.threshold_C_pos).to(self.device)
        #self.irradiance_maps_0 -= random_sample
        
        self.lpLogFrame1 = self.irradiance_maps_0
        self.lpLogFrame0 = self.irradiance_maps_0
        
    def inference(self, images):
        logNewFrame = self.ln_map(images)

        irradiance_values_pos = F.relu(logNewFrame-self.irradiance_maps_0) // self.threshold_C_pos
        irradiance_values_neg = F.relu(-(self.lpLogFrame1-self.irradiance_maps_0)) // self.threshold_C_neg
        
        self.irradiance_maps_0 += (irradiance_values_pos * self.threshold_C_pos - irradiance_values_neg * self.threshold_C_neg)
        '''
        logNewFrame = self.ln_map(images)
        inten01 = (images + 20)/275 # limit max time constant to ~1/10 of white intensity level
        tau = (1 / (np.pi * 2 * self.cutoff_hz))
        eps = inten01 * (self.deltaTime / tau)
        eps[eps[:] > 1] = 1  # keep filter stable
        self.lpLogFrame0 = (1-eps)*self.lpLogFrame0+eps*logNewFrame
        self.lpLogFrame1 = (1-eps)*self.lpLogFrame1+eps*self.lpLogFrame0
        self.lpLogFrame1 = logNewFrame

        self.irradiance_maps_0 = self.irradiance_maps_0-0.1*self.deltaTime*0.2

        irradiance_values_pos = F.relu(self.lpLogFrame1-self.irradiance_maps_0) // self.threshold_C_pos
        irradiance_values_neg = F.relu(-(self.lpLogFrame1-self.irradiance_maps_0)) // self.threshold_C_neg
        
        self.irradiance_maps_0 += (irradiance_values_pos * self.threshold_C_pos - irradiance_values_neg * self.threshold_C_neg)
        
        num_iters = int(max(irradiance_values_pos.max(),irradiance_values_neg.max()).item())
        #print(num_iters)
        #print('output', output1.max().item(),-output1.min().item(),F.relu(output1).sum().item(), F.relu(-output1).sum().item())
        for j in range(0,num_iters):
            SHOT_NOISE_INTEN_FACTOR = 0.25
            shotNoiseFactor = (
                (self.shot_noise_rate_hz/2)*self.deltaTime/num_iters) * \
                ((SHOT_NOISE_INTEN_FACTOR-1)*inten01+1)
            # =1 for inten=0 and SHOT_NOISE_INTEN_FACTOR for inten=1

            rand01 = torch.rand(self.irradiance_maps_0.size()).to(self.device)  # draw samples
            # probability for each pixel is
            # dt*rate*nom_thres/actual_thres.
            # That way, the smaller the threshold,
            # the larger the rate
            shotOnProbThisSample = shotNoiseFactor*0.2/self.threshold_C_pos
            # array with True where ON noise event
            shotOnCord = torch.zeros_like(self.irradiance_maps_0).to(self.device)
            shotOnCord[rand01 > (1-shotOnProbThisSample)] = 1
            
            shotOffProbThisSample = shotNoiseFactor*0.2/self.threshold_C_neg
            # array with True where ON noise event
            shotOffCord = torch.zeros_like(self.irradiance_maps_0).to(self.device)
            shotOffCord[rand01 < shotOffProbThisSample] = 1
            self.irradiance_maps_0 += shotOnCord*self.threshold_C_pos
            self.irradiance_maps_0 -= shotOffCord*self.threshold_C_neg
            irradiance_values_pos += shotOnCord
            irradiance_values_neg += shotOffCord
        #'''
        '''
        logNewFrame = self.ln_map(images)
        inten01 = (images + 20)/275 # limit max time constant to ~1/10 of white intensity level
        tau = (1 / (np.pi * 2 * self.cutoff_hz))
        eps = inten01 * (self.deltaTime / tau)
        eps[eps[:] > 1] = 1  # keep filter stable
        self.lpLogFrame0 = (1-eps)*self.lpLogFrame0+eps*logNewFrame
        self.lpLogFrame1 = (1-eps)*self.lpLogFrame1+eps*self.lpLogFrame0

        self.irradiance_maps_0 = self.irradiance_maps_0-0.1*self.deltaTime*0.2

        irradiance_values_pos = F.relu(self.lpLogFrame1-self.irradiance_maps_0) // self.threshold_C_pos
        irradiance_values_neg = F.relu(-(self.lpLogFrame1-self.irradiance_maps_0)) // self.threshold_C_neg
        
        self.irradiance_maps_0 += (irradiance_values_pos * self.threshold_C_pos - irradiance_values_neg * self.threshold_C_neg)
        '''
        return irradiance_values_pos, irradiance_values_neg
    
    def get_th(self, images):
        #x = F.interpolate(images, self.args.size, mode='bilinear')
        x = ((images/255.0)-0.449)/0.226
        normal_mean, normal_std = self.d3p(x)
        #normal_std = torch.exp(normal_std)
        normal_std = normal_std.pow(2)
        #normal_mean = F.interpolate(normal_mean, images.size()[2:4], mode='nearest')
        #normal_std = F.interpolate(normal_std, images.size()[2:4], mode='nearest')
        return {'contrast_threshold_sigma_pos':normal_std,'contrast_threshold_sigma_neg':normal_std,'contrast_threshold_pos':normal_mean,'contrast_threshold_neg':normal_mean}
    #def forward(self, images, optic_flows):
    def forward(self, images):
        #min_x = images.size()[2]//2-self.args.size[0]//2
        #max_x = images.size()[2]//2+self.args.size[0]//2
        #min_y = images.size()[3]//2-self.args.size[1]//2
        #max_y = images.size()[3]//2+self.args.size[1]//2
        
        ###x = images[:,:,min_x:max_x,min_y:max_y]
        #x = ((images[:,images.size()[1]//2:images.size()[1]//2+1,min_x:max_x,min_y:max_y]/255.0)-0.449)/0.226
        x = ((images[:,images.size()[1]//2:images.size()[1]//2+1]/255.0)-0.449)/0.226

        normal_mean, normal_std = self.d3p(x)
        #std = log(sigma^2)
        #normal_std = torch.sqrt(torch.exp(normal_std)+1e-8)
        #simfipy
        #normal_mean = torch.exp(normal_mean)
        #normal_std = torch.exp(normal_std)
        normal_std = normal_std.pow(2)
        
        #normal_mean = normal_mean*0+0.2
        #normal_std = normal_std*0+0.05

        normal_sample_pos = torch.randn(normal_mean.size()).to(self.device)
        normal_sample_neg = torch.randn(normal_mean.size()).to(self.device)
        threshold_C_pos_origin = normal_sample_pos * normal_std + normal_mean
        threshold_C_neg_origin = normal_sample_neg * normal_std + normal_mean
        threshold_C_pos = F.relu(threshold_C_pos_origin - self.minimum_threshold) + self.minimum_threshold
        threshold_C_neg = F.relu(threshold_C_neg_origin - self.minimum_threshold) + self.minimum_threshold

        numframes = 10
        #deltaTime = 0.005*(numframes+1)/(numframes)
        #cutoff_hz = 30
        #shot_noise_rate_hz = 0.1
        ####images_origin = (images*0.226+0.449)*255
        images_origin = images[:,0:1]
        irradiance_maps_0 = self.ln_map(images_origin)

        random_sample = (torch.rand(normal_mean.size())*0.2).to(self.device)
        irradiance_maps_0 -= random_sample
        
        irradiance_maps = torch.zeros((irradiance_maps_0.size()[0],numframes+1,irradiance_maps_0.size()[2],irradiance_maps_0.size()[3])).to(self.device)
        #lpLogFrame1 = irradiance_maps_0
        #lpLogFrame0 = irradiance_maps_0
        
        irradiance_maps[:,1:11] = self.ln_map(images[:,1:11])
        for i in range(1,numframes+1):
            irradiance_maps[:,i:i+1] = irradiance_maps[:,i:i+1]-irradiance_maps_0
        threshold_C_pos_cp = threshold_C_pos.clone().detach()
        irradiance_maps_pos_trunc = F.relu(irradiance_maps) // threshold_C_pos_cp * threshold_C_pos_cp
        irradiance_values_pos = irradiance_maps_pos_trunc / threshold_C_pos
        
        threshold_C_neg_cp = threshold_C_neg.clone().detach()
        irradiance_maps_neg_trunc = F.relu(-irradiance_maps) // threshold_C_neg_cp * threshold_C_neg_cp
        irradiance_values_neg = irradiance_maps_neg_trunc / threshold_C_neg
        
        irradiance_values = irradiance_values_pos - irradiance_values_neg
        #irradiance_maps_trunc = irradiance_values.frac()+irradiance_values.floor()
        diff_img = irradiance_values[:,1:numframes+1] - irradiance_values[:,0:numframes]
        #diff_img[(diff_img>=0) * (irradiance_maps[1:numframes]>0)] += 0
        #diff_img[(diff_img>0) * (irradiance_maps[1:numframes]<=0)] += -1
        #diff_img[(diff_img<=0) * (irradiance_maps[1:numframes]<0)] += 0
        #diff_img[(diff_img<0) * (irradiance_maps[1:numframes]>=0)] += 1
        diff_img=diff_img-((diff_img>0)*(irradiance_maps[:,1:numframes+1]<=0)).float()+((diff_img<0)*(irradiance_maps[:,1:numframes+1]>=0)).float()
        #'''
        output_pos = (F.relu(diff_img)).sum(dim=1,keepdim=True)
        output_neg = (F.relu(-diff_img)).sum(dim=1,keepdim=True)
        output = torch.cat((output_pos, output_neg), 1)
        return images[:,0:1], self.ln_map(images[:,0:1]), self.ln_map(images[:,10:11]), output, normal_mean, normal_std