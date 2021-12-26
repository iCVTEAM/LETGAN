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

        resnet = models.resnet34(pretrained=False)
        #resnet.load_state_dict(torch.load('/media/localdisk1/usr/gdx/project/Pytorch/PyTorchModel/resnet34-333f7ec4.pth'), strict=True)
        ## -------------Encoder--------------

        self.inconv = nn.Sequential(nn.Conv2d(1,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        
        #stage 1
        self.encoder1 = nn.Sequential(BasicBlock(64,64)) #224

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)
        
        #stage 2
        self.encoder2 = nn.Sequential(BasicBlock(64,64),
            BasicBlock(64,64)) #112

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        #stage 3
        self.encoder3 = resnet.layer1 #56
        #stage 4
        self.encoder4 = resnet.layer2 #28
        #stage 5
        self.encoder5 = resnet.layer3 #14
        #stage 6
        self.encoder6 = resnet.layer4 #7

        ## -------------Bridge--------------

        #stage Bridge
        self.bridge = nn.Sequential(nn.Conv2d(512,512,3,dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)) # 7

        ## -------------Decoder--------------

        #stage 6d
        self.decoder6_m = nn.Sequential(nn.Conv2d(1024,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,3,dilation=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))# 7

        #stage 5d
        self.decoder5_m = nn.Sequential(nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))# 14

        #stage 4d
        self.decoder4_m = nn.Sequential(nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))# 28

        #stage 3d
        self.decoder3_m = nn.Sequential(nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))# 56

        #stage 2d
        self.decoder2_m = nn.Sequential(nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))# 112

        #stage 1d
        self.decoder1_m = nn.Sequential(nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))# 224
            
        self.final_m = nn.Sequential(nn.Conv2d(64,1,3,padding=1))
        
        #stage 6d
        self.decoder6_s = nn.Sequential(nn.Conv2d(1024,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,3,dilation=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))# 7

        #stage 5d
        self.decoder5_s = nn.Sequential(nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))# 14

        #stage 4d
        self.decoder4_s = nn.Sequential(nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))# 28

        #stage 3d
        self.decoder3_s = nn.Sequential(nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))# 56

        #stage 2d
        self.decoder2_s = nn.Sequential(nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))# 112

        #stage 1d
        self.decoder1_s = nn.Sequential(nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))# 224
            
        self.final_s = nn.Sequential(nn.Conv2d(64,1,3,padding=1),)
        '''
        #stage 6d
        self.decoder6_e = nn.Sequential(nn.Conv2d(1024,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,3,dilation=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))# 7

        #stage 5d
        self.decoder5_e = nn.Sequential(nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))# 14

        #stage 4d
        self.decoder4_e = nn.Sequential(nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))# 28

        #stage 3d
        self.decoder3_e = nn.Sequential(nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))# 56

        #stage 2d
        self.decoder2_e = nn.Sequential(nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'))# 112

        #stage 1d
        self.decoder1_e = nn.Sequential(nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))# 224
            
        self.final_e = nn.Conv2d(64,1,3,padding=1)
        '''
        self.resample2d = Resample2d()
        
        self.t_m = nn.Conv2d(512,1,3,padding=1)
        self.t_pool_m = nn.AdaptiveAvgPool2d((1,1))
        #self.final_pool_s = nn.AdaptiveMaxPool2d((1,1))
    
    def forward2(self, images, irradiance_maps, optic_flows, image_path):
        #min_x = irradiance_maps.size()[2]//2-self.args.size[0]//2
        #max_x = irradiance_maps.size()[2]//2+self.args.size[0]//2
        #min_y = irradiance_maps.size()[3]//2-self.args.size[1]//2
        #max_y = irradiance_maps.size()[3]//2+self.args.size[1]//2
        image_name = image_path.split('/')[-1]
        folder_name = image_name[:-4]
        os.mkdir(os.path.join('./Data/V2E/Img',folder_name))
        #imageio.imwrite('00000.png', images[0,:,min_x:max_x,min_y:max_y].squeeze().cpu().data.numpy())
        np.save(os.path.join('./Data/V2E/Flo',folder_name+'.npy'), optic_flows[0].cpu().data.numpy())
        frame_num = 10
        for i in range(0,frame_num+1):
            resample = self.resample2d(images, optic_flows*i/frame_num)
            imageio.imwrite(os.path.join('./Data/V2E/Img',folder_name,str(i).zfill(5)+'.png'), ((resample[0].cpu().data.numpy()*np.array([[[.226]]])+np.array([[[.449]]]))*255).transpose(1,2,0).astype(np.uint8))
            #np.save(os.path.join('./Data/VTE/Img',folder_name,str(i)+'.npy'), ((resample[0,:,min_x:max_x,min_y:max_y].cpu().data.numpy()*np.array([[[.226]]])+np.array([[[.449]]]))*255).transpose(1,2,0).astype(np.uint8))
    def ln_map(self, map):
        new_map = map.clone()
        new_map[map < self.args.log_threshold] = map[map < self.args.log_threshold]/self.args.log_threshold*math.log(self.args.log_threshold)
        new_map[map >= self.args.log_threshold] = torch.log(map[map >= self.args.log_threshold])
        #new_map = torch.log(self.args.log_eps + map/255.0)
        return new_map
        
    def inferenceBefore(self, images, deltaTime):
        x = F.interpolate(images, self.args.size, mode='bilinear')
        ## -------------Encoder-------------
        x = self.inconv(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        enc6 = self.encoder6(enc5)
        ## -------------Bridge-------------
        hbg = self.bridge(enc6)
        #normal_mean = self.t_pool_m(self.t_m(hbg))
        #'''
        ## -------------Decoder-------------
        dec6_m = self.decoder6_m(torch.cat((hbg,enc6),1))
        dec5_m = self.decoder5_m(torch.cat((dec6_m,enc5),1))
        dec4_m = self.decoder4_m(torch.cat((dec5_m,enc4),1))
        dec3_m = self.decoder3_m(torch.cat((dec4_m,enc3),1))
        dec2_m = self.decoder2_m(torch.cat((dec3_m,enc2),1))
        dec1_m = self.decoder1_m(torch.cat((dec2_m,enc1),1))
        normal_mean = self.final_m(dec1_m)
        normal_mean = F.interpolate(normal_mean, (self.args.reshape_size[1], self.args.reshape_size[0]), mode='nearest')
        
        dec6_s = self.decoder6_s(torch.cat((hbg,enc6),1))
        dec5_s = self.decoder5_s(torch.cat((dec6_s,enc5),1))
        dec4_s = self.decoder4_s(torch.cat((dec5_s,enc4),1))
        dec3_s = self.decoder3_s(torch.cat((dec4_s,enc3),1))
        dec2_s = self.decoder2_s(torch.cat((dec3_s,enc2),1))
        dec1_s = self.decoder1_s(torch.cat((dec2_s,enc1),1))
        normal_std = self.final_s(dec1_s)
        normal_std = F.interpolate(normal_std, (self.args.reshape_size[1], self.args.reshape_size[0]), mode='nearest')
        #'''

        normal_sample_pos = torch.randn(normal_mean.size()).to(self.device)
        normal_sample_neg = torch.randn(normal_mean.size()).to(self.device)
        threshold_C_pos_origin = normal_sample_pos * normal_std + normal_mean
        threshold_C_neg_origin = normal_sample_neg * normal_std + normal_mean
        #threshold_C_pos_origin = normal_sample_pos * 0.05 + 0.2
        #threshold_C_neg_origin = normal_sample_neg * 0.05 + 0.2
        real_normal_sample = torch.randn(normal_mean.size()).to(self.device)
        threshold_C_pos_origin = real_normal_sample * 0.0001 + 0.2
        threshold_C_neg_origin = real_normal_sample * 0.0001 + 0.224353937718
        #print('threshold mean,std:',normal_mean.mean().item(), normal_std.mean().item())
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
        ## -------------Encoder-------------
        x = self.inconv(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        enc6 = self.encoder6(enc5)
        ## -------------Bridge-------------
        hbg = self.bridge(enc6)
        #normal_mean = self.t_pool_m(self.t_m(hbg))
        #'''
        ## -------------Decoder-------------
        dec6_m = self.decoder6_m(torch.cat((hbg,enc6),1))
        dec5_m = self.decoder5_m(torch.cat((dec6_m,enc5),1))
        dec4_m = self.decoder4_m(torch.cat((dec5_m,enc4),1))
        dec3_m = self.decoder3_m(torch.cat((dec4_m,enc3),1))
        dec2_m = self.decoder2_m(torch.cat((dec3_m,enc2),1))
        dec1_m = self.decoder1_m(torch.cat((dec2_m,enc1),1))
        normal_mean = self.final_m(dec1_m)
        #normal_mean = torch.sqrt(torch.exp(normal_mean)+1e-8)
        
        dec6_s = self.decoder6_s(torch.cat((hbg,enc6),1))
        dec5_s = self.decoder5_s(torch.cat((dec6_s,enc5),1))
        dec4_s = self.decoder4_s(torch.cat((dec5_s,enc4),1))
        dec3_s = self.decoder3_s(torch.cat((dec4_s,enc3),1))
        dec2_s = self.decoder2_s(torch.cat((dec3_s,enc2),1))
        dec1_s = self.decoder1_s(torch.cat((dec2_s,enc1),1))
        normal_std = self.final_s(dec1_s)
        #'''
        #std = log(sigma^2)
        normal_std = torch.sqrt(torch.exp(normal_std)+1e-8)
        #normal_mean = F.interpolate(normal_mean, images.size()[2:4], mode='nearest')
        #normal_std = F.interpolate(normal_std, images.size()[2:4], mode='nearest')
        return {'contrast_threshold_sigma_pos':normal_std,'contrast_threshold_sigma_neg':normal_std,'contrast_threshold_pos':normal_mean,'contrast_threshold_neg':normal_mean}
    
    #def forward(self, images, optic_flows):
    def forward(self, images, real):
        #min_x = images.size()[2]//2-self.args.size[0]//2
        #max_x = images.size()[2]//2+self.args.size[0]//2
        #min_y = images.size()[3]//2-self.args.size[1]//2
        #max_y = images.size()[3]//2+self.args.size[1]//2
        
        ###x = images[:,:,min_x:max_x,min_y:max_y]
        #x = ((images[:,images.size()[1]//2:images.size()[1]//2+1,min_x:max_x,min_y:max_y]/255.0)-0.449)/0.226
        if real == 0:
            x = ((images[:,images.size()[1]//2:images.size()[1]//2+1]/255.0)-0.449)/0.226

            ## -------------Encoder-------------
            x = self.inconv(x)
            enc1 = self.encoder1(x)
            enc2 = self.encoder2(self.pool1(enc1))
            enc3 = self.encoder3(self.pool2(enc2))
            enc4 = self.encoder4(enc3)
            enc5 = self.encoder5(enc4)
            enc6 = self.encoder6(enc5)
            ## -------------Bridge-------------
            hbg = self.bridge(enc6)
            #normal_mean = self.t_pool_m(self.t_m(hbg))
            #'''
            ## -------------Decoder-------------
            dec6_m = self.decoder6_m(torch.cat((hbg,enc6),1))
            dec5_m = self.decoder5_m(torch.cat((dec6_m,enc5),1))
            dec4_m = self.decoder4_m(torch.cat((dec5_m,enc4),1))
            dec3_m = self.decoder3_m(torch.cat((dec4_m,enc3),1))
            dec2_m = self.decoder2_m(torch.cat((dec3_m,enc2),1))
            dec1_m = self.decoder1_m(torch.cat((dec2_m,enc1),1))
            normal_mean = self.final_m(dec1_m)
            #normal_mean = torch.sqrt(torch.exp(normal_mean)+1e-8)
            
            dec6_s = self.decoder6_s(torch.cat((hbg,enc6),1))
            dec5_s = self.decoder5_s(torch.cat((dec6_s,enc5),1))
            dec4_s = self.decoder4_s(torch.cat((dec5_s,enc4),1))
            dec3_s = self.decoder3_s(torch.cat((dec4_s,enc3),1))
            dec2_s = self.decoder2_s(torch.cat((dec3_s,enc2),1))
            dec1_s = self.decoder1_s(torch.cat((dec2_s,enc1),1))
            normal_std = self.final_s(dec1_s)
            #'''
            #std = log(sigma^2)
            normal_std = torch.sqrt(torch.exp(normal_std)+1e-8)
            #normal_std = normal_std.pow(2)
            
            normal_sample_pos = torch.randn(normal_mean.size()).to(self.device)
            normal_sample_neg = torch.randn(normal_mean.size()).to(self.device)
            threshold_C_pos_origin = normal_sample_pos * normal_std + normal_mean
            threshold_C_neg_origin = normal_sample_neg * normal_std + normal_mean
        
        if real == 1:
            normal_sample_pos = torch.randn([1,1,160,160]).to(self.device)
            normal_sample_neg = torch.randn([1,1,160,160]).to(self.device)
            threshold_C_pos_origin = normal_sample_pos * 0.05 + 0.2
            threshold_C_neg_origin = normal_sample_neg * 0.05 + 0.2
            normal_mean = threshold_C_pos_origin
            normal_std = threshold_C_neg_origin
        
        #threshold_C_pos_origin = normal_mean
        #threshold_C_neg_origin = normal_mean
        #threshold_C_pos_origin = normal_sample_pos * 0.05 + normal_mean
        #threshold_C_neg_origin = normal_sample_neg * 0.05 + normal_mean
        #threshold_C_pos_origin = normal_sample_pos * 0.05 + 0.2
        #threshold_C_neg_origin = normal_sample_neg * 0.05 + 0.2
        threshold_C_pos = F.relu(threshold_C_pos_origin - self.minimum_threshold) + self.minimum_threshold
        threshold_C_neg = F.relu(threshold_C_neg_origin - self.minimum_threshold) + self.minimum_threshold

        numframes = 10
        #deltaTime = 0.005*(numframes+1)/(numframes)
        #cutoff_hz = 30
        #shot_noise_rate_hz = 0.1
        ####images_origin = (images*0.226+0.449)*255
        images_origin = images[:,0:1]
        irradiance_maps_0 = self.ln_map(images_origin)

        random_sample = (torch.rand([1,1,160,160])*0.2).to(self.device)
        irradiance_maps_0 -= random_sample
        
        #images_next = self.resample2d(images_origin, optic_flows*9.0/numframes)[:,:,min_x:max_x,min_y:max_y]
        #irradiance_maps = self.ln_map(images_next)
        #diff_img = (irradiance_maps+8*0.1*deltaTime*0.2-irradiance_maps_0) / threshold_C_pos
        #'''
        irradiance_maps = torch.zeros((irradiance_maps_0.size()[0],numframes+1,irradiance_maps_0.size()[2],irradiance_maps_0.size()[3])).to(self.device)
        
        #lpLogFrame1 = irradiance_maps_0
        #lpLogFrame0 = irradiance_maps_0
        
        for i in range(1,numframes+1):
            #images_next = self.resample2d(images_origin, optic_flows*i/numframes)[:,:,min_x:max_x,min_y:max_y]
            images_next = images[:,i:i+1]#[:,:,min_x:max_x,min_y:max_y]
            lpLogFrame1 = self.ln_map(images_next)
            #logNewFrame = self.ln_map(images_next)
            #inten01 = (images_next + 20)/275 # limit max time constant to ~1/10 of white intensity level
            #tau = (1 / (np.pi * 2 * cutoff_hz))
            #eps = inten01 * (deltaTime / tau)
            #print(images_next[0,0,44,44].item(),logNewFrame[0,0,44,44].item(),inten01[0,0,44,44].item())
            #print(deltaTime,tau)
            #eps = inten01 * 1.03672558
            #eps[eps[:] > 1] = 1  # keep filter stable
            #lpLogFrame0 = (1-eps)*lpLogFrame0+eps*logNewFrame
            #lpLogFrame1 = (1-eps)*lpLogFrame1+eps*lpLogFrame0

            #irradiance_maps[:,i:i+1] = lpLogFrame1+(i-1)*0.1*deltaTime*0.2 -irradiance_maps_0
            
            irradiance_maps[:,i:i+1] = lpLogFrame1-irradiance_maps_0
        #print(irradiance_maps[0,:,44,44].cpu().data.numpy()+irradiance_maps_0[0,0,44,44].item())
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
        #print(output_pos.max().item(),output_neg.max().item())
        #print('output', output_pos.sum(), output_neg.sum())
        #output = diff_img.sum(dim=1,keepdim=True)
        output = torch.cat((output_pos, output_neg), 1)
        #print(output)
        #print(output.size())
        #return images, irradiance_maps, resample, output, threshold_C_pos, threshold_C_neg
        #return images[:,:,min_x:max_x,min_y:max_y], irradiance_maps_0, images_next, output, threshold_C_pos, threshold_C_neg
        #return ((images[:,images.size()[1]//2:images.size()[1]//2+1,min_x:max_x,min_y:max_y]/255.0)-0.449)/0.226, images[:,0:1,min_x:max_x,min_y:max_y], logNewFrame, output, threshold_C_pos, threshold_C_neg
        #irradiance_maps, simu_images, simu_images2, simulated_event_maps, fake_threshold_C_pos, fake_threshold_C_neg
        return images[:,0:1], self.ln_map(images[:,0:1]), self.ln_map(images[:,10:11]), output, normal_mean, normal_std
        
    def forward3(self, images, optic_flows):
        #optic_flows = optic_flows*0.9#forward2!!!!!!!
        #resample = self.resample2d(irradiance_maps, optic_flows)
        #images_next = self.resample2d(images, optic_flows)
        #print(irradiance_maps[0,0,23,24])
        #print(resample[0,0,23,24])
        #print(images.size(),irradiance_maps.size(),resample.size())
        #print('real after 1',resample[0,0,56,24])

        min_x = images.size()[2]//2-self.args.size[0]//2
        max_x = images.size()[2]//2+self.args.size[0]//2
        min_y = images.size()[3]//2-self.args.size[1]//2
        max_y = images.size()[3]//2+self.args.size[1]//2
        #print(min_x,min_y)
        #images = images[:,:,min_x:max_x,min_y:max_y]
        #irradiance_maps = irradiance_maps[:,:,min_x:max_x,min_y:max_y]
        #resample = resample[:,:,min_x:max_x,min_y:max_y]
        #images_next = images_next[:,:,min_x:max_x,min_y:max_y]
        #print(irradiance_maps.size(), resample.size())
        
        
        x = images[:,:,min_x:max_x,min_y:max_y]

        ## -------------Encoder-------------
        x = self.inconv(x)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        enc6 = self.encoder6(enc5)
        ## -------------Bridge-------------
        hbg = self.bridge(enc6)
        ## -------------Decoder-------------
        dec6_m = self.decoder6_m(torch.cat((hbg,enc6),1))
        dec5_m = self.decoder5_m(torch.cat((dec6_m,enc5),1))
        dec4_m = self.decoder4_m(torch.cat((dec5_m,enc4),1))
        dec3_m = self.decoder3_m(torch.cat((dec4_m,enc3),1))
        dec2_m = self.decoder2_m(torch.cat((dec3_m,enc2),1))
        dec1_m = self.decoder1_m(torch.cat((dec2_m,enc1),1))
        normal_mean = self.final_m(dec1_m)
        
        dec6_s = self.decoder6_s(torch.cat((hbg,enc6),1))
        dec5_s = self.decoder5_s(torch.cat((dec6_s,enc5),1))
        dec4_s = self.decoder4_s(torch.cat((dec5_s,enc4),1))
        dec3_s = self.decoder3_s(torch.cat((dec4_s,enc3),1))
        dec2_s = self.decoder2_s(torch.cat((dec3_s,enc2),1))
        dec1_s = self.decoder1_s(torch.cat((dec2_s,enc1),1))
        normal_std = self.final_s(dec1_s)
        '''
        dec6_e = self.decoder6_e(torch.cat((hbg,enc6),1))
        dec5_e = self.decoder5_e(torch.cat((dec6_e,enc5),1))
        dec4_e = self.decoder4_e(torch.cat((dec5_e,enc4),1))
        dec3_e = self.decoder3_e(torch.cat((dec4_e,enc3),1))
        dec2_e = self.decoder2_e(torch.cat((dec3_e,enc2),1))
        dec1_e = self.decoder1_e(torch.cat((dec2_e,enc1),1))
        eps = self.final_e(dec1_e)
        '''
        #mean = normal[:,0,:]
        #std = normal[:,1,:]
        #print(x.size(),normal_mean.size(),normal_std.size())
        #print(a)
        normal_sample = torch.randn(normal_mean.size())
        if self.is_cuda_available:
            normal_sample = normal_sample.cuda()
        #threshold_C_pos = normal_sample[:,0:1,:] * normal[:,1:2,:] + normal[:,0:1,:]
        #threshold_C_neg = normal_sample[:,1:2,:] * normal[:,1:2,:] + normal[:,0:1,:]
        threshold_C_pos = normal_sample * normal_std + normal_mean
        threshold_C_neg = normal_sample * normal_std + normal_mean
        #threshold_C_pos = normal_sample * 0.03 + normal_mean
        #threshold_C_neg = normal_sample * 0.03 + normal_std
        #threshold_C_pos = normal_sample * 0.05 + 0.2
        #threshold_C_neg = normal_sample * 0.05 + 0.2
        #print(threshold_C_pos[0,0,23,24])
        #print(threshold_C_neg[0,0,23,24])
        threshold_C_pos = F.relu(threshold_C_pos - self.minimum_threshold) + self.minimum_threshold
        #threshold_C_neg = F.relu(threshold_C_neg - self.minimum_threshold) + self.minimum_threshold
        #print(threshold_C_pos[0,0,23,24])
        #print(threshold_C_neg[0,0,23,24])
        #print(irradiance_maps.size())
        #print(optic_flows.size())
        '''
        irradiance_maps_leak = F.relu(irradiance_maps-0.015) #leak noise 0.1 duration time 0.03s
        eps = images_next*0.1278+0.31045
        #eps = F.sigmoid(eps)
        #eps[eps[:] > 1] = 1  # keep filter stable
        resample_filter = (1-eps)*irradiance_maps+eps*resample
        '''
        '''
        deltaTime = 0.003
        cutoff_hz = 30
        #irradiance_maps_leak = F.relu(irradiance_maps-0.1*deltaTime*10/0.2) #leak noise 0.1 duration time 0.03s
        eps = (images_next*0.226+0.449)+0.1
        tau = (1 / (np.pi * 2 * cutoff_hz))
        eps = eps * (deltaTime / tau)
        eps[eps[:] > 1] = 1  # keep filter stable
        resample_filter = (1-eps)*irradiance_maps+eps*resample
        #resample_filter = (1-eps)*irradiance_maps+eps*resample_filter
        #resample_filter = resample
        '''

        #images *= 0
        numframes = 10
        deltaTime = 0.055/(numframes)
        cutoff_hz = 30
        #irradiance_maps_leak = F.relu(irradiance_maps[:,:,min_x:max_x,min_y:max_y]-0.1*deltaTime*10*0.2) #leak noise 0.1 duration time 0.05s
        a = self.resample2d(images, optic_flows*0)[:,:,min_x:max_x,min_y:max_y]
        #a = self.ln_map((a*0.226+0.449)*255)
        #a = self.resample2d(irradiance_maps, optic_flows*0)[:,:,min_x:max_x,min_y:max_y]
        #a = irradiance_maps[:,:,min_x:max_x,min_y:max_y]
        #normal_sample = torch.randn(a.size())
        #if self.is_cuda_available:
        #    normal_sample = normal_sample.cuda()
        #threshold_C_pos = normal_sample * 0.05 + 0.2
        #threshold_C_pos = F.relu(threshold_C_pos - self.minimum_threshold) + self.minimum_threshold
        
        
        random_sample = torch.rand(a.size())*0.2
        if self.is_cuda_available:
            random_sample = random_sample.cuda()
        a -= random_sample
        
        b = torch.zeros((a.size()[0],numframes+1,a.size()[2],a.size()[3])).cuda()
        #b[:,0:1] = a
        #irradiance_maps_leak = a-0.1*deltaTime*9*0.15
        
        lpLogFrame1 = a
        lpLogFrame0 = a
        
        shot_noise_rate_hz = 0.1
        
        #output_pos = torch.zeros_like(a).cuda()
        #output_neg = torch.zeros_like(a).cuda()
        #threshold_C_pos = torch.zeros_like(a).cuda()+0.2
        for i in range(1,numframes+1):
            #'''
            images_next = self.resample2d(images, optic_flows*i/numframes)[:,:,min_x:max_x,min_y:max_y]
            #print(images_next[0,0,198,198].item())
            images_next = (images_next*0.226+0.449)*255
            #logNewFrame = self.resample2d(irradiance_maps, optic_flows*i/10)[:,:,min_x:max_x,min_y:max_y]
            #print(images_next[0,0,198,198].item())
            inten01 = (images_next + 20)/275 # limit max time constant to ~1/10 of white intensity level
            tau = (1 / (np.pi * 2 * cutoff_hz))
            # make the update proportional to the local intensity
            eps = inten01 * (deltaTime / tau)
            eps[eps[:] > 1] = 1  # keep filter stable
            # first internal state is updated
            #print(deltaTime , tau)
            #print(-1,irradiance_maps[0,0,24,24])
            #print(0,irradiance_maps_leak[0,0,24,24])
            #print(1,resample[0,0,24,24])
            #print(1.5,eps[0,0,24,24])
            #print(images_next[0,0,198,198].item())
            logNewFrame = self.ln_map(images_next)
            
            lpLogFrame0 = (1-eps)*lpLogFrame0+eps*logNewFrame
            # then 2nd internal state (output) is updated from first
            lpLogFrame1 = (1-eps)*lpLogFrame1+eps*lpLogFrame0
            #lpLogFrame0 = (1-eps)*irradiance_maps+eps*resample
            # then 2nd internal state (output) is updated from first
            #resample_filter = (1-eps)*irradiance_maps+eps*lpLogFrame0
            #print(a[0,0,197,220].item(),logNewFrame[0,0,197,220].item(),logNewFrame[0,0,197,220].item(),eps[0,0,197,220].item(),lpLogFrame0[0,0,197,220].item(),lpLogFrame1[0,0,197,220].item())
            #print(a[0,0,198,198].item(),logNewFrame[0,0,198,197].item(),logNewFrame[0,0,198,198].item(),eps[0,0,198,198].item(),lpLogFrame0[0,0,198,198].item(),lpLogFrame1[0,0,198,198].item())
            #print(lpLogFrame1[0,0,198,198].item())
            #'''
            #lpLogFrame1 = self.resample2d(irradiance_maps, optic_flows*i/10)[:,:,min_x:max_x,min_y:max_y]
            
            b[:,i:i+1] = lpLogFrame1+(i-1)*0.1*deltaTime*0.2 -a
            '''
            a = a-0.1*deltaTime*0.15
            #print('aa', 0.1*deltaTime*0.15)
            #resample_filter = lpLogFrame1
            diffImg = lpLogFrame1 - a
            output1 = diffImg // threshold_C_pos
            #print(output1.max(),output1.min())
            a += output1*threshold_C_pos
            output_pos += F.relu(output1)
            output_neg += F.relu(-output1)
            num_iters = 0#int(max(output1.max(),-output1.min()).item())
            #print(num_iters)
            #print('output', output1.max().item(),-output1.min().item(),F.relu(output1).sum().item(), F.relu(-output1).sum().item())
            for j in range(0,num_iters):
                SHOT_NOISE_INTEN_FACTOR = 0.25
                shotNoiseFactor = (
                    (shot_noise_rate_hz/2)*deltaTime/num_iters) * \
                    ((SHOT_NOISE_INTEN_FACTOR-1)*inten01+1)
                # =1 for inten=0 and SHOT_NOISE_INTEN_FACTOR for inten=1

                rand01 = torch.rand(a.size())  # draw samples
                if self.is_cuda_available:
                    rand01 = rand01.cuda()
                # probability for each pixel is
                # dt*rate*nom_thres/actual_thres.
                # That way, the smaller the threshold,
                # the larger the rate
                shotOnProbThisSample = shotNoiseFactor*0.15/threshold_C_pos
                # array with True where ON noise event
                shotOnCord = torch.zeros_like(a).cuda()
                shotOnCord[rand01 > (1-shotOnProbThisSample)] = 1
                
                shotOffProbThisSample = shotNoiseFactor*0.15/threshold_C_pos
                # array with True where ON noise event
                shotOffCord = torch.zeros_like(a).cuda()
                shotOffCord[rand01 < shotOffProbThisSample] = 1
                a += shotOnCord*threshold_C_pos
                a -= shotOffCord*threshold_C_pos
                output_pos += shotOnCord
                output_neg += shotOffCor
        #'''
        #print(b[0,:,198,198].cpu().data.numpy())
        #b -= a
        #print(b[:,0,0,198,198].cpu().data.numpy())
        b = b // threshold_C_pos
        #print(b[0,:,198,198].cpu().data.numpy())
        diffImg = b[:,1:numframes+1] - b[:,0:numframes]
        #print(diffImg.size(),b.size())
        #diffImg[(diffImg>=0) * (b[1:numframes]>0)] += 0
        #diffImg[(diffImg>0) * (b[1:numframes]<=0)] += -1
        #diffImg[(diffImg<=0) * (b[1:numframes]<0)] += 0
        #diffImg[(diffImg<0) * (b[1:numframes]>=0)] += 1
        diffImg=diffImg-((diffImg>0)*(b[:,1:numframes+1]<=0)).float()+((diffImg<0)*(b[:,1:numframes+1]>=0)).float()
        
        
        print(diffImg[0,:,198,198].cpu().data.numpy())
        #print(threshold_C_pos[0,0,198,198].cpu().data.numpy())
        #diffImg = diffImg // threshold_C_pos
        #print(diffImg[:,0,0,198,198].cpu().data.numpy())

        output_pos = (F.relu(diffImg)).sum(dim=1,keepdim=True)
        output_neg = (F.relu(-diffImg)).sum(dim=1,keepdim=True)
        #print('output', output_pos.sum(), output_neg.sum())
        #output = diffImg.sum(dim=1,keepdim=True)
        output = torch.cat((output_pos, output_neg), 1)
        '''
        deltaTime = 0.05/9
        cutoff_hz = 30
        #irradiance_maps_leak = F.relu(irradiance_maps[:,:,min_x:max_x,min_y:max_y]-0.1*deltaTime*10*0.2) #leak noise 0.1 duration time 0.05s
        a = self.resample2d(images, optic_flows*0)[:,:,min_x:max_x,min_y:max_y]
        a = self.ln_map((a*0.226+0.449)*255)
        #a = self.resample2d(irradiance_maps, optic_flows*0)[:,:,min_x:max_x,min_y:max_y]
        
        #normal_sample = torch.randn(a.size())
        #if self.is_cuda_available:
        #    normal_sample = normal_sample.cuda()
        #threshold_C_pos = normal_sample * 0.05 + 0.2
        #threshold_C_pos = F.relu(threshold_C_pos - self.minimum_threshold) + self.minimum_threshold
        
        
        random_sample = torch.rand(a.size())*0.2
        if self.is_cuda_available:
            random_sample = random_sample.cuda()
        a -= random_sample
        
        #irradiance_maps_leak = a-0.1*deltaTime*9*0.15
        
        lpLogFrame1 = a
        lpLogFrame0 = a
        
        shot_noise_rate_hz = 0.1
        
        output_pos = torch.zeros_like(a).cuda()
        output_neg = torch.zeros_like(a).cuda()
        #threshold_C_pos = torch.zeros_like(a).cuda()+0.2
        for i in range(1,10):
            images_next = self.resample2d(images, optic_flows*i/10)[:,:,min_x:max_x,min_y:max_y]
            #print(images_next[0,0,198,198].item())
            images_next = (images_next*0.226+0.449)*255
            #logNewFrame = self.resample2d(irradiance_maps, optic_flows*i/10)[:,:,min_x:max_x,min_y:max_y]
            #print(images_next[0,0,198,198].item())
            inten01 = (images_next + 20)/275 # limit max time constant to ~1/10 of white intensity level
            tau = (1 / (np.pi * 2 * cutoff_hz))
            # make the update proportional to the local intensity
            eps = inten01 * (deltaTime / tau)
            eps[eps[:] > 1] = 1  # keep filter stable
            # first internal state is updated
            #print(deltaTime , tau)
            #print(-1,irradiance_maps[0,0,24,24])
            #print(0,irradiance_maps_leak[0,0,24,24])
            #print(1,resample[0,0,24,24])
            #print(1.5,eps[0,0,24,24])
            #print(images_next[0,0,198,198].item())
            logNewFrame = self.ln_map(images_next)
            
            lpLogFrame0 = (1-eps)*lpLogFrame0+eps*logNewFrame
            # then 2nd internal state (output) is updated from first
            lpLogFrame1 = (1-eps)*lpLogFrame1+eps*lpLogFrame0
            #lpLogFrame0 = (1-eps)*irradiance_maps+eps*resample
            # then 2nd internal state (output) is updated from first
            #resample_filter = (1-eps)*irradiance_maps+eps*lpLogFrame0
            #print(a[0,0,197,220].item(),logNewFrame[0,0,197,220].item(),logNewFrame[0,0,197,220].item(),eps[0,0,197,220].item(),lpLogFrame0[0,0,197,220].item(),lpLogFrame1[0,0,197,220].item())
            #print(a[0,0,198,198].item(),logNewFrame[0,0,198,197].item(),logNewFrame[0,0,198,198].item(),eps[0,0,198,198].item(),lpLogFrame0[0,0,198,198].item(),lpLogFrame1[0,0,198,198].item())
            #print(images_next[0,0,198,198].item())
            a = a-0.1*deltaTime*0.15
            #print('aa', 0.1*deltaTime*0.15)
            #resample_filter = lpLogFrame1
            diffImg = lpLogFrame1 - a
            output1 = diffImg // threshold_C_pos
            print(lpLogFrame1[0,0,198,198].item(),a[0,0,198,198].item(),diffImg[0,0,198,198].item(),threshold_C_pos[0,0,198,198].item(),output1[0,0,198,198].item())
            #print(output1.max(),output1.min())
            a += output1*threshold_C_pos
            output_pos += F.relu(output1)
            output_neg += F.relu(-output1)
            num_iters = 0#int(max(output1.max(),-output1.min()).item())
            #print(num_iters)
            #print('output', output1.max().item(),-output1.min().item(),F.relu(output1).sum().item(), F.relu(-output1).sum().item())
            for j in range(0,num_iters):
                SHOT_NOISE_INTEN_FACTOR = 0.25
                shotNoiseFactor = (
                    (shot_noise_rate_hz/2)*deltaTime/num_iters) * \
                    ((SHOT_NOISE_INTEN_FACTOR-1)*inten01+1)
                # =1 for inten=0 and SHOT_NOISE_INTEN_FACTOR for inten=1

                rand01 = torch.rand(a.size())  # draw samples
                if self.is_cuda_available:
                    rand01 = rand01.cuda()
                # probability for each pixel is
                # dt*rate*nom_thres/actual_thres.
                # That way, the smaller the threshold,
                # the larger the rate
                shotOnProbThisSample = shotNoiseFactor*0.15/threshold_C_pos
                # array with True where ON noise event
                shotOnCord = torch.zeros_like(a).cuda()
                shotOnCord[rand01 > (1-shotOnProbThisSample)] = 1
                
                shotOffProbThisSample = shotNoiseFactor*0.15/threshold_C_pos
                # array with True where ON noise event
                shotOffCord = torch.zeros_like(a).cuda()
                shotOffCord[rand01 < shotOffProbThisSample] = 1
                a += shotOnCord*threshold_C_pos
                a -= shotOffCord*threshold_C_pos
                output_pos += shotOnCord
                output_neg += shotOffCor
        print('outputtrue', output_pos.sum(), output_neg.sum())
        
        #'''
        
        
        #resample_pos = F.relu(resample_the)
        #resample_neg = F.relu(-resample_the)
        #resample_pos = F.relu(diffImg) / threshold_C_pos
        #resample_neg = F.relu(-diffImg) / threshold_C_neg
        #print(resample_pos[0,0,23,24])
        #print(resample_neg[0,0,23,24])
        #output = resample_pos - resample_neg
        #print(output[0,0,23,24])

        #resample_pos = self.sigmoid(resample_pos)
        #resample_neg = self.sigmoid(resample_neg)
        #resample_pos = resample_pos / regularization
        #resample_neg = resample_neg / regularization

        #output = torch.cat((resample_pos,resample_neg),dim=1)
        
        '''
        real_normal_sample = torch.randn(normal_mean.size())
        if self.is_cuda_available:
            real_normal_sample = real_normal_sample.cuda()
        real_threshold_C_pos = real_normal_sample * 0.021 + 0.15
        real_threshold_C_neg = real_normal_sample * 0.021 + 0.15
        real_threshold_C_pos = F.relu(real_threshold_C_pos - self.minimum_threshold) + self.minimum_threshold
        real_threshold_C_neg = F.relu(real_threshold_C_neg - self.minimum_threshold) + self.minimum_threshold
        real_resample_pos = F.relu(diffImg) / real_threshold_C_pos
        real_resample_neg = F.relu(-diffImg) / real_threshold_C_neg
        
        #real_resample_pos = self.sigmoid(real_resample_pos)
        #real_resample_neg = self.sigmoid(real_resample_neg)
        
        #real_resample_pos = real_resample_pos / regularization
        #real_resample_neg = real_resample_neg / regularization

        real_output = torch.cat((real_resample_pos,real_resample_neg),dim=1)
        '''
        #randSample = torch.rand_like(output)
        #if self.is_cuda_available:
        #    randSample = randSample.cuda()
        #output = output - torch.log(-torch.log(randSample))
        #output = F.softmax(output, dim=1)
        if False and self.debug_mode:
            print('fake pos mean std, neg mean std', threshold_C_pos.mean(), threshold_C_pos.std(),threshold_C_neg.mean(), threshold_C_neg.std())
            print('fake th',threshold_C_pos[0,0,56,24])
            print('fake image before',images[0,0,56,24])
            print('fake image after',images_next[0,0,56,24])
            print('fake before',irradiance_maps[0,0,56,24])
            print('fake after',resample[0,0,56,24])
            #print('fake before 1',irradiance_maps_leak[0,0,56,24])
            #print('fake after 1',resample_filter[0,0,56,24])
            
        if False and self.debug_mode:            
            #print('INFO: generatornet forward print Image')
            #print('real max', real_output[0].mean())
            #print('fake max', output[0].mean())
            #print('real max', real_output[0].max().max().max())
            #y = resample_pos[0,0].max(0)[0].max(0)[1]
            #x = resample_pos[0,0].max(1)[0].max(0)[1]
            #print(x,y)
            #print(resample)
            #print('regularization',regularization)
            #print(x,y,resample_pos[0,0,x.item(),y.item()],
            #    irradiance_maps[0,0,x.item(),y.item()], resample[0,0,x.item(),y.item()], 
            #    threshold_C_pos[0,0,x.item(),y.item()])
            #print('fake pos max', resample_pos[0].max().max().max())
            #print('fake neg max', resample_neg[0].max().max().max())
            #print('real pos mean std, neg mean std', real_threshold_C_pos.mean(), real_threshold_C_pos.std(),real_threshold_C_neg.mean(), real_threshold_C_neg.std())
            #print('fake pos mean std, neg mean std', threshold_C_pos.mean(), threshold_C_pos.std(),threshold_C_neg.mean(), threshold_C_neg.std())
            #imageio.imwrite('image.bmp', images[0].squeeze().cpu().data.numpy().transpose(1,2,0))
            #imageio.imwrite('light_irradiance_maps.bmp', irradiance_maps[0].squeeze().cpu().data.numpy())
            #imageio.imwrite('threshold_C_pos.bmp', threshold_C_pos[0].squeeze().cpu().data.numpy())
            #imageio.imwrite('threshold_C_neg.bmp', threshold_C_neg[0].squeeze().cpu().data.numpy())
            #imageio.imwrite('image_next.bmp',resample[0].squeeze().cpu().data.numpy())
            #imageio.imwrite('diffImg.bmp', diffImg[0].squeeze().cpu().data.numpy())
            #imageio.imwrite('resample_pos.bmp', resample_pos[0].squeeze().cpu().data.numpy())
            #imageio.imwrite('resample_neg.bmp', resample_neg[0].squeeze().cpu().data.numpy())
            #imageio.imwrite('output.bmp', output.squeeze().cpu().data.numpy())
            #print(diffImg.size())
            #print(threshold_C_neg.size())
            #print(resample_neg.size())
            zero = torch.zeros_like(resample_pos)
            output3 = torch.cat((resample_pos,zero),dim=1)
            output3 = torch.cat((output3,resample_neg),dim=1)
            #print(output3.size())
            imageio.imwrite('event_simulator.bmp', np.trunc((output3[0]).squeeze().cpu().data.numpy().transpose(1,2,0)))
                            
            #real_output3 = torch.cat((real_resample_pos,zero),dim=1)
            #real_output3 = torch.cat((real_output3,real_resample_neg),dim=1)
            #imageio.imwrite('event_real.bmp', np.trunc((real_output3[0]).squeeze().cpu().data.numpy().transpose(1,2,0)))
            '''
            total_num = 20
            index_num = 0
            for i in range(0, optic_flows.size()[2]):
                for j in range(0, optic_flows.size()[3]):
                    if index_num >= total_num:
                        break
                    index_num += 1
                    print(i,j,optic_flows[0,0,i,j],optic_flows[0,1,i,j], int(i+optic_flows[0,1,i,j]),int(j+optic_flows[0,0,i,j]),resample[0,0,int(i+optic_flows[0,1,i,j]),int(j+optic_flows[0,0,i,j])],irradiance_maps[0,0,i,j])
            '''
        #imageio.imwrite('image_next.bmp',resample[0].squeeze().cpu().data.numpy())
        #return output, irradiance_maps, real_output, resample, images, \
        #    real_threshold_C_pos, threshold_C_pos, \
        #    real_threshold_C_neg, threshold_C_neg
        #return images, irradiance_maps, resample, output, threshold_C_pos, threshold_C_neg
        return images, a, images, output, threshold_C_pos, threshold_C_pos

class SegNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()

        layers = [
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ] * num_layers
        layers += [
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class EvSegNetOld(nn.Module):

    def __init__(self, args):
        super().__init__()
        
        self.minimum_threshold = args.minimum_threshold

        self.debug_mode = args.debug
        self.is_cuda_available = args.cuda
        self.resizeShape = args.size
        self.args = args
        
        decoders = list(models.vgg16(pretrained=False).features.children())

        self.dec1 = nn.Sequential(*decoders[:5])
        self.dec2 = nn.Sequential(*decoders[5:10])
        self.dec3 = nn.Sequential(*decoders[10:17])
        self.dec4 = nn.Sequential(*decoders[17:24])
        self.dec5 = nn.Sequential(*decoders[24:])

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        m.requires_grad = False
        #'''
        self.enc5_m = SegNetEnc(512, 512, 1)
        self.enc4_m = SegNetEnc(1024, 256, 1)
        self.enc3_m = SegNetEnc(512, 128, 1)
        self.enc2_m = SegNetEnc(256, 64, 0)
        self.enc1_m = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.final_m = nn.Conv2d(64, 1, 3, padding=1)
        
        self.enc5_s = SegNetEnc(512, 512, 1)
        self.enc4_s = SegNetEnc(1024, 256, 1)
        self.enc3_s = SegNetEnc(512, 128, 1)
        self.enc2_s = SegNetEnc(256, 64, 0)
        self.enc1_s = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.final_s = nn.Conv2d(64, 1, 3, padding=1)
        '''
        self.enc_m = nn.Sequential(
            SegNetEnc(512, 512, 1),
            SegNetEnc(512, 256, 1),
            SegNetEnc(256, 128, 1),
            SegNetEnc(128, 64, 0),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1))
        
        self.enc_s = nn.Sequential(
            SegNetEnc(512, 512, 1),
            SegNetEnc(512, 256, 1),
            SegNetEnc(256, 128, 1),
            SegNetEnc(128, 64, 0),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1))
        '''
        self.warp = Resample2d()
        #self.channelnorm = ChannelNorm()
        #self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, images, irradiance_maps, optic_flows):    
        '''
            Attention, input size should be the 32x. 
        '''
        #irradiance_maps = torch.log(self.args.log_eps + irradiance_maps_nl)
        #regularization = torch.zeros((1))
        #if self.is_cuda_available:
        #    regularization = regularization.cuda()
        #regularization = torch.log((regularization+self.args.log_eps+255)/self.args.log_eps)/self.minimum_threshold
        x = images
        #x = F.interpolate(images, self.resizeShape, mode='nearest')
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        dec5 = self.dec5(dec4)
        
        #normal_mean = self.enc_m(dec5)
        #normal_std = self.enc_s(dec5)
        #'''
        enc5_m = self.enc5_m(dec5)
        enc4_m = self.enc4_m(torch.cat([dec4, enc5_m], 1))
        enc3_m = self.enc3_m(torch.cat([dec3, enc4_m], 1))
        enc2_m = self.enc2_m(torch.cat([dec2, enc3_m], 1))
        enc1_m = self.enc1_m(torch.cat([dec1, enc2_m], 1))
        normal_mean = self.final_m(enc1_m)
        
        enc5_s = self.enc5_s(dec5) 
        enc4_s = self.enc4_s(torch.cat([dec4, enc5_s], 1))
        enc3_s = self.enc3_s(torch.cat([dec3, enc4_s], 1))
        enc2_s = self.enc2_s(torch.cat([dec2, enc3_s], 1))
        enc1_s = self.enc1_s(torch.cat([dec1, enc2_s], 1))
        normal_std = self.final_s(enc1_s)
        #'''
        #normal = F.interpolate(self.final(enc1), x.size()[2:])
        
        #normal = enc1

        #mean = normal[:,0,:]
        #std = normal[:,1,:]
        #print(normal_mean.size(),normal_std.size())
        normal_sample = torch.randn(normal_mean.size())
        if self.is_cuda_available:
            normal_sample = normal_sample.cuda()
        #threshold_C_pos = normal_sample[:,0:1,:] * normal[:,1:2,:] + normal[:,0:1,:]
        #threshold_C_neg = normal_sample[:,1:2,:] * normal[:,1:2,:] + normal[:,0:1,:]
        threshold_C_pos = normal_sample * normal_std + normal_mean
        threshold_C_neg = normal_sample * normal_std + normal_mean
        #print(threshold_C_pos[0,0,23,24])
        #print(threshold_C_neg[0,0,23,24])
        threshold_C_pos = F.relu(threshold_C_pos - self.minimum_threshold) + self.minimum_threshold
        threshold_C_neg = F.relu(threshold_C_neg - self.minimum_threshold) + self.minimum_threshold
        #print(threshold_C_pos[0,0,23,24])
        #print(threshold_C_neg[0,0,23,24])
        #print(irradiance_maps.size())
        #print(optic_flows.size())
        resample = self.warp(irradiance_maps, optic_flows)
        #print(irradiance_maps[0,0,23,24])
        #print(resample[0,0,23,24])
        #print(resample.size())
        diffImg = irradiance_maps - resample 
        #print(diffImg[0,0,23,24])
        #normDiffImg = self.channelnorm(diffImg)
        #diffImg = F.interpolate(diffImg, x.size()[2:])
        #print(diffImg[0,0,23,24])
        resample_pos = F.relu(diffImg) / threshold_C_pos
        resample_neg = F.relu(-diffImg) / threshold_C_neg
        #print(resample_pos[0,0,23,24])
        #print(resample_neg[0,0,23,24])
        #output = resample_pos - resample_neg
        #print(output[0,0,23,24])

        #resample_pos = self.sigmoid(resample_pos)
        #resample_neg = self.sigmoid(resample_neg)
        #resample_pos = resample_pos / regularization
        #resample_neg = resample_neg / regularization

        output = torch.cat((resample_pos,resample_neg),dim=1)
        
        
        real_normal_sample = torch.randn(normal_mean.size())
        if self.is_cuda_available:
            real_normal_sample = real_normal_sample.cuda()
        real_threshold_C_pos = real_normal_sample * 0.021 + 0.15
        real_threshold_C_neg = real_normal_sample * 0.021 + 0.15
        real_threshold_C_pos = F.relu(real_threshold_C_pos - self.minimum_threshold) + self.minimum_threshold
        real_threshold_C_neg = F.relu(real_threshold_C_neg - self.minimum_threshold) + self.minimum_threshold
        real_resample_pos = F.relu(diffImg) / real_threshold_C_pos
        real_resample_neg = F.relu(-diffImg) / real_threshold_C_neg
        
        #real_resample_pos = self.sigmoid(real_resample_pos)
        #real_resample_neg = self.sigmoid(real_resample_neg)
        
        #real_resample_pos = real_resample_pos / regularization
        #real_resample_neg = real_resample_neg / regularization

        real_output = torch.cat((real_resample_pos,real_resample_neg),dim=1)
        
        #randSample = torch.rand_like(output)
        #if self.is_cuda_available:
        #    randSample = randSample.cuda()
        #output = output - torch.log(-torch.log(randSample))
        #output = F.softmax(output, dim=1)

        if False and self.debug_mode:            
            print('INFO: generatornet forward print Image')
            #print('real max', real_output[0].mean().mean().mean())
            print('fake max', output[0].mean().mean().mean())
            #print('real max', real_output[0].max().max().max())
            #y = resample_pos[0,0].max(0)[0].max(0)[1]
            #x = resample_pos[0,0].max(1)[0].max(0)[1]
            #print(x,y)
            #print(resample)
            #print('regularization',regularization)
            #print(x,y,resample_pos[0,0,x.item(),y.item()],
            #    irradiance_maps[0,0,x.item(),y.item()], resample[0,0,x.item(),y.item()], 
            #    threshold_C_pos[0,0,x.item(),y.item()])
            #print('fake pos max', resample_pos[0].max().max().max())
            #print('fake neg max', resample_neg[0].max().max().max())
            #print('real pos mean std, neg mean std', real_threshold_C_pos.mean(), real_threshold_C_pos.std(),real_threshold_C_neg.mean(), real_threshold_C_neg.std())
            print('fake pos mean std, neg mean std', threshold_C_pos.mean(), threshold_C_pos.std(),threshold_C_neg.mean(), threshold_C_neg.std())
            imageio.imwrite('image.bmp', images[0].squeeze().cpu().data.numpy().transpose(1,2,0))
            imageio.imwrite('light_irradiance_maps.bmp', irradiance_maps[0].squeeze().cpu().data.numpy())
            #imageio.imwrite('threshold_C_pos.bmp', threshold_C_pos[0].squeeze().cpu().data.numpy())
            #imageio.imwrite('threshold_C_neg.bmp', threshold_C_neg[0].squeeze().cpu().data.numpy())
            imageio.imwrite('light_resample.bmp',resample[0].squeeze().cpu().data.numpy())
            #imageio.imwrite('diffImg.bmp', diffImg[0].squeeze().cpu().data.numpy())
            #imageio.imwrite('resample_pos.bmp', resample_pos[0].squeeze().cpu().data.numpy())
            #imageio.imwrite('resample_neg.bmp', resample_neg[0].squeeze().cpu().data.numpy())
            #imageio.imwrite('output.bmp', output.squeeze().cpu().data.numpy())
            #print(diffImg.size())
            #print(threshold_C_neg.size())
            #print(resample_neg.size())
            zero = torch.zeros_like(resample_pos)
            output3 = torch.cat((resample_pos,zero),dim=1)
            output3 = torch.cat((output3,resample_neg),dim=1)
            #print(output3.size())
            imageio.imwrite('event_simulator.bmp', np.trunc((output3[0]).squeeze().cpu().data.numpy().transpose(1,2,0)))
                            
            #real_output3 = torch.cat((real_resample_pos,zero),dim=1)
            #real_output3 = torch.cat((real_output3,real_resample_neg),dim=1)
            #imageio.imwrite('event_real.bmp', np.trunc((real_output3[0]).squeeze().cpu().data.numpy().transpose(1,2,0)))
            '''
            total_num = 20
            index_num = 0
            for i in range(0, optic_flows.size()[2]):
                for j in range(0, optic_flows.size()[3]):
                    if index_num >= total_num:
                        break
                    index_num += 1
                    print(i,j,optic_flows[0,0,i,j],optic_flows[0,1,i,j], int(i+optic_flows[0,1,i,j]),int(j+optic_flows[0,0,i,j]),resample[0,0,int(i+optic_flows[0,1,i,j]),int(j+optic_flows[0,0,i,j])],irradiance_maps[0,0,i,j])
            '''
        return output, irradiance_maps, real_output, resample, images, \
            real_threshold_C_pos, threshold_C_pos, \
            real_threshold_C_neg, threshold_C_neg
