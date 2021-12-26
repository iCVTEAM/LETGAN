import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from DataLoader.transform import MyTransform, MyOpticFlows
from PIL import Image
from .resample2d_package.resample2d import Resample2d
import cv2
import imageio
import math

class FlowGenerator(nn.Module):
    def __init__(self, args, info):
    #def __init__(self, args, info, len_seq):
        super(FlowGenerator, self).__init__()
        self.args = args
        #self.len_seq = len_seq
        self.args.reshape_size = info['size']
        self.tmax = info['tmax']
        background = info['background_path']
        foreground_list = info['foreground_path']
        self.back_params = info['background_params']
        self.fore_params_list = info['foreground_params']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = MyOpticFlows(self.args)
        self.foreground_list = []
        for index, foreground in enumerate(foreground_list):
            image = self.readImg(foreground, self.fore_params_list[index][0],self.fore_params_list[index][1])
            self.foreground_list.append(image)
        self.background = self.readImg(background,self.back_params[0],self.back_params[1])[:,:,0]
        self.tnow = 0
        self.dt = 0
        #imageio.imwrite('1.png', self.background.astype(np.uint8))
        #self.background = self.repeat3(self.background)
        #self.back_flow = np.zeros((1,2,self.args.reshape_size[1], self.args.reshape_size[0]))
        #imageio.imwrite('2.png', self.background.astype(np.uint8))
        #self.iter = 0
    def fliplr(self, matrix):
        return np.flip(matrix, 1)
        
    def flipud(self, matrix):
        return np.flip(matrix, 0)
        
    def repeat3(self, background):
        output = np.zeros((background.shape[0]*3, background.shape[1]*3))
        output[0:background.shape[0], 0:background.shape[1]] = self.flipud(self.fliplr(background))
        output[0:background.shape[0], background.shape[1]:background.shape[1]*2] = self.flipud(background)
        output[0:background.shape[0], background.shape[1]*2:background.shape[1]*3] = self.flipud(self.fliplr(background))
        output[background.shape[0]:background.shape[0]*2, 0:background.shape[1]] = self.fliplr(background)
        output[background.shape[0]:background.shape[0]*2, background.shape[1]:background.shape[1]*2] = background
        output[background.shape[0]:background.shape[0]*2, background.shape[1]*2:background.shape[1]*3] = self.fliplr(background)
        output[background.shape[0]*2:background.shape[0]*3, 0:background.shape[1]] = self.flipud(self.fliplr(background))
        output[background.shape[0]*2:background.shape[0]*3, background.shape[1]:background.shape[1]*2] = self.flipud(background)
        output[background.shape[0]*2:background.shape[0]*3, background.shape[1]*2:background.shape[1]*3] = self.flipud(self.fliplr(background))
        return output
        
    #def setIter(self, iter):
    #    self.iter = iter
        
    def readImg(self, image_name,median_filter_size, gaussian_blur_sigma):
        image = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
        channel_num = len(cv2.split(image))
        #print(channel_num)
        #print(image_name)
        if channel_num == 3:
            b,g,r = cv2.split(image)
            #image = cv2.merge([r,g,b])
            image = 0.07*b+0.72*g+0.21*r
            image_alpha =  np.ones(r.shape)*255
        elif channel_num == 4:
            b,g,r,a = cv2.split(image)
            image = 0.07*b+0.72*g+0.21*r
            image_alpha =  a
        elif channel_num == 1:
            image_alpha =  np.ones(image.shape)*255
        else:
            assert 1==0
        image = image.astype(np.uint8)
        
        if median_filter_size > 0:
            image = cv2.medianBlur(image, int(median_filter_size))
        if gaussian_blur_sigma > 0:
            image = cv2.Guassianblur(image, (21, 21), gaussian_blur_sigma)
        img = np.zeros((image.shape[0], image.shape[1], 3))
        img[:,:,0] = image
        img[:,:,1] = image_alpha
        return img
        '''
        image = Image.open(image_name).convert('RGBA')
        image_alpha =  np.array(image)[:,:,3]
        image = np.array(image.convert('L'))#.astype(np.uint8)
        if median_filter_size > 0:
            image = cv2.medianBlur(image, int(median_filter_size))
        if gaussian_blur_sigma > 0:
            image = cv2.Guassianblur(image, (21, 21), gaussian_blur_sigma)
        img = np.zeros((image.shape[0], image.shape[1], 3))
        img[:,:,0] = image
        img[:,:,1] = image_alpha
        return img
        '''
    def getDt(self,flow):
        simu_minimum_framerate = 1.0/72.0
        simu_maximum_framerate = 1.0/1000.0
        dt = 0.5/math.sqrt(np.max(flow[0]*flow[0]+flow[1]*flow[1]))
        dt = dt if dt < simu_minimum_framerate else simu_minimum_framerate
        dt = dt if dt > simu_maximum_framerate else simu_maximum_framerate
        #print(dt)
        return dt
    def getImg(self):
        self.tnow = self.tnow + self.dt
        #self.tnow = self.tmax*self.iter/self.len_seq
        affine_matrix, flow = self.transform.getOpticFlows(self.background.shape, self.tnow, self.tmax, self.back_params)
        image = cv2.warpPerspective(self.background, affine_matrix, self.args.reshape_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        #imageio.imwrite('2.png', image.astype(np.uint8))
        for foreground, fore_params in zip(self.foreground_list, self.fore_params_list):
            affine_matrix,optic_flow = self.transform.getOpticFlows((foreground.shape[0], foreground.shape[1]), self.tnow, self.tmax, fore_params)
            perspective  = cv2.warpPerspective(foreground, affine_matrix, self.args.reshape_size, cv2.INTER_LINEAR)
            image, flow = self.addImg(image, flow, perspective, optic_flow)
        #self.iter += 1
        self.dt = self.getDt(flow)
        img = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        flow = torch.from_numpy(flow.astype(np.float32)).unsqueeze(0).to(self.device)
        #print(flow)
        return img, flow, self.tnow
        #return img, flow
    
    def addImg(self, background, backflow, foreground, foreflow):
        alpha_th = 1
        foreground_image = foreground[:,:,0]
        foreground_alpha = foreground[:,:,1]
        background[foreground_alpha>alpha_th] = foreground_image[foreground_alpha>alpha_th]
        
        backflow_x = backflow[0]
        foreflow_x = foreflow[0]
        backflow_y = backflow[1]
        foreflow_y = foreflow[1]
        backflow_x[foreground_alpha>alpha_th] = foreflow_x[foreground_alpha>alpha_th]
        backflow_y[foreground_alpha>alpha_th] = foreflow_y[foreground_alpha>alpha_th]
        backflow[0]=backflow_x
        backflow[1]=backflow_y
        return background, backflow
        
    def resample(self, img, flow):
        t_width, t_height = img.shape[3], img.shape[2]
        xx, yy = torch.meshgrid(torch.arange(t_width), torch.arange(t_height))  # xx, yy -> WxH
        #xx, yy = xx.to(image0.device), yy.to(image0.device)
        xx = xx.to(img.device)
        yy = yy.to(img.device)
        xx.transpose_(0, 1)
        yy.transpose_(0, 1)
        xx, yy = xx.float(), yy.float()

        flow01_x = flow[:, 0, :, :]  # N x H x W
        flow01_y = flow[:, 1, :, :]  # N x H x W

        warping_grid_x = xx + flow01_x  # N x H x W
        warping_grid_y = yy + flow01_y  # N x H x W

        # normalize warping grid to [-1,1]
        warping_grid_x = (2 * warping_grid_x / (t_width - 1)) - 1
        warping_grid_y = (2 * warping_grid_y / (t_height - 1)) - 1

        warping_grid = torch.stack(
            [warping_grid_x, warping_grid_y], dim=3)  # 1 x H x W x 2

        return F.grid_sample(img, warping_grid)

class FlowGenerator2(nn.Module):
    def __init__(self, args, info, len_seq):
        super(FlowGenerator, self).__init__()
        self.args = args
        self.tmax = len_seq
        self.args.reshape_size = info['size']
        tmax = info['tmax']
        #self.back_params = np.random.rand(12)
        #self.fore_params_list = [np.random.rand(12) for i in range(len(foreground_list))]
        background = info['background_path']
        print(background)
        foreground_list = info['foreground_path']
        self.back_params = info['background_params']
        self.fore_params_list = info['foreground_params']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = MyOpticFlows(self.args)
        #size_min = 200
        #size_max = 200
        self.foreground_list = []
        self.fore_size = []
        for foreground in foreground_list:
            size = 100#random.randint(size_min,size_max)
            image = self.readImg(foreground, [size,size])
            self.fore_size.append([size,size])
            foreground_pos = [0,0]
            self.foreground_list.append(torch.zeros((1,2,self.args.reshape_size[1],self.args.reshape_size[0])).to(self.device))
            self.foreground_list[-1][:,:,foreground_pos[0]:foreground_pos[0]+image.size()[2],foreground_pos[1]:foreground_pos[1]+image.size()[3]] = image
        self.background = self.readImg(background, self.args.reshape_size)[:,0:1]
        self.back_flow = torch.zeros((1,2,self.args.reshape_size[1], self.args.reshape_size[0])).to(self.device)
        self.fore_flow_list = [torch.zeros((1,2,self.args.reshape_size[1], self.args.reshape_size[0])).to(self.device) for i in range(len(foreground_list))]
        
        #self.back_origin_flow = torch.zeros((1,2,self.args.reshape_size[1], self.args.reshape_size[0])).to(self.device)
        #self.fore_origin_flow_list = [torch.zeros((1,2,self.args.reshape_size[1], self.args.reshape_size[0])).to(self.device) for i in range(len(foreground_list))]
        self.fore_flow_once_step_list = [torch.zeros((1,2,self.args.reshape_size[1], self.args.reshape_size[0])).to(self.device) for i in range(len(foreground_list))]

        
        self.resample2d = Resample2d()
        
        #self.numframes = 1
        self.iter = 0
        
    def readImg(self, image_name, size):
        image = Image.open(image_name).convert('RGBA')
        image = image.resize(size,Image.BILINEAR)
        image_alpha =  np.array(image)[:,:,3].astype(np.float32)
        image_alpha = torch.from_numpy(image_alpha).unsqueeze(0).unsqueeze(0).to(self.device)
        image = np.array(image.convert('L')).astype(np.float32)
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
        return torch.cat((image, image_alpha), dim=1)
    
    def getFlow(self, size, t, tmax, params):
        optic_flow = self.transform.getOpticFlows(size, t, tmax, params)
        optic_flow = optic_flow.astype(np.float32)
        optic_flow = torch.from_numpy(optic_flow).unsqueeze(0).to(self.device)
        #optic_flow[0,0]=optic_flow[0,0]*0+6000
        #optic_flow[0,1]=optic_flow[0,1]*0
        return optic_flow
    
    def setFlow(self):
        self.back_flow = self.getFlow(size = self.args.reshape_size, t = self.iter, tmax=self.tmax, params=self.back_params)
        for i, optic_flow in enumerate(self.fore_flow_list):
            self.fore_flow_list[i] = self.getFlow(size = self.fore_size[i], t = self.iter, tmax=self.tmax, params=self.fore_params_list[i])
        if self.iter == 0:
            self.back_flow_once_step = self.getFlow(size = self.args.reshape_size,t = 1, tmax=self.tmax, params=self.back_params)
            for i, optic_flow in enumerate(self.fore_flow_once_step_list):
                self.fore_flow_once_step_list[i] = self.getFlow(size = self.fore_size[i], t = 1, tmax=self.tmax, params=self.fore_params_list[i])
        
    def addImg(self, background, flow_output, foreground, flow):
        alpha_th = 244
        foreground_image = foreground[:,0:1]
        foreground_alpha = foreground[:,1:2]
        background[foreground_alpha>alpha_th] = foreground_image[foreground_alpha>alpha_th]
        flow_output_x = flow_output[:,0:1]
        flow_x = flow[:,0:1]
        flow_output_y = flow_output[:,1:2]
        flow_y = flow[:,1:2]
        flow_output_x[foreground_alpha>alpha_th] = flow_x[foreground_alpha>alpha_th]
        flow_output_y[foreground_alpha>alpha_th] = flow_y[foreground_alpha>alpha_th]
        return background, torch.cat((flow_output_x, flow_output_y), dim=1)

    def getImg(self):
        self.setFlow()
        flow = self.resample2d(self.back_flow_once_step, self.back_flow)
        image = self.resample2d(self.background, self.back_flow)
        for foreground, fore_flow_once_step, fore_flow in zip(self.foreground_list, self.fore_flow_once_step_list, self.fore_flow_list):
            foreground = self.resample2d(foreground, fore_flow)
            flow_step = self.resample2d(fore_flow_once_step, fore_flow)
            image, flow = self.addImg(image, flow, foreground, flow_step)
        self.iter += 1
        return image, flow
        
    def resample(self, img, flow):
        t_width, t_height = img.shape[3], img.shape[2]
        xx, yy = torch.meshgrid(torch.arange(t_width), torch.arange(t_height))  # xx, yy -> WxH
        #xx, yy = xx.to(image0.device), yy.to(image0.device)
        xx = xx.to(img.device)
        yy = yy.to(img.device)
        xx.transpose_(0, 1)
        yy.transpose_(0, 1)
        xx, yy = xx.float(), yy.float()

        flow01_x = flow[:, 0, :, :]  # N x H x W
        flow01_y = flow[:, 1, :, :]  # N x H x W

        warping_grid_x = xx + flow01_x  # N x H x W
        warping_grid_y = yy + flow01_y  # N x H x W

        # normalize warping grid to [-1,1]
        warping_grid_x = (2 * warping_grid_x / (t_width - 1)) - 1
        warping_grid_y = (2 * warping_grid_y / (t_height - 1)) - 1

        warping_grid = torch.stack(
            [warping_grid_x, warping_grid_y], dim=3)  # 1 x H x W x 2

        return F.grid_sample(img, warping_grid)
        
    def getImg2(self):
        #do not use !!!!
        if(self.iter % self.numframes == 0):
            self.setFlow()
            self.iter = 0
        flow_tmp = self.back_origin_flow+self.back_flow*self.iter/self.numframes
        flow = self.resample(self.back_flow/self.numframes, flow_tmp)
        image = self.resample(self.background, flow_tmp)
        for foreground, fore_origin_flow, fore_flow in zip(self.foreground_list, self.fore_origin_flow_list, self.fore_flow_list):
            flow_tmp = fore_origin_flow+fore_flow*self.iter/self.numframes
            foreground = self.resample(foreground, flow_tmp)
            flow_step = self.resample(fore_flow/self.numframes, flow_tmp)
            image, flow = self.addImg(image, flow, foreground, flow_step)
        self.iter += 1
        return image, flow
        
class FlowGeneratorold(nn.Module):
    def __init__(self, args, foreground_list, background, len_seq):
        super(FlowGenerator, self).__init__()
        self.args = args
        size_min = 100
        size_max = 200
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = MyTransform(args, reshape_size=args.reshape_size,crop_size=args.size,debug_mode=args.debug,train_mode=False)
        
        self.foreground_list = []
        for foreground in foreground_list:
            size = random.randint(size_min,size_max)
            image = self.readImg(foreground, [size,size])
            foreground_pos = [(self.args.reshape_size[1]-image.size()[2])//2,(self.args.reshape_size[0]-image.size()[3])//2]
            self.foreground_list.append(torch.zeros((1,2,self.args.reshape_size[1],self.args.reshape_size[0])).to(self.device))
            self.foreground_list[-1][:,:,foreground_pos[0]:foreground_pos[0]+image.size()[2],foreground_pos[1]:foreground_pos[1]+image.size()[3]] = image
        self.background = self.readImg(background, self.args.reshape_size)[:,0:1]
        self.back_flow = torch.zeros((1,2,self.args.reshape_size[1], self.args.reshape_size[0])).to(self.device)
        self.fore_flow_list = [torch.zeros((1,2,self.args.reshape_size[1], self.args.reshape_size[0])).to(self.device) for i in range(len(foreground_list))]
        
        #self.back_origin_flow = torch.zeros((1,2,self.args.reshape_size[1], self.args.reshape_size[0])).to(self.device)
        #self.fore_origin_flow_list = [torch.zeros((1,2,self.args.reshape_size[1], self.args.reshape_size[0])).to(self.device) for i in range(len(foreground_list))]
        self.fore_flow_once_step_list = [torch.zeros((1,2,self.args.reshape_size[1], self.args.reshape_size[0])).to(self.device) for i in range(len(foreground_list))]
        self.back_random = np.random.rand(12)
        self.fore_random_list = [np.random.rand(12) for i in range(len(foreground_list))]
        
        self.resample2d = Resample2d()
        
        #self.numframes = 1
        self.iter = 0
        self.first = [random.random() for i in range(len(foreground_list))]
        for i in range(len(foreground_list)):
            tx = (self.fore_random_list[i][3]*(self.args.max_translate_x-self.args.min_translate_x)+self.args.min_translate_x)*len_seq
            ty = (self.fore_random_list[i][4]*(self.args.max_translate_y-self.args.min_translate_y)+self.args.min_translate_y)*len_seq
            tmax = tx if tx > ty else ty
            self.first[i] = tmax*(self.first[i]*0.5+0.25)
        
    def readImg(self, image_name, size):
        image = Image.open(image_name).convert('RGBA')
        image = image.resize(size,Image.BILINEAR)
        image_alpha =  np.array(image)[:,:,3].astype(np.float32)
        image_alpha = torch.from_numpy(image_alpha).unsqueeze(0).unsqueeze(0).to(self.device)
        image = np.array(image.convert('L')).astype(np.float32)
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
        return torch.cat((image, image_alpha), dim=1)
    
    def getFlow(self, rate=1, translate=True, params=None):
        optic_flow = self.transform.getOpticFlows(self.args.reshape_size, rate, translate, params)
        optic_flow = optic_flow.astype(np.float32)
        optic_flow = torch.from_numpy(optic_flow).unsqueeze(0).to(self.device)
        #optic_flow[0,0]=optic_flow[0,0]*0+6000
        #optic_flow[0,1]=optic_flow[0,1]*0
        return optic_flow
    
    def setFlow(self):
        self.back_flow = self.getFlow(translate=False, params=self.back_random,rate=self.iter*0.05)
        for i, optic_flow in enumerate(self.fore_flow_list):
            self.fore_flow_list[i] = self.getFlow(params=self.fore_random_list[i],rate=self.iter-self.first[i])
        if self.iter == 0:
            self.back_flow_once_step = self.getFlow(params=self.back_random)
            for i, optic_flow in enumerate(self.fore_flow_once_step_list):
                self.fore_flow_once_step_list[i] = self.getFlow(params=self.fore_random_list[i])
        
    def setFlowOrigin(self):
        self.back_origin_flow += self.back_flow
        self.back_flow = self.getFlow()
        for i, optic_flow in enumerate(self.fore_flow_list):
            self.fore_origin_flow_list[i] += optic_flow
            self.fore_flow_list[i] = self.getFlow()
        
    def addImg(self, background, flow_output, foreground, flow):
        foreground_image = foreground[:,0:1]
        foreground_alpha = foreground[:,1:2]
        background[foreground_alpha>0.1] = foreground_image[foreground_alpha>0.1]
        flow_output_x = flow_output[:,0:1]
        flow_x = flow[:,0:1]
        flow_output_y = flow_output[:,1:2]
        flow_y = flow[:,1:2]
        flow_output_x[foreground_alpha>0.1] = flow_x[foreground_alpha>0.1]
        flow_output_y[foreground_alpha>0.1] = flow_y[foreground_alpha>0.1]
        return background, torch.cat((flow_output_x, flow_output_y), dim=1)

    def getImg(self):
        self.setFlow()
        flow = self.resample2d(self.back_flow_once_step, self.back_flow)
        image = self.resample2d(self.background, self.back_flow)
        for foreground, fore_flow_once_step, fore_flow in zip(self.foreground_list, self.fore_flow_once_step_list, self.fore_flow_list):
            foreground = self.resample2d(foreground, fore_flow)
            flow_step = self.resample2d(fore_flow_once_step, fore_flow)
            image, flow = self.addImg(image, flow, foreground, flow_step)
        self.iter += 1
        return image, flow
        
    def resample(self, img, flow):
        t_width, t_height = img.shape[3], img.shape[2]
        xx, yy = torch.meshgrid(torch.arange(t_width), torch.arange(t_height))  # xx, yy -> WxH
        #xx, yy = xx.to(image0.device), yy.to(image0.device)
        xx = xx.to(img.device)
        yy = yy.to(img.device)
        xx.transpose_(0, 1)
        yy.transpose_(0, 1)
        xx, yy = xx.float(), yy.float()

        flow01_x = flow[:, 0, :, :]  # N x H x W
        flow01_y = flow[:, 1, :, :]  # N x H x W

        warping_grid_x = xx + flow01_x  # N x H x W
        warping_grid_y = yy + flow01_y  # N x H x W

        # normalize warping grid to [-1,1]
        warping_grid_x = (2 * warping_grid_x / (t_width - 1)) - 1
        warping_grid_y = (2 * warping_grid_y / (t_height - 1)) - 1

        warping_grid = torch.stack(
            [warping_grid_x, warping_grid_y], dim=3)  # 1 x H x W x 2

        return F.grid_sample(img, warping_grid)
        
    def getImg2(self):
        #do not use !!!!
        if(self.iter % self.numframes == 0):
            self.setFlow()
            self.iter = 0
        flow_tmp = self.back_origin_flow+self.back_flow*self.iter/self.numframes
        flow = self.resample(self.back_flow/self.numframes, flow_tmp)
        image = self.resample(self.background, flow_tmp)
        for foreground, fore_origin_flow, fore_flow in zip(self.foreground_list, self.fore_origin_flow_list, self.fore_flow_list):
            flow_tmp = fore_origin_flow+fore_flow*self.iter/self.numframes
            foreground = self.resample(foreground, flow_tmp)
            flow_step = self.resample(fore_flow/self.numframes, flow_tmp)
            image, flow = self.addImg(image, flow, foreground, flow_step)
        self.iter += 1
        return image, flow