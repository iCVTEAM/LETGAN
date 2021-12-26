'''
transform image and target to tensor
'''

import numpy as np
import torch
from .functional import RandomCrop, CenterCrop,RandomFlip,RandomRotate
from PIL import Image
import random
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms import Normalize
from Debug.functional import showMessage
import torch.nn.functional as F
import math
import random

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import imageio

class MyTransform(object):
    '''
        1. self-define transform rules, including resize, crop, flip. (crop and flip only for training set)
        2. training set augmentation with RandomCrop and RandomFlip.
        3. validation set using CenterCrop
    '''
    def __init__(self, args, reshape_size=None, crop_size=None , augment=True, debug_mode=False, train_mode=False): 
        self.reshape_size = reshape_size
        self.crop_size = crop_size
        self.augment = augment
        self.debug_mode = debug_mode
        self.train_mode = train_mode
        self.flip = RandomFlip(0.5)
        self.rotate = RandomRotate(32)
        
        self.use_log_images = args.use_log_images
        #self.log_eps = args.log_eps
        
        self.log_threshold = args.log_threshold
        
        args.min_translate_x = 0
        args.min_translate_y = 0
        args.min_angle_z = 0
        self.args = args
        
        [irradiance_map_height, irradiance_map_width] = self.reshape_size
        x = np.array([i for i in range(irradiance_map_width)])
        y = np.array([i for i in range(irradiance_map_height)])
        self.optic_flow = np.ones((4, irradiance_map_width*irradiance_map_height))
        self.X,self.Y = np.meshgrid(x, y)
        self.optic_flow[0:1] = self.Y.reshape((1, irradiance_map_height*irradiance_map_width))
        self.optic_flow[1:2] = self.X.reshape((1, irradiance_map_height*irradiance_map_width))
        
    def ln_map(self, map):
        map[map < self.log_threshold] = map[map < self.log_threshold]/self.log_threshold*math.log(self.log_threshold)
        map[map >= self.log_threshold] = np.log(map[map >= self.log_threshold])
        #map = np.log(self.log_eps + np.array(map)/255.0)
        return map
    def get_rotate_matrix_x(self, angle_x):
        return np.array([[1, 0                ,  0                ],
                         [0, math.cos(angle_x), -math.sin(angle_x)],
                         [0, math.sin(angle_x),  math.cos(angle_x)]])
                         
    def get_rotate_matrix_y(self, angle_y):
        return np.array([[ math.cos(angle_y), 0, math.sin(angle_y)],
                         [ 0,                 1, 0                ],
                         [-math.sin(angle_y), 0, math.cos(angle_y)]])
                         
    def get_rotate_matrix_z(self, angle_z):
        return np.array([[math.cos(angle_z), -math.sin(angle_z), 0],
                         [math.sin(angle_z),  math.cos(angle_z), 0],
                         [0                ,  0,                 1]])
        
    def getOpticFlows(self, size, rate=1, translate=True, params=None):
        if params is None:
            [r_a_x, r_a_y, r_a_z, r_t_x, r_t_y, r_t_z, s_a_x, s_a_y, s_a_z, s_t_x, s_t_y, s_t_z] = np.random.rand(12)
        else:
            [r_a_x, r_a_y, r_a_z, r_t_x, r_t_y, r_t_z, s_a_x, s_a_y, s_a_z, s_t_x, s_t_y, s_t_z] = params
        angle_x = 0
        angle_y = 0
        angle_z = (r_a_z*(self.args.max_angle_z-self.args.min_angle_z)+self.args.min_angle_z)/180*math.pi*rate
        angle_z = angle_z if s_a_z<0.5 else -angle_z
        #print(angle_z)
        #angle_x = self.args.max_angle_x/180*math.pi
        #angle_z = self.args.max_angle_z/180*math.pi
        if translate:
            translate_x = (r_t_x*(self.args.max_translate_x-self.args.min_translate_x)+self.args.min_translate_x)*rate
            translate_x = translate_x if s_t_x<0.5 else -translate_x
            translate_y = (r_t_y*(self.args.max_translate_y-self.args.min_translate_y)+self.args.min_translate_y)*rate
            translate_y = translate_y if s_t_y<0.5 else -translate_y
            translate_z = 0
        else:
            translate_x = 0
            translate_y = 0
            translate_z = 0
        
        [irradiance_map_height, irradiance_map_width] = size# W H
        
        #rotate_matrix_x = self.get_rotate_matrix_x(angle_x)
        #rotate_matrix_y = self.get_rotate_matrix_y(angle_y)
        #rotate_matrix_z = self.get_rotate_matrix_z(angle_z)
        
        translate_matrix = np.array([[1, 0, 0, irradiance_map_height//2],
                                     [0, 1, 0, irradiance_map_width//2 ],
                                     [0, 0, 1, 1                       ],
                                     [0, 0, 0, 1                       ]])
        
        translate_matrix_neg = np.array([[1, 0, 0, -irradiance_map_height//2+translate_x],
                                         [0, 1, 0, -irradiance_map_width//2+translate_y ],
                                         [0, 0, 1, -1                       ],
                                         [0, 0, 0, 1                        ]])
                                       
        #rotate_matrix = rotate_matrix_x.dot(rotate_matrix_y).dot(rotate_matrix_z)
        rotate_matrix = self.get_rotate_matrix_z(angle_z).dot(self.get_rotate_matrix_y(angle_y)).dot(self.get_rotate_matrix_x(angle_x))
        affine_matrix = np.zeros((4,4))
        affine_matrix[0:3,0:3] = rotate_matrix
        #affine_matrix[0,3] = translate_x
        #affine_matrix[1,3] = translate_y
        #affine_matrix[2,3] = translate_z
        affine_matrix[3,3] = 1
        
        #convert rotate center [0,0] to [height/2, weight/2]
        affine_matrix = translate_matrix.dot(affine_matrix).dot(translate_matrix_neg)
        
        if self.debug_mode:
            print('INFO: transform ', affine_matrix, angle_x, angle_y, angle_z)
        
        '''
        #print(irradiance_map_width, irradiance_map_height)
        optic_flow = np.ones((4, irradiance_map_width*irradiance_map_height))
        for i in range(0, irradiance_map_height):
            for j in range(0, irradiance_map_width):
                optic_flow[0,i*irradiance_map_width+j] = i
                optic_flow[1,i*irradiance_map_width+j] = j
        '''
        optic_flow = self.optic_flow
        '''
        offest = np.zeros((3, 3))
        offest[0,0] = 1
        offest[0,2] = -0.5*irradiance_map_width
        offest[1,1] = -1
        offest[1,2] = 0.5*irradiance_map_height
        offest[2,2] = 1
        de_offest = np.zeros((3, 3))
        de_offest[0,0] = 1
        de_offest[0,2] = 0.5*irradiance_map_width
        de_offest[1,1] = -1
        de_offest[1,2] = 0.5*irradiance_map_height
        de_offest[2,2] = 1
        '''
        #if self.debug_mode:
        #    print('INFO: dataset __getitem__ ')
        #    inflect = affine_matrix.dot(optic_flow)
        #    print(optic_flow[0:2,0:10])
        #    print(inflect[0:2,0:10])
            #for i in range(0, irradiance_map_height):
            #    for j in range(0, irradiance_map_width):
            #        print(i,j,inflect[0,i*irradiance_map_width+j],inflect[1,i*irradiance_map_width+j])
        #optic_flow[0,:] -= irradiance_map_height//2
        #optic_flow[1,:] -= irradiance_map_width//2
        
        
        optic_flow = affine_matrix[0:2].dot(optic_flow)-optic_flow[0:2]
        #optic_flow[0:3,:] = de_offest.dot(rotate_matrix_x).dot(offest).dot(optic_flow[0:3,:])-optic_flow[0:3,:]
        
        #optic_flow[0,:] += irradiance_map_height//2
        #optic_flow[1,:] += irradiance_map_width//2
        
        #optic_flow = optic_flow[0:2,:]
        #print(optic_flow[:,0:10])
        optic_flow = optic_flow.reshape((2, irradiance_map_height, irradiance_map_width))
        #optic_flow = self.loadNpy(optic_flowName)
        #print(irradiance_map.size)
        #print(optic_flow.shape)
        #warp need [2 H W]->[2 W H]
        #optic_flow = optic_flow[[1,0],:]
        optic_flow = optic_flow.transpose((0,2,1))
        #if self.debug_mode:
        #    print('INFO: dataset __getitem__ ')
            #inflect = affine_matrix.dot(optic_flow)
        #    total_num = 20
        #    index_num = 0
        #    for i in range(0, irradiance_map_height):
        #        for j in range(0, irradiance_map_width):
        #            if index_num >= total_num:
        #                break
        #            index_num += 1
        #            print(i,j,optic_flow[0,i,j],optic_flow[1,i,j])
        return optic_flow
    def channel2to3(self, output2):
        '''
        output2 [2,H,W] [0,1]
        output3 [3,H,W]   [0,255]
        '''
        output3 = np.zeros((3,output2.shape[1], output2.shape[2]))
        output3[0,:] = output2[0,:]
        output3[2,:] = output2[1,:]
        
        #regularization = math.log((self.args.log_eps+255)/self.args.log_eps)/self.args.minimum_threshold
        output3[output3>1] = output3[output3>1]+125
        output3[output3>255]=255
        #max_value = output3.max()
        
        #return np.trunc(output3/max_value*255)
        return np.trunc(output3)
    def __call__(self, irradiance_map, real_event_image=None, real_event_map=None, real_event_image2=None):
        ###if not self.train_mode and self.reshape_size is not None:
        ###    irradiance_map = irradiance_map.resize(self.reshape_size,Image.BILINEAR)
        #print(images.size)
        '''
        if self.reshape_size is not None:
            irradiance_map = irradiance_map.resize(self.reshape_size,Image.BILINEAR)
            if self.train_mode:
                real_event_map_tmp = real_event_map[0]
                real_event_map_tmp = Image.fromarray(real_event_map_tmp)
                real_event_map_tmp = real_event_map_tmp.resize(self.crop_size,Image.BILINEAR)
                real_event_map_tmp = np.array(real_event_map_tmp)
                real_event_map_0 = real_event_map_tmp
                
                real_event_map_tmp = real_event_map[1]
                real_event_map_tmp = Image.fromarray(real_event_map_tmp)
                real_event_map_tmp = real_event_map_tmp.resize(self.crop_size,Image.BILINEAR)
                real_event_map_tmp = np.array(real_event_map_tmp)
                real_event_map_1 = real_event_map_tmp

                real_event_map = np.stack((real_event_map_0, real_event_map_1))
        '''
            #foreground = foreground.resize(self.reshape_size,Image.BILINEAR)
        #if not self.train_mode:
        #    optic_flow_fore = self.getOpticFlows(irradiance_map.size)
        #    optic_flow = optic_flow_fore
        #optic_flow_back = self.getOpticFlows(irradiance_map, 0.1)
        #optic_flow = np.zeros((2, irradiance_map.size[1], irradiance_map.size[0]))
        #foreground = np.array(foreground)
        #print(optic_flow_back.shape,foreground.shape, optic_flow.shape)
        #if random.random() < 0.1:
        #    optic_flow[:,foreground==0] = optic_flow_back[:,foreground==0]
        #if random.random() < 0.9:
        #    optic_flow[:,foreground>0] = optic_flow_fore[:,foreground>0]
        #optic_flow = optic_flow_fore
        #optic_flow[:,foreground==0] = optic_flow_back[:,foreground==0]
        #optic_flow[:,foreground>0] = optic_flow_fore[:,foreground>0]
        
        ###image = ToTensor()(irradiance_map)
        ###image = Normalize([.449], [.226])(image)
        #image = Normalize([.485, .456, .406], [.229, .224, .225])(images) #normalize with the params of imagenet
       
        ###optic_flow = optic_flow.astype(np.float32)
        #print(optic_flow.shape)
        #optic_flow = optic_flow.transpose((0,2,1))
        #print(optic_flow.shape)
        #optic_flowX = optic_flow[0,:]
        #optic_flowY = optic_flow[1,:]
        #opticFLows = np.vstack((optic_flowY[np.newaxis,:],optic_flowX[np.newaxis,:]))
        #print(opticFLows.shape)
        ###optic_flow = torch.from_numpy(optic_flow)
        #if self.reshape_size is not None:
        #    optic_flow = optic_flow.unsqueeze(0)
        #    print(optic_flow.size())
        #    optic_flow = F.interpolate(optic_flow, size=self.reshape_size,
        #                           mode='nearest')
        #    optic_flow = optic_flow.squeeze(0)
        
        #print(irradiance_map.size)
        #irradiance_map = irradiance_map.transpose(Image.TRANSPOSE)
        #print(irradiance_map.size)
        #irradiance_map = ToTensor()(irradiance_map)
        #print(irradiance_map.size()) 
        irradiance_map = np.array(irradiance_map)
        #print(irradiance_map[56+18,24+61])
        irradiance_map = irradiance_map.astype(np.float32)
        ###if self.use_log_images:
        ###    irradiance_map = self.ln_map(irradiance_map)
        #irradiance_map = irradiance_map.T
        
        #irradiance_map = irradiance_map[np.newaxis,:]
        #irradiance_map = irradiance_map.transpose((0,2,1))
        irradiance_map = torch.from_numpy(irradiance_map)
        
        min_x = random.randint(0,irradiance_map.size()[1]-self.args.size[0])
        max_x = min_x+self.args.size[0]
        min_y = random.randint(0,irradiance_map.size()[2]-self.args.size[1])
        max_y = min_y+self.args.size[1]
        irradiance_map = irradiance_map[:,min_x:max_x,min_y:max_y]
        #print(irradiance_map.shape)
        #if self.debug_mode:
        #    pass
            #showMessage(fileName='transform.py', className='MyTransform', functionName='__call__', lineNumber=57, variableName='image',
            #        variableValue=type(image))
            
        if self.train_mode:
            real_event_image = np.array(real_event_image)
            real_event_image = real_event_image.astype(np.float32)
            if self.use_log_images:
                real_event_image = self.ln_map(real_event_image)
            #real_event_image = real_event_image[np.newaxis,:]
            real_event_image = torch.from_numpy(real_event_image)
            #if (self.debug_mode):
            #    print('INFO: transform ')
            #    imageio.imwrite('realEveOld.bmp', (self.channel2to3(real_event_map)).astype(np.uint8).transpose((1,2,0)))
            real_event_map = real_event_map.astype(np.float32)
            #real_event_map[0] = cv2.medianBlur(real_event_map[0], ksize=3)
            #real_event_map[1] = cv2.medianBlur(real_event_map[1], ksize=3)
            real_event_map = torch.from_numpy(real_event_map)
            
            real_event_image2 = np.array(real_event_image2)
            real_event_image2 = real_event_image2.astype(np.float32)
            if self.use_log_images:
                real_event_image2 = self.ln_map(real_event_image2)
            #real_event_image = real_event_image[np.newaxis,:]
            real_event_image2 = torch.from_numpy(real_event_image2)
            
            #min_x = real_event_map.size()[1]//2-self.args.size[0]//2
            #max_x = real_event_map.size()[1]//2+self.args.size[0]//2
            #min_y = real_event_map.size()[2]//2-self.args.size[1]//2
            #max_y = real_event_map.size()[2]//2+self.args.size[1]//2
            min_x = random.randint(0,real_event_map.size()[1]-self.args.size[0])
            max_x = min_x+self.args.size[0]
            min_y = random.randint(0,real_event_map.size()[2]-self.args.size[1])
            max_y = min_y+self.args.size[1]
            real_event_image = real_event_image[:,min_x:max_x,min_y:max_y]
            real_event_map = real_event_map[:,min_x:max_x,min_y:max_y]
            real_event_image2 = real_event_image2[:,min_x:max_x,min_y:max_y]
            
            
            #print('bjl',real_event_map[0].sum(),real_event_map[1].sum())
            #[180,240]-0>[224,224]
            #real_event_image_new = torch.zeros((1, self.args.size[0], self.args.size[1]))
            #real_event_map_new = torch.zeros((2, self.args.size[0], self.args.size[1]))
            #real_event_image_new[:,0:real_event_map.size()[1]] = real_event_image[:,:,min_y:max_y]
            #real_event_map_new[:,0:real_event_map.size()[1]] = real_event_map[:,:,min_y:max_y]
            #real_event_image = real_event_image_new
            #real_event_map = real_event_map_new
            
            #print('bjl',real_event_map[0].sum(),real_event_map[1].sum())
            #real_event_map = real_event_map[0:1]-real_event_map[1:2]
            #print(irradiance_map.size(),real_event_image.size(),real_event_map.size())
            '''
            min_x = random.randint(0,irradiance_map.size()[1]-self.args.size[0])
            max_x = min_x+self.args.size[0]
            min_y = random.randint(0,irradiance_map.size()[2]-self.args.size[1])
            max_y = min_y+self.args.size[1]
            real_event_image = real_event_image[:,min_x:max_x,min_y:max_y]
            real_event_map = real_event_map[:,min_x:max_x,min_y:max_y]
            '''
            #print(irradiance_map.size(), optic_flow.size(), real_event_image.size(), real_event_map.size())
            
            if self.augment:
                if random.random() < 0.5:
                    irradiance_map = self.fliplr(irradiance_map)
                    #image = self.fliplr(image)
                    #optic_flow = self.fliplr(optic_flow)
                    #optic_flow[0] *= -1
                    real_event_image = self.fliplr(real_event_image)
                    real_event_image2 = self.fliplr(real_event_image2)
                    real_event_map = self.fliplr(real_event_map)
                if random.random() < 0.5:
                    irradiance_map = self.flipud(irradiance_map)
                    #image = self.flipud(image)
                    #optic_flow = self.flipud(optic_flow)
                    #optic_flow[1] *= -1
                    real_event_image = self.flipud(real_event_image)
                    real_event_image2 = self.flipud(real_event_image2)
                    real_event_map = self.flipud(real_event_map)
                if random.random() < 0.5:
                    irradiance_map = self.transpose(irradiance_map)
                    #image = self.transpose(image)
                    #optic_flow = self.transpose(optic_flow)
                    #optic_flow = optic_flow[[1,0]]
                    #C*W*H->C*H*W
                    #irradiance_map = self.fill(irradiance_map)
                    #image = self.fill(image)
                    #optic_flow = self.fill(optic_flow)
                    
                    real_event_image = self.transpose(real_event_image)
                    real_event_image2 = self.transpose(real_event_image2)
                    real_event_map = self.transpose(real_event_map)
            
            #return image, irradiance_map, optic_flow, real_event_image, real_event_map
            return irradiance_map, real_event_image, real_event_map, real_event_image2
        else:
            return image, irradiance_map, optic_flow
    
    def fill(self, map):
        [C,W,H]=map.size()#W>H
        zero = torch.zeros((C,H,W))
        min_x = W//2-H//2
        max_x = W//2+H//2
        zero[:,:,min_x:max_x] = map[:,min_x:max_x,:]
        return zero
        
    def fliplr(self, matrix):
        length = matrix.size()[2]
        return matrix[:,:,[length-1-i for i in range(length)]]
        
    def flipud(self, matrix):
        length = matrix.size()[1]
        return matrix[:,[length-1-i for i in range(length)],:]
        
    def transpose(self, matrix):
        return matrix.transpose(1,2)
        
class MyOpticFlows():
    '''
        1. self-define transform rules, including resize, crop, flip. (crop and flip only for training set)
        2. training set augmentation with RandomCrop and RandomFlip.
        3. validation set using CenterCrop
    '''
    def __init__(self, args): 
        self.args = args
        
        [irradiance_map_height, irradiance_map_width] = args.reshape_size
        x = np.array([i for i in range(irradiance_map_width)])
        y = np.array([i for i in range(irradiance_map_height)])
        self.optic_flow = np.ones((3, irradiance_map_width*irradiance_map_height))
        self.X,self.Y = np.meshgrid(x, y)
        self.optic_flow[1:2] = self.Y.reshape((1, irradiance_map_height*irradiance_map_width))
        self.optic_flow[0:1] = self.X.reshape((1, irradiance_map_height*irradiance_map_width))
        self.irradiance_map_height = irradiance_map_height
        self.irradiance_map_width = irradiance_map_width
        
    def get_rotate_matrix_x(self, angle_x):
        return np.array([[1, 0                ,  0                ],
                         [0, math.cos(angle_x), -math.sin(angle_x)],
                         [0, math.sin(angle_x),  math.cos(angle_x)]])
                         
    def get_rotate_matrix_y(self, angle_y):
        return np.array([[ math.cos(angle_y), 0, math.sin(angle_y)],
                         [ 0,                 1, 0                ],
                         [-math.sin(angle_y), 0, math.cos(angle_y)]])
                         
    def get_rotate_matrix_z(self, angle_z):
        return np.array([[math.cos(angle_z), -math.sin(angle_z), 0],
                         [math.sin(angle_z),  math.cos(angle_z), 0],
                         [0                ,  0,                 1]])
        
    def getOpticFlows(self, size, t, tmax, params):
        [median_filter_size, gaussian_blur_sigma, theta0_deg, theta1_deg, x0, x1, y0, y1, sx0, sx1, sy0, sy1] = params
        
        #x0 = 0
        #x1 = 1
        #y0 = 0
        #y1 = 0
        #theta0_deg = 0
        #theta1_deg = 0
        
        [irradiance_map_height, irradiance_map_width] = size# W H
            
        
        theta0 = theta0_deg/180*math.pi
        theta1 = theta1_deg/180*math.pi
        
        dtheta = theta1-theta0
        dx = x1-x0
        dy = y1-y0
        dsx = sx1-sx0
        dsy = sy1-sy0
        
        theta = theta0+t/tmax*dtheta
        x = x0+t/tmax*dx
        y = y0+t/tmax*dy
        sx = sx0+t/tmax*dsx
        sy = sy0+t/tmax*dsy
        
        dtheta_dt = 1/tmax*dtheta
        dx_dt = 1/tmax*dx
        dy_dt = 1/tmax*dy
        dsx_dt = 1/tmax*dsx
        dsy_dt = 1/tmax*dsy
        
        
        
        
        affine_matrix = np.array([[sx*math.cos(theta), -sy*math.sin(theta), x],
                                  [sx*math.sin(theta),  sy*math.cos(theta), y],
                                  [0                   ,  0,                    1]])
        daffine_matrix = np.array([[dsx_dt*math.cos(theta)-dtheta_dt*math.sin(theta)*sx, -dsy_dt*math.sin(theta)-dtheta_dt*math.cos(theta)*sy, dx_dt],
                                   [dsx_dt*math.sin(theta)+dtheta_dt*math.cos(theta)*sx,  dsy_dt*math.cos(theta)-dtheta_dt*math.sin(theta)*sy, dy_dt],
                                   [0                                            ,  0                                            , 0]])
        translate_matrix_neg = np.array([[irradiance_map_width, 0                     ,irradiance_map_width*0.5],
                                     [0                    , irradiance_map_height, irradiance_map_height*0.5],
                                     [0                    , 0                    ,1                       ]])
        
        translate_matrix = np.array([[self.irradiance_map_width, 0                     , self.irradiance_map_width*0.5],
                                         [0                    , self.irradiance_map_height, self.irradiance_map_height*0.5],
                                         [0                    , 0                    ,1                       ]])
        affine_matrix = translate_matrix.dot(affine_matrix).dot(np.linalg.inv(translate_matrix_neg))
        
        daffine_matrix = translate_matrix.dot(daffine_matrix).dot(np.linalg.inv(translate_matrix_neg)).dot(np.linalg.inv(affine_matrix))
        optic_flow = self.optic_flow
        #print(optic_flow)
        optic_flow = daffine_matrix[0:2].dot(optic_flow)#-optic_flow[0:2]
        optic_flow = optic_flow.reshape((2, self.irradiance_map_height, self.irradiance_map_width))
        #optic_flow = optic_flow.transpose((0,2,1))
        #print(optic_flow)
        #return optic_flow
        return affine_matrix, optic_flow
