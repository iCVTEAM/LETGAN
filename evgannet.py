import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import imageio
import random
import time
import math
from torch.utils import model_zoo
from torchvision import models
from NetWorks.generatornet import EvSegNet
from NetWorks.discriminatornet import DiscriminatorNet
from NetWorks.utils import *
from NetWorks.__init__ import loadPertrainedParameter
from NetWorks.sync_batchnorm import convert_model
from scipy import stats

from torch.optim import SGD, Adam, lr_scheduler, RMSprop
from torchvision.transforms import ToPILImage
from PIL import Image
from torch.utils.data import DataLoader
from DataLoader.transform import MyTransform
from DataLoader.dataset import MyDataSet
#from Debug.functional import showMessage, outputMapWrite
from Criterion.criterion import Criterion

from sklearn.utils.linear_assignment_ import linear_assignment
from NetWorks.gradientnet import GradientNet
from NetWorks.contrastnet import ContrastNet
from tensorboardX import SummaryWriter

from adabelief_pytorch import AdaBelief

from utils.km import KuhnMunkres,Bayes

class EvGanNet(nn.Module):

    def __init__(self, args, savedir):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.debug_mode = args.debug
        self.patch = args.patch
        self.Qangle = args.Qangle
        self.Qstrength = args.Qstrength
        self.Qcoherence = args.Qcoherence
        
        self.savedir = savedir
        self.patch_size = args.size[0]//args.patch
        
        self.simulator = EvSegNet(self.args)
        self.discriminator = [DiscriminatorNet(self.patch_size) for i in range(0, self.Qangle*self.Qstrength*self.Qcoherence)]
            
        self.simulator = self.simulator.apply(weights_init)
        self.simulator = loadPertrainedParameter(args, self.simulator)
        for i in range(0, self.Qangle*self.Qstrength*self.Qcoherence):
            self.discriminator[i].apply(weights_init)
        
        #'''
        self.simulator = torch.nn.DataParallel(self.simulator)
        for i in range(0, self.Qangle*self.Qstrength*self.Qcoherence):
            torch.nn.DataParallel(self.discriminator[i])
            
        self.simulator = convert_model(self.simulator)
        for i in range(0, self.Qangle*self.Qstrength*self.Qcoherence):
            self.discriminator[i] = convert_model(self.discriminator[i])
        #'''
        self.simulator = self.simulator.to(self.device)
        for i in range(0, self.Qangle*self.Qstrength*self.Qcoherence):
            self.discriminator[i].to(self.device)
                
                                             
        self.transform = MyTransform(args=args, reshape_size=(346,260),crop_size=args.size,debug_mode=args.debug, train_mode=True)
        self.dataset_train = MyDataSet(args, self.transform, debug_mode=args.debug, train_mode=True)
        self.train_loader = DataLoader(self.dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True) #train data loader
            
        #save log
        self.automated_log_path = self.savedir + "/automated_log.txt"
        with open(self.automated_log_path, "a") as myfile:
            myfile.write("Epoch\tLoss-d-f\t\tLoss-d-r\t\tgrad-pena\t\tLoss-g\t\tLearnRate-d\t\tLearnRate-g")
                
        self.simulator.train()
        for i in range(0, self.Qangle*self.Qstrength*self.Qcoherence):
            self.discriminator[i].train()
        
        
        #self.simulator_optimizer = Adam(self.simulator.parameters(), args.g_lr, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)
        #self.simulator_optimizer = RMSprop(self.simulator.parameters(), lr=args.g_lr)
        '''
        param = dict(self.simulator.named_parameters())
        param_new = []
        for k,v in param.items():
            if '_s' in k:
                param_new += [{'params': [v], 'lr': args.g_lr}]
            else:
                param_new += [{'params': [v], 'lr': args.g_lr}]
        '''
        
        
        #self.simulator_optimizer = RMSprop(param_new)
        self.simulator_optimizer = AdaBelief(self.simulator.parameters(), lr=args.g_lr, eps=1e-12, betas=(0.9,0.999))
        '''
        for p in self.simulator_optimizer.param_groups:
            outputs = ''
            for k, v in p.items():
                if k is 'params':
                    outputs += (k + ': ' + str(v[0].shape).ljust(30) + ' ')
                else:
                    outputs += (k + ': ' + str(v).ljust(10) + ' ')
            print(outputs)
        '''
        adjust_lr = lambda iter: args.lr_end+iter/(len(self.train_loader)*args.warmup) if iter < (len(self.train_loader)*args.warmup) else args.lr_end+0.5*(1+math.cos((iter-(len(self.train_loader)*args.warmup))/(args.trainstep*len(self.train_loader)-len(self.train_loader)*args.warmup)*math.pi))
        self.simulator_scheduler = lr_scheduler.LambdaLR(self.simulator_optimizer, \
            lr_lambda=adjust_lr)    #  learning rate changed every epoch
        #lr_lambda=lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)
        
        #self.discriminator_optimizer = [Adam(self.discriminator[i].parameters(), args.d_lr, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4) \
        #    for i in range(0, self.Qangle*self.Qcoherence)]
        #self.discriminator_optimizer = [RMSprop(self.discriminator[i].parameters(), lr=args.d_lr) \
        #    for i in range(0, self.Qangle*self.Qcoherence)]
        self.discriminator_optimizer = [AdaBelief(self.discriminator[i].parameters(), lr=args.d_lr, eps=1e-12, betas=(0.9,0.999)) \
            for i in range(0, self.Qangle*self.Qstrength*self.Qcoherence)]
        self.discriminator_scheduler = [lr_scheduler.LambdaLR(self.discriminator_optimizer[i], lr_lambda=adjust_lr) \
            for i in range(0, self.Qangle*self.Qstrength*self.Qcoherence)]
        
        '''
        optimizer = Adam(model.parameters(), args.lr, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4) 
        lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)  
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)    #  learning rate changed every epoch       
        '''
        '''
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
        lambda1 = lambda epoch: pow(0.1,epoch//args.steps_loss+0)  
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)    #  learning rate changed every epoch       
        '''
        
        self.data_r = np.zeros((self.Qangle, self.Qstrength, self.Qcoherence, self.patch*self.patch*args.batch_size, 3), dtype = np.int32)
        self.index_r = np.zeros((self.Qangle, self.Qstrength, self.Qcoherence, 1), dtype = np.int32)
        
        self.data_f = np.zeros((self.Qangle, self.Qstrength, self.Qcoherence, self.patch*self.patch*args.batch_size, 3), dtype = np.int32)
        self.index_f = np.zeros((self.Qangle, self.Qstrength, self.Qcoherence, 1), dtype = np.int32)
        
        self.weight = np.ones((self.Qangle * self.Qstrength * self.Qcoherence), dtype = np.float32)/(self.Qangle * self.Qstrength * self.Qcoherence)
        self.weight_num = np.zeros((self.Qangle * self.Qstrength * self.Qcoherence), dtype = np.float32)
        
        # Matrix preprocessing
        # Preprocessing normalized Gaussian matrix W for hashkey calculation
        self.weighting = self.gaussian2d([self.patch_size, self.patch_size], 2)
        self.weighting = np.diag(self.weighting.ravel())
        #self.weighting_gpu = torch.from_numpy(self.weighting).float().to(self.device)
        
        self.gradient_model = GradientNet().to(self.device)
        self.contrast_model = ContrastNet().to(self.device)
        self.contrast_p = (4*(self.patch_size-2)*(self.patch_size-2)+3*(2*(self.patch_size-2)+2*(self.patch_size-2))+2*4)
        #self.km = KuhnMunkres(self.Qstrength)
        self.km = Bayes(self.Qstrength)

    def gaussian2d(self, shape=(3,3),sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h
        
    def hashTableGpu(self, gx, gy,Qangle,Qstrength,Qcoherence):
        #gx,gy = self.gradient_model.inference(patch)
        #G = torch.cat((gx.view(1, -1),gy.view(1, -1)), dim=0).permute(1,0)
        #x = G.permute(1,0).mm(self.weighting_gpu).mm(G)
        #print(self.weighting.shape, G.shape, x.shape)
        #x = G.T*self.weighting*G
        #u, s, v = torch.svd(x)
        #eigenvalues = s.cpu().data.numpy()
        #eigenvectors = v.cpu().data.numpy()
        
        G = np.matrix((gx.ravel(),gy.ravel())).T
        x = G.T.dot(self.weighting).dot(G)
        #print(self.weighting.shape, G.shape, x.shape)
        #x = G.T*self.weighting*G
        [eigenvalues,eigenvectors] = np.linalg.eig(x)
        '''
        # Make sure V and D contain only real numbers
        nonzerow = np.count_nonzero(np.isreal(eigenvalues))
        nonzerov = np.count_nonzero(np.isreal(eigenvectors))
        if nonzerow != 0:
            eigenvalues = np.real(eigenvalues)
        if nonzerov != 0:
            eigenvectors = np.real(eigenvectors)
        '''
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # Sort w and v according to the descending order of w
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
        
        #For angle
        angle = np.math.atan2(eigenvectors[1,0],eigenvectors[0,0])
        #print([eigenvalues,eigenvectors])
        if angle<0:
            angle += np.pi
        
        #For strength
        strength = 0 if eigenvalues.sum() == 0 else (eigenvalues[0]/(eigenvalues.sum())-0.5)*2
        
        if(eigenvalues[1]< 0):
            eigenvalues[1] = 0
        #For coherence
        lamda1 = np.math.sqrt(eigenvalues[0])
        lamda2 = np.math.sqrt(eigenvalues[1])
        coherence = 0 if lamda1+lamda2 == 0 else np.abs((lamda1-lamda2)/(lamda1+lamda2))
        
        #Quantization
        anglecopy = angle
        angle = np.floor(angle/(np.pi/Qangle))
        strength = np.floor(strength/(1.0/Qstrength))
        coherence = np.floor(coherence/(1.0/Qcoherence))
        
        #angle [0,Qangle]
        angle = 0 if angle < 0 else angle
        angle = Qangle-1 if angle >= Qangle else angle
        
        #strength [0,Qstrength]
        strength = 0 if strength < 0 else strength
        strength = Qstrength-1 if strength >= Qstrength else strength
        
        #coherence [0,Qcoherence]
        coherence = 0 if coherence < 0 else coherence
        coherence = Qcoherence-1 if coherence >= Qcoherence else coherence
        
        angle,strength,coherence = int(angle),int(strength),int(coherence)
        
        assert angle >= 0 and angle < Qangle, "angle overflow: {}".format(angle)
        assert strength >= 0 and strength < Qstrength, "strength overflow: {}".format(strength)
        assert coherence >= 0 and coherence < Qcoherence, "coherence overflow: {}".format(coherence)
        
        return angle,strength,coherence,anglecopy
        
    def hashTable(self, patch,Qangle,Qstrength,Qcoherence):
        [gx,gy] = np.gradient(patch)
        G = np.matrix((gx.ravel(),gy.ravel())).T
        x = G.T.dot(self.weighting).dot(G)
        #print(self.weighting.shape, G.shape, x.shape)
        #x = G.T*self.weighting*G
        [eigenvalues,eigenvectors] = np.linalg.eig(x)#0.00007
        
        # Make sure V and D contain only real numbers
        nonzerow = np.count_nonzero(np.isreal(eigenvalues))
        nonzerov = np.count_nonzero(np.isreal(eigenvectors))
        if nonzerow != 0:
            eigenvalues = np.real(eigenvalues)
        if nonzerov != 0:
            eigenvectors = np.real(eigenvectors)
            
        # Sort w and v according to the descending order of w
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
        
        #For angle
        angle = np.math.atan2(eigenvectors[1,0],eigenvectors[0,0])
        #print([eigenvalues,eigenvectors])
        if angle<0:
            angle += np.pi
        
        #For strength
        strength = 0 if eigenvalues.sum() == 0 else (eigenvalues[0]/(eigenvalues.sum())-0.5)*2
        
        #For coherence
        lamda1 = np.math.sqrt(eigenvalues[0])
        lamda2 = np.math.sqrt(eigenvalues[1])
        coherence = 0 if lamda1+lamda2 == 0 else np.abs((lamda1-lamda2)/(lamda1+lamda2))
        
        #Quantization
        angle = np.floor(angle/(np.pi/Qangle))
        strength = np.floor(strength/(1.0/Qstrength))
        coherence = np.floor(coherence/(1.0/Qcoherence))
        
        #angle [0,Qangle]
        angle = 0 if angle < 0 else angle
        angle = Qangle-1 if angle >= Qangle else angle
        
        #strength [0,Qstrength]
        strength = 0 if strength < 0 else strength
        strength = Qstrength-1 if strength >= Qstrength else strength
        
        #coherence [0,Qcoherence]
        coherence = 0 if coherence < 0 else coherence
        coherence = Qcoherence-1 if coherence >= Qcoherence else coherence
        
        angle,strength,coherence = int(angle),int(strength),int(coherence)
        
        assert angle >= 0 and angle < Qangle, "angle overflow: {}".format(angle)
        assert strength >= 0 and strength < Qstrength, "strength overflow: {}".format(strength)
        assert coherence >= 0 and coherence < Qcoherence, "coherence overflow: {}".format(coherence)

        return angle,strength,coherence
    '''
    def toStrength(self, strength_value):
        strength = strength_value/(strength_value.max()+0.0001)
        strength = np.floor(strength/(1.0/self.Qstrength)-1)
        return int(strength)
    '''
    '''
    def toStrength(self, event_max, event_mean):
        strength = 0 if event_max == 0 else event_mean/event_max
        strength = np.floor(strength/(1.0/self.Qstrength))
        strength = 0 if strength < 0 else strength
        strength = self.Qstrength-1 if strength >= self.Qstrength else strength
        return int(strength)
    '''
    def toStrength(self, event_pos_cg, event_neg_cg):
        strength = (event_pos_cg+event_neg_cg)/2
        strength = np.floor(strength/(1.0/self.Qstrength))
        strength = 0 if strength < 0 else strength
        strength = self.Qstrength-1 if strength >= self.Qstrength else strength
        return int(strength)
        
    def toMotion(self, angle, angle2):
        subangle = angle - angle2
        if subangle<0:
            subangle += np.pi
        motion = math.cos(subangle)
        motion = np.floor((motion+1)/(2.0/self.Qstrength))
        motion = 0 if motion < 0 else motion
        motion = self.Qstrength-1 if motion >= self.Qstrength else motion
        return int(motion)
    
    def contrast(self, patch, p=0):
        m, n = patch.shape
        b = 0.0
        for i in range(1,m-1):
            for j in range(1,n-1):
                if patch[i,j] != 0:
                    b += ((patch[i,j]-patch[i,j+1])**2 + (patch[i,j]-patch[i,j-1])**2 + (patch[i,j]-patch[i+1,j])**2 + (patch[i,j]-patch[i-1,j])**2)
        if p==0:
            p = (4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+2*4)
        cg = b/p
        #if(cg > 1):
        #    print(cg)
        return cg
    
    def classifyfast(self, patchx, patchy, event_pos_cg, event_neg_cg, patch_channel, patch_x, patch_y, label):
        angle, strength, coherence,anglecopy = self.hashTableGpu(patchx, patchy, self.Qangle, self.Qstrength, self.Qcoherence)
        #angle2, strength2, coherence2,anglecopy2 = self.hashTableGpu(patchx2, patchy2, self.Qangle, self.Qstrength, self.Qcoherence)
        #strength = self.toMotion(anglecopy, anglecopy2)
        #print(angle, strength, coherence)
        #print(event_pos_cg, event_neg_cg)
        strength = self.toStrength(event_pos_cg, event_neg_cg)
        '''
        strength_event = self.toStrength(event_pos_cg, event_neg_cg)
        if strength > strength_event:
            strength = strength - 1
        elif strength < strength_event:
            strength = strength + 1
        '''
        '''
        if label:
            strength_event = self.toStrength(event_pos_cg, event_neg_cg)
            #print(strength,strength_event)
            strength = self.km.get_add_path(strength,strength_event)
            
        else:
            strength = self.km.get_path(strength)
        #strength = 0
        #print(angle, strength, coherence)
        '''
        if label:
            self.data_r[angle, strength, coherence, self.index_r[angle, strength, coherence]] = [patch_channel, patch_x, patch_y]
            self.index_r[angle, strength, coherence] += 1
        else:
            self.data_f[angle, strength, coherence, self.index_f[angle, strength, coherence]] = [patch_channel, patch_x, patch_y]
            self.index_f[angle, strength, coherence] += 1
    
    '''
    def classifyfast(self, patchx, patchy, event_max, event_mean, patch_channel, patch_x, patch_y, label):
        angle, strength, coherence = self.hashTableGpu(patchx, patchy, self.Qangle, self.Qstrength, self.Qcoherence)
        #print(angle, strength, coherence)
        if label:
            strength_event = self.toStrength(event_max, event_mean)
            #print(strength,strength_event)
            strength = self.km.get_add_path(strength,strength_event)
            
        else:
            strength = self.km.get_path(strength)
        #strength = 0
        #print(angle, strength, coherence)
        if label:
            self.data_r[angle, strength, coherence, self.index_r[angle, strength, coherence]] = [patch_channel, patch_x, patch_y]
            self.index_r[angle, strength, coherence] += 1
        else:
            self.data_f[angle, strength, coherence, self.index_f[angle, strength, coherence]] = [patch_channel, patch_x, patch_y]
            self.index_f[angle, strength, coherence] += 1
    '''
        #except:
        #    print('INFO: evgannet classify ', self.patch*self.patch*2*self.args.batch_size,self.patch,self.args.batch_size,self.index[angle, coherence])
        
        #strengthDes = eventPatch.max()
        #if label:
        #    self.costMatrix[strength, strengthDesandom] -= 1
    
    def classify(self, img_patch, patch_channel, patch_x, patch_y, label):
        #st = time.time()
        angle, strength, coherence = self.hashTable(img_patch, self.Qangle, self.Qstrength, self.Qcoherence)
        #print('classify:',time.time()-st)
        if label:
            self.data_r[angle, coherence, self.index_r[angle, coherence]] = [patch_channel, patch_x, patch_y]
            self.index_r[angle, coherence] += 1
        else:
            self.data_f[angle, coherence, self.index_f[angle, coherence]] = [patch_channel, patch_x, patch_y]
            self.index_f[angle, coherence] += 1
        #except:
        #    print('INFO: evgannet classify ', self.patch*self.patch*2*self.args.batch_size,self.patch,self.args.batch_size,self.index[angle, coherence])
        
        #strengthDes = eventPatch.max()
        #if label:
        #    self.costMatrix[strength, strengthDesandom] -= 1
        
    def preparedData(self, angle, strength, coherence, map, label, is_d=True):
        #if self.debug_mode:
        #    print('INFO: evgannet preparedData ', angle, coherence, self.index[angle, coherence])
        if label:
            randList = [i for i in range(0, self.index_r[angle, strength, coherence, 0])]
            random.shuffle(randList)
            data = self.data_r[angle, strength, coherence,randList]
        else:
            randList = [i for i in range(0, self.index_f[angle, strength, coherence, 0])]
            random.shuffle(randList)
            data = self.data_f[angle, strength, coherence,randList]
        
        if data.shape[0]:
            output = map[data[0,0]:data[0,0]+1,:,data[0,1]:data[0,1]+self.patch_size, data[0,2]:data[0,2]+self.patch_size]
        else:
            return torch.zeros((0,0,0,0))
        num = self.args.d_batch_size if is_d and self.args.d_batch_size < data.shape[0] else data.shape[0]
        for i in range(1, num):
            output = torch.cat((output, map[data[i,0]:data[i,0]+1,:,data[i,1]:data[i,1]+self.patch_size, data[i,2]:data[i,2]+self.patch_size]),0)
        return output
                
    def computeAcc(self, output, label):
        #output = output.max(1)[1] #index of max-channel
        
        if self.is_cuda_available:
            label = label.cpu()
            output = output.cpu()
        label = label.data.numpy()
        output = output.data.numpy()
        #print(output,label)
        output = np.around(output).astype(np.int32)
        output[output>1]=1
        output[output<0]=0
        
        
        #print(output,label)
        
        check_list = (output-label).tolist()
        #print(check_list)
        right_num = check_list.count(0)
        accuracy = 1.0*right_num/len(label)

        
        return accuracy
        
    '''
    update_option
    0:update discriminator
    1:update generatornet
    '''
    
    def zero_grad(self,update_option):
        if update_option:
            self.simulator_optimizer.zero_grad()
        else:
            for i in range(0, self.Qangle*self.Qstrength*self.Qcoherence):
                self.discriminator_optimizer[i].zero_grad()
                
    def lr_step(self,epoch):
        self.simulator_scheduler.step(epoch)
        for i in range(0, self.Qangle*self.Qstrength*self.Qcoherence):
            self.discriminator_scheduler[i].step(epoch)
                
    def step(self, update_option):
        if update_option:
            self.simulator_optimizer.step()
        else:
            for i in range(0, self.Qangle*self.Qstrength*self.Qcoherence):
                self.discriminator_optimizer[i].step()
                
    def getLr(self, update_option):
        if update_option:
            return self.simulator_optimizer.param_groups[0]['lr']
        else:
            return self.discriminator_optimizer[0].param_groups[0]['lr']
            
    def clipParam(self, update_option):
        if update_option:
            return
            
        for i in range(0, self.Qangle*self.Qstrength*self.Qcoherence):
            # modification: clip param for discriminator
            for parm in self.discriminator[i].parameters():
                parm.data.clamp_(-self.args.clamp_num,self.args.clamp_num)
                
    def gradientPenalty(self, real_event_maps, simulated_event_maps, discriminator):
        # gradient penalty
        batch_size = real_event_maps.size()[0] if real_event_maps.size()[0] < simulated_event_maps.size()[0] else simulated_event_maps.size()[0]
        alpha = torch.rand((batch_size, 1, 1, 1)).to(self.device)

        interpolates = alpha * real_event_maps[0:batch_size] + (1 - alpha) * simulated_event_maps[0:batch_size]
        interpolates.requires_grad = True

        disc_interpolates = discriminator(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]

        #print(real_event_maps.size())
        #print(simulated_event_maps.size())
        #print(interpolates.size())
        #print(disc_interpolates.size())
        #print(gradients.size())
        gradient_penalty = self.args.gp_lambda * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
        return gradient_penalty
        
    def channel2to3(self, output2):
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
        output3[output3>1] = output3[output3>1]+125
        output3[output3>255]=255
        #max_value = output3.max()
        
        #return np.trunc(output3/max_value*255)
        return np.trunc(output3)
    def gradient(self, x):
        g_pos = self.gradient_model(x[:,0:1])
        g_neg = self.gradient_model(x[:,1:2])
        return torch.cat((g_pos, g_neg),dim=1)
        
    def processBeforeDis(self, event, img1, img2):
        '''
        #event_map = event[:,0:1]-event[:,1:2]
        grad_map = self.gradient(event)
        grad_map = F.tanh(grad_map/100)
        event_map = F.tanh(event)
        eg_map = torch.cat((event_map, grad_map), dim=1)
        return eg_map
        '''
        grad_map = self.gradient(event)
        #i_map = torch.cat(((img1-math.log(self.args.log_eps))/(math.log(self.args.log_eps+1)-math.log(self.args.log_eps)), \
        #    (img2-math.log(self.args.log_eps))/(math.log(self.args.log_eps+1)-math.log(self.args.log_eps))), dim=1)
        #i_map = (img2-img1)/(math.log(self.args.log_eps+1)-math.log(self.args.log_eps))
        p_map = F.tanh(grad_map)
        #ip_map = torch.cat((i_map, p_map), dim=1)
        #return ip_map
        e_map = F.tanh(event)
        ep_map = torch.cat((e_map, p_map), dim=1)
        return ep_map
        
    def train(self):
        writer = SummaryWriter(comment='ED', log_dir=self.args.tb_path)
        '''
        dummyInput1 = torch.randn(1,3,self.args.size[0], self.args.size[1])
        dummyInput1 = dummyInput1.cuda() if self.is_cuda_available else dummyInput1
        
        dummyInput2 = torch.randn(1,1,self.args.size[0], self.args.size[1])
        dummyInput2 = dummyInput2.cuda() if self.is_cuda_available else dummyInput2
        
        dummyInput3 = torch.randn(1,2,self.args.size[0], self.args.size[1])
        dummyInput3 = dummyInput3.cuda() if self.is_cuda_available else dummyInput3
        writer.add_graph(self.simulator, (dummyInput1, dummyInput2, dummyInput3))
        '''
        start_epoch = 0
        #criterion = nn.SmoothL1Loss()
        #criterion = nn.KLDivLoss(reduction='mean')
        #criterion = nn.MSELoss(reduction='mean')
        #criterion = nn.BCEWithLogitsLoss()
        #true_threshold_C_pos = torch.zeros(self.args.batch_size, 1, 224, 224).cuda()+0.2
        
        for epoch in range(start_epoch, self.args.num_epochs):
            print("----- TRAINING - EPOCH", epoch, "-----")
            epoch_loss_d_f = []
            epoch_loss_d_r = []
            epoch_loss_g = []
            epoch_gp = []
            epoch_time = []
            start_time = time.time()
            #epoch_acc = []
            
            used_lr_d = self.getLr(0)
            print("LEARNING RATE discriminator: ", used_lr_d)
            used_lr_g = self.getLr(1)
            print("LEARNING RATE generator    : ", used_lr_g)
            
            #for step, (rgb_images_cpu, irradiance_maps_cpu, optic_flows_cpu, real_event_images_cpu, real_event_maps_cpu) in enumerate(train_loader):
            '''
            for step, (images_cpu, irradiance_maps_cpu, optic_flows_cpu, real_event_images_cpu, real_event_maps_cpu) in enumerate(self.train_loader):
                images, irradiance_maps, optic_flows, real_event_images, real_event_maps = \
                images_cpu.to(self.device), \
                irradiance_maps_cpu.to(self.device), \
                optic_flows_cpu.to(self.device), \
                real_event_images_cpu.to(self.device), \
                real_event_maps_cpu.to(self.device)
            '''
            for step, (irradiance_maps_cpu, real_event_images_cp, real_event_maps_cp, real_event_images2_cp) in enumerate(self.train_loader):
                iter = epoch*len(self.train_loader)+step
                self.lr_step(iter)
                #print(irradiance_maps_cpu.size(), real_event_images_cpu.size(), real_event_maps_cpu.size())
                #irradiance_maps, real_event_images, real_event_maps, real_event_images2 = \
                #irradiance_maps_cpu.to(self.device), \
                #real_event_images_cpu.to(self.device), \
                #real_event_maps_cpu.to(self.device), \
                #real_event_images2_cpu.to(self.device)
                irradiance_maps = irradiance_maps_cpu.to(self.device)
                
                #imageio.imwrite('simImg.bmp', irradiance_maps[0,0].squeeze().cpu().data.numpy())
                #imageio.imwrite('realImg.bmp', real_event_images[0].squeeze().cpu().data.numpy())
                #imageio.imwrite('realEve.bmp', (self.channel2to3(real_event_maps_cpu)).astype(np.uint8).transpose((1,2,0)))
                #return
                
                #simulated_event_maps, real_event_images, real_event_maps, real_event_images_next, images, \
                #    real_threshold_C_pos, fake_threshold_C_pos, real_threshold_C_neg, fake_threshold_C_neg  = self.simulator(irradiance_maps, optic_flows)
                a, simu_images, simu_images2, simulated_event_maps, fake_threshold_C_pos, fake_threshold_C_neg  = self.simulator(irradiance_maps,0)
                irradiance_maps, real_event_images, real_event_images2, real_event_maps,b,c  = self.simulator(irradiance_maps, 1)
                real_event_maps_cpu = real_event_maps.cpu().data.numpy()
                #return
                
                #print(simulated_event_maps.size())
                #print(irradiance_maps.size(), simulated_event_maps.size(), fake_threshold_C_pos.size(), fake_threshold_C_neg.size())
                '''
                
                #normal_sample = torch.randn(fake_threshold_C_pos.size())
                #if self.is_cuda_available:
                #    normal_sample = normal_sample.cuda()
                #true_threshold_C_pos = normal_sample * 0.05 + 0.2
                loss_g = criterion(self.processBeforeDis(simulated_event_maps,simulated_event_maps), self.processBeforeDis(real_event_maps,real_event_maps))
                #print(fake_threshold_C_pos.size(), true_threshold_C_pos.size())
                #loss_g = criterion(fake_threshold_C_pos, true_threshold_C_pos)
                #loss_g = criterion(F.tanh(simulated_event_maps), F.tanh(real_event_maps))
                self.zero_grad(1)
                loss_g.backward()
                self.step(1)
                loss_d_r = loss_g
                loss_d_f = loss_g
                gradient_penalty = loss_g
                up_dis_num = 1
                #'''
                '''
                
                up_dis_num = 1
                rem = self.processBeforeDis(real_event_maps.clone().detach(), real_event_images, real_event_images2)
                output = self.discriminator[0](rem)
                loss_d_r = (-output.mean())
                
                sem = self.processBeforeDis(simulated_event_maps.clone().detach(), simu_images, simu_images2)
                output = self.discriminator[0](sem)
                loss_d_f = (output.mean())
                
                gradient_penalty = self.gradientPenalty(rem, sem, self.discriminator[0])
                
                loss_d = loss_d_r + loss_d_f + gradient_penalty
                self.zero_grad(0)
                loss_d.backward()
                self.step(0)
                #self.clipParam(0)
                
                if step % self.args.train_g_interval == 0:
                    output = self.discriminator[0](self.processBeforeDis(simulated_event_maps, simu_images, simu_images2))
                    loss_g = (-output.mean())
                    #mu:fake_threshold_C_pos, log_var:fake_threshold_C_neg
                    #kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                    kl_div = 0.00001*(-0.5 * torch.sum(1 + fake_threshold_C_neg - fake_threshold_C_pos.pow(2) - fake_threshold_C_neg.exp()))
                    kl_div_show = (-0.5 * torch.mean(1-math.log(0.0025) + fake_threshold_C_neg.pow(2).log() - (fake_threshold_C_pos.pow(2) + fake_threshold_C_neg.pow(2) -0.4*fake_threshold_C_pos + 0.04)/0.0025))
                    
                    #loss_t = criterion(self.processBeforeDis(simulated_event_maps), self.processBeforeDis(real_event_maps))
                    #loss_g += loss_t
                    self.zero_grad(1)
                    #(loss_g+kl_div).backward()
                    loss_g.backward()
                    self.step(1)
                #'''
                simu_images_cpu = simu_images.cpu().data.numpy()
                simulated_event_maps_cpu = simulated_event_maps.cpu().data.numpy()
                
                #real_event_images_cpu = real_event_images.cpu().data.numpy()
                #real_event_maps_cpu = real_event_maps.cpu().data.numpy()
                
                #imageio.imwrite('simImg2.bmp', ((images[0,0].cpu().numpy()*0.226+0.449)*255).astype(np.uint8))
                #imageio.imwrite('simEve.bmp', (self.channel2to3(simulated_event_maps_cpu)).astype(np.uint8).transpose((1,2,0)))
                #imageio.imwrite('simImg.bmp', irradiance_maps[0,0].squeeze().cpu().data.numpy())
                #imageio.imwrite('realImg.bmp', real_event_images[0].squeeze().cpu().data.numpy())
                #imageio.imwrite('realEve.bmp', (self.channel2to3(real_event_maps_cpu)).astype(np.uint8).transpose((1,2,0)))
                #return
                
                if (self.debug_mode):
                    print('INFO: evgannet train ')
                    img = ((images[0,0].cpu().numpy()*0.226+0.449)*255).astype(np.uint8)
                    imageio.imwrite('simImgOld.bmp', img)
                    imageio.imwrite('simImg.bmp', irradiance_maps[0].squeeze().cpu().data.numpy())
                    imageio.imwrite('simEve.bmp', (self.channel2to3(simulated_event_maps_cpu)).astype(np.uint8).transpose((1,2,0)))
                    imageio.imwrite('realImg.bmp', real_event_images[0].squeeze().cpu().data.numpy())
                    imageio.imwrite('realEve.bmp', (self.channel2to3(real_event_maps_cpu)).astype(np.uint8).transpose((1,2,0)))
                    print('real num pos neg', F.relu(real_event_maps_cpu).sum(),F.relu(-real_event_maps_cpu).sum())
                    print('fake num pos neg', F.relu(simulated_event_maps).sum(),F.relu(-simulated_event_maps).sum())
                    print('real before',real_event_images[0,0,56,24])
                    print('real before 1',img[56,24])
                    print('real event',real_event_maps_cpu[0,0,56,24])
                    print('fake event',simulated_event_maps_cpu[0,0,56,24])
                    return
                #'''
                #simulated_strengths = self.pool(simulated_event_maps[:,0:1] + simulated_event_maps[:,0:2])
                #real_strengths = self.pool(real_event_maps[:,0:1] + real_event_maps[:,0:2])
                
                #simulated_strengths_cpu = simulated_strengths.cpu().data.numpy()
                #real_strengths_cpu = real_strengths.cpu().data.numpy()
                
                #simulated_strengths_cpu = self.toStrength(simulated_strengths_cpu)
                #real_strengths_cpu = self.toStrength(real_strengths_cpu)
                
                loss_d_r = 0
                accuracy_d_r = 0
                up_dis_num = 0
                #st = time.time()
                gx,gy = self.gradient_model.inference(real_event_images)
                gx = gx.cpu().data.numpy()
                gy = gy.cpu().data.numpy()
                
                #gx2,gy2 = self.gradient_model.inference(real_event_images2)
                #gx2 = gx2.cpu().data.numpy()
                #gy2 = gy2.cpu().data.numpy()
                
                cg = self.contrast_model.inference(real_event_maps)
                cg = self.contrast_model.inference(real_event_images)
                cg = cg.cpu().data.numpy()
                for index in range(0, irradiance_maps_cpu.shape[0]):
                    for x in range(0, irradiance_maps_cpu.shape[2], self.patch_size):
                        for y in range(0, irradiance_maps_cpu.shape[3], self.patch_size):
                            #if random.random() < 0.5:
                            #    continue
                            #img_patch = irradiance_maps_cpu[index,0,x:x+self.patch_size, y:y+self.patch_size]
                            #event_patch = simulated_event_maps_cpu[index,:,x:x+self.patch_size, y:y+self.patch_size]
                            ##self.classify(img_patch, simulated_strengths_cpu[index, 0, x//self.patch, y//self.patch], label = 0)
                            #self.classify(img_patch, event_patch, label = 0)

                            #img_patch = real_event_images_cpu[index,0,x:x+self.patch_size, y:y+self.patch_size]
                            patchx = gx[index,0,x:x+self.patch_size, y:y+self.patch_size]
                            patchy = gy[index,0,x:x+self.patch_size, y:y+self.patch_size]
                            
                            #patchx2 = gx2[index,0,x:x+self.patch_size, y:y+self.patch_size]
                            #patchy2 = gy2[index,0,x:x+self.patch_size, y:y+self.patch_size]
                            event_patch = real_event_maps_cpu[index,:,x:x+self.patch_size, y:y+self.patch_size]
                            patchcg = cg[index,:,x:x+self.patch_size, y:y+self.patch_size]
                            #self.classify(img_patch, real_strengths_cpu[index, 0, x//self.patch, y//self.patch], label = 1)
                            if event_patch.sum() >= self.args.min_event_num:
                                #print(event_patch.numpy().sum()/np.count_nonzero(event_patch.numpy()), stats.mode(event_patch.numpy().reshape(-1))[0][0], stats.mode(event_patch.numpy().reshape(-1)) )
                                #counts = np.bincount(int(event_patch.numpy().reshape(1,-1)))
                                #print(counts,np.argmax(counts))
                                #print(patchcg[0],patchcg[0].sum())
                                #self.classifyfast(patchx, patchy, self.contrast(event_patch.numpy()[0]), self.contrast(event_patch.numpy()[1]), index, x, y, label = 1)
                                self.classifyfast(patchx, patchy, patchcg[0].sum()/self.contrast_p, patchcg[0].sum()/self.contrast_p, index, x, y, label = 1)
                                #self.classifyfast(patchx, patchy, patchx2, patchy2, index, x, y, label = 1)
                #print('1:', time.time()-st)
                #st = time.time()
                #indices = linear_assignment(self.cost_matrix)
                for i in range(0, self.Qangle):
                    for j in range(0, self.Qstrength):
                        for k in range(0, self.Qcoherence):
                            data = self.preparedData(i,j,k, torch.cat((real_event_images, real_event_maps, real_event_images2),dim=1).detach(), label = 1)
                            #print(real_event_maps.size(), data.size())
                            is_data_available = data.size()[0]
                            #print('INFO: evgannet train ', i, j, is_data_available)
                            #if self.debug_mode:
                            #    print('INFO: evgannet train ', i, j, is_data_available)
                            #    time.sleep(0.1)
                            if is_data_available:
                                output = self.discriminator[i*self.Qcoherence*self.Qstrength+j*self.Qcoherence+k](self.processBeforeDis(data[:,1:3], data[:,0:1], data[:,3:4]))
                                #loss_d_r += self.criterion(output, label)
                                loss_d_r += (-output.mean()*self.weight[i*self.Qcoherence*self.Qstrength+j*self.Qcoherence+k])
                                #accuracy_d_r += self.computeAcc(output.squeeze(),label.squeeze())
                                #up_dis_num += 1
                #print('2:', time.time()-st)
                #st = time.time()
                
                loss_d_f = 0
                accuracy_d_f = 0
                gx,gy = self.gradient_model.inference(simu_images)
                gx = gx.cpu().data.numpy()
                gy = gy.cpu().data.numpy()
                
                #gx2,gy2 = self.gradient_model.inference(simu_images2)
                #gx2 = gx2.cpu().data.numpy()
                #gy2 = gy2.cpu().data.numpy()
                
                cg = self.contrast_model.inference(simulated_event_maps)
                cg = self.contrast_model.inference(simu_images)
                cg = cg.cpu().data.numpy()
                for index in range(0, irradiance_maps_cpu.shape[0]):
                    for x in range(0, irradiance_maps_cpu.shape[2], self.patch_size):
                        for y in range(0, irradiance_maps_cpu.shape[3], self.patch_size):
                            #if random.random() < 0.5:
                            #    continue
                            #img_patch = simu_images_cpu[index,0,x:x+self.patch_size, y:y+self.patch_size]
                            patchx = gx[index,0,x:x+self.patch_size, y:y+self.patch_size]
                            patchy = gy[index,0,x:x+self.patch_size, y:y+self.patch_size]
                            
                            #patchx2 = gx2[index,0,x:x+self.patch_size, y:y+self.patch_size]
                            #patchy2 = gy2[index,0,x:x+self.patch_size, y:y+self.patch_size]
                            event_patch = simulated_event_maps_cpu[index,:,x:x+self.patch_size, y:y+self.patch_size]
                            patchcg = cg[index,:,x:x+self.patch_size, y:y+self.patch_size]
                            #event_patch = np.zeros((2,self.patch_size, self.patch_size))
                            #self.classify(img_patch, simulated_strengths_cpu[index, 0, x//self.patch, y//self.patch], label = 0)
                            if event_patch.sum() >= self.args.min_event_num:
                                #self.classify(img_patch, index, x, y, label = 0)
                                #self.classifyfast(patchx, patchy, self.contrast(event_patch[0]), self.contrast(event_patch[1]), index, x, y, label = 0)
                                self.classifyfast(patchx, patchy, patchcg[0].sum()/self.contrast_p, patchcg[0].sum()/self.contrast_p, index, x, y, label = 0)
                                #self.classifyfast(patchx, patchy, patchx2, patchy2, index, x, y, label = 0)
                #print('3:', time.time()-st)
                #st = time.time()
                for i in range(0, self.Qangle):
                    for j in range(0, self.Qstrength):
                        for k in range(0, self.Qcoherence):
                            data = self.preparedData(i,j,k, torch.cat((simu_images, simulated_event_maps, simu_images2),dim=1).detach(), label = 0)
                            #print(data.size())
                            #output = self.discriminator[i*self.Qangle+indices[j]](data)
                            is_data_available = data.size()[0]
                            #if self.debug_mode:
                            #    print('INFO: evgannet train ', data.size(), label.size(), is_data_available)
                            if is_data_available:
                                output = self.discriminator[i*self.Qcoherence*self.Qstrength+j*self.Qcoherence+k](self.processBeforeDis(data[:,1:3], data[:,0:1], data[:,3:4]))
                                loss_d_f += (output.mean()*self.weight[i*self.Qcoherence*self.Qstrength+j*self.Qcoherence+k])
                                #accuracy_d_f += self.computeAcc(output.squeeze(),label.squeeze())
                                #print(output.squeeze(),label.squeeze())
                                #up_dis_num += 1
                #print('4:', time.time()-st)
                #st = time.time()
                #print(output.squeeze(),label.squeeze())
                weight = np.zeros((self.Qangle * self.Qstrength * self.Qcoherence), dtype = np.float32)
                gradient_penalty = 0
                for i in range(0, self.Qangle):
                    for j in range(0, self.Qstrength):
                        for k in range(0, self.Qcoherence):
                            data_r = self.preparedData(i,j,k, torch.cat((real_event_images, real_event_maps, real_event_images2),dim=1).detach(), label = 1)
                            data_f = self.preparedData(i,j,k, torch.cat((simu_images, simulated_event_maps, simu_images2),dim=1).detach(), label = 0)
                            #output = self.discriminator[i*self.Qangle+indices[j]](data)
                            is_data_available = data_r.size()[0] if data_r.size()[0] < data_f.size()[0] else data_f.size()[0]
                            #if self.debug_mode:
                            #print('INFO: evgannet train ', i, j, k, is_data_available)
                            if is_data_available:
                                #print(data_r.size(),data_f.size())
                                gradient_penalty += (self.gradientPenalty(self.processBeforeDis(data_r[:,1:3], data_r[:,0:1], data_r[:,3:4]), 
                                    self.processBeforeDis(data_f[:,1:3], data_f[:,0:1], data_f[:,3:4]), self.discriminator[i*self.Qcoherence*self.Qstrength+j*self.Qcoherence+k])*self.weight[i*self.Qcoherence*self.Qstrength+j*self.Qcoherence+k])
                                up_dis_num += 1
                                weight[i*self.Qcoherence*self.Qstrength+j*self.Qcoherence+k] = is_data_available
                self.weight = (self.weight + weight / weight.sum())/2
                
                #print('5:', time.time()-st)
                #st = time.time()
                #print(loss_d_r.item(), loss_d_f.item())
                loss_d = loss_d_r + loss_d_f + gradient_penalty
                self.zero_grad(0)
                loss_d.backward()
                self.step(0)
                #self.clipParam(0)
                #todo
                #loss_g = loss_d
                #accuracy_g = 0
                if step % self.args.train_g_interval == 0:
                    #clear self.data
                    #self.index_r = np.zeros((self.Qangle, self.Qcoherence, 1), dtype = np.int32)
                    
                    #simulated_event_maps, real_event_images, real_event_maps = self.simulator(rgb_images, irradiance_maps, optic_flows)
                    #simulated_event_maps_cpu = simulated_event_maps.cpu().data.numpy()
                    loss_g = 0
                    accuracy_g = 0
                    pass_dis_num = 0
                    #for index in range(0, irradiance_maps_cpu.shape[0]):
                    #    for x in range(0, irradiance_maps_cpu.shape[2], self.patch_size):
                    #        for y in range(0, irradiance_maps_cpu.shape[3], self.patch_size):
                    #            img_patch = irradiance_maps_cpu[index,0,x:x+self.patch_size, y:y+self.patch_size]
                    #            event_patch = simulated_event_maps_cpu[index,:,x:x+self.patch_size, y:y+self.patch_size]
                    #            #self.classify(img_patch, simulated_strengths_cpu[index, 0, x//self.patch, y//self.patch], label = 0)
                    #            self.classify(img_patch, event_patch, label = 1)

                    for i in range(0, self.Qangle):
                        for j in range(0, self.Qstrength):
                            for k in range(0, self.Qcoherence):
                                data = self.preparedData(i,j,k, torch.cat((simu_images, simulated_event_maps, simu_images2),dim=1), label = 0, is_d=False)
                                #output = self.discriminator[i*self.Qangle+indices[j]](data)
                                is_data_available = data.size()[0]
                                #if self.debug_mode:
                                #    print('INFO: evgannet train ', data.size(), label.size(), is_data_available)
                                if is_data_available:
                                    output = self.discriminator[i*self.Qcoherence*self.Qstrength+j*self.Qcoherence+k](self.processBeforeDis(data[:,1:3], data[:,0:1], data[:,3:4]))
                                    loss_g += (-output.mean()*self.weight[i*self.Qcoherence*self.Qstrength+j*self.Qcoherence+k])
                                    #accuracy_g += self.computeAcc(output.squeeze(),label.squeeze())
                                    pass_dis_num += 1
                                
                    #self.weight_num += weight
                    #weight_num_max = 100000
                    #self.weight_num[self.weight_num > weight_num_max] = weight_num_max
                    #normal_sample = torch.randn(fake_threshold_C_pos.size())
                    #if self.is_cuda_available:
                    #    normal_sample = normal_sample.cuda()
                    #true_threshold_C_pos = normal_sample * 0.05 + 0.2
                    #loss_t = criterion(fake_threshold_C_pos, true_threshold_C_pos)
                    #loss_g += loss_t
                    #print(loss_g.item())
                    #L2 VAE
                    #tmp_sigma = 0.03
                    #tmp_mean = 0.2
                    kl_div_show = (-0.5 * torch.mean(1-math.log(0.0025) + fake_threshold_C_neg.pow(2).log() - (fake_threshold_C_pos.pow(2) + fake_threshold_C_neg.pow(2) -0.4*fake_threshold_C_pos + 0.04)/0.0025))
                                        
                    kl_div = 0.00001*(-0.5 * torch.sum(1 + fake_threshold_C_neg - fake_threshold_C_pos.pow(2) - fake_threshold_C_neg.exp()))
                    #kl_div = 0.00000001*(-0.5 * torch.sum(fake_threshold_C_neg.pow(2).log() - fake_threshold_C_pos.pow(2) - fake_threshold_C_neg.pow(2) + 2*fake_threshold_C_pos))
                    #kl_div = 0.00000001*(-0.5 * torch.sum(1 + fake_threshold_C_neg.pow(2).log() - (fake_threshold_C_pos.pow(2) + fake_threshold_C_neg.pow(2))))
                    self.zero_grad(1)
                    (loss_g+kl_div).backward()
                    #loss_g.backward()
                    self.step(1)
                #print('6:', time.time()-st)
                #clear self.data
                self.index_r = np.zeros((self.Qangle, self.Qstrength, self.Qcoherence, 1), dtype = np.int32)
                self.index_f = np.zeros((self.Qangle, self.Qstrength, self.Qcoherence, 1), dtype = np.int32)
                self.km.clean()
                #'''
                if (self.debug_mode):
                    pass
                
                #accuracy /= up_dis_num
                epoch_loss_d_f.append(loss_d_f.item())
                epoch_loss_d_r.append(loss_d_r.item())
                epoch_gp.append(gradient_penalty.item())
                epoch_loss_g.append(loss_g.item())
                #epoch_acc.append(accuracy)
                
                if self.args.show_interval > 0 and (step+1) % self.args.show_interval == 0:
                    epoch_time.append(time.time() - start_time)
                    start_time = time.time()
                    print('TRAIN Epoch: %d [%d/%d] Loss_kl_show: %.3f \tLoss_kl: %.3f \tLoss_d_f: %.3f \tLoss_d_r: %.3f \tLoss_g: %.3f \n\tgp: %.3f\tUDN: %d time: %.3f\tweight: %.8f \t\t%.8f'%(epoch,
                        (step+1)*self.args.batch_size, len(self.train_loader)*self.args.batch_size,
                        kl_div_show.item(),
                        kl_div.item(),
                        epoch_loss_d_f[-1], 
                        epoch_loss_d_r[-1], 
                        epoch_loss_g[-1], 
                        epoch_gp[-1], 
                        up_dis_num,
                        (epoch_time[-1]),
                        self.weight.min(), self.weight.max()))
                    #print('accuracy_d_r: %.3f, accuracy_d_f: %.3f, accuracy_g: %.3f'%(accuracy_d_r, accuracy_d_f, accuracy_g))
                        
                    writer.add_scalars('Loss', {'Loss-kl-show':kl_div_show.item()}, step+epoch*len(self.train_loader))
                    writer.add_scalars('Loss', {'Loss-kl':kl_div.item(), 'Loss-d-f':epoch_loss_d_f[-1], 'Loss-d-r':epoch_loss_d_r[-1], 'Loss-g':epoch_loss_g[-1]}, step+epoch*len(self.train_loader))
                    #writer.add_scalars('Loss', {'Loss-d-f':epoch_loss_d_f[-1], 'Loss-d-r':epoch_loss_d_r[-1], 'Loss-g':epoch_loss_g[-1]}, step+epoch*len(self.train_loader))
                    writer.add_scalars('gradient penalty', {'gp':epoch_gp[-1]}, step+epoch*len(self.train_loader))
                    writer.add_scalars('lr', {'used_lr_d':self.getLr(0), 'used_lr_g':self.getLr(1)}, step+epoch*len(self.train_loader))
                    #print(self.weight.shape)
                    #for i in range(4):
                    #    writer.add_scalars('Loss weight', {'lw'+str(i):self.weight[i]}, step+epoch*len(self.train_loader))
                    #writer.add_image('Image', ((images[0].cpu().numpy()*np.array([[[.226]]])+np.array([[[.449]]]))*255).astype(np.uint8),step+epoch*len(self.train_loader))
                    #writer.add_image('Image', ((images[0].cpu().numpy()*np.array([[[.229]], [[.224]], [[.225]]])+np.array([[[.485]], [[.456]], [[.406]]]))*255).astype(np.uint8),step+epoch*len(self.train_loader))
                    grey_scale_map = irradiance_maps[0]#torch.exp(irradiance_maps[0])-self.args.log_eps if self.args.use_log_images else irradiance_maps[0]
                    writer.add_image('simulated map', np.trunc(grey_scale_map.cpu().numpy()).astype(np.uint8),step+epoch*len(self.train_loader))
                    grey_scale_map = torch.exp(real_event_images[0])-self.args.log_eps if self.args.use_log_images else real_event_images[0]
                    writer.add_image('real map', np.trunc(grey_scale_map.cpu().numpy()).astype(np.uint8),step+epoch*len(self.train_loader))
                    #grey_scale_map = torch.exp(irradiance_maps_next[0])-self.args.log_eps if self.args.use_log_images else real_event_images_next[0]
                    #writer.add_image('simulated next map', np.trunc(grey_scale_map.cpu().numpy()).astype(np.uint8),step+epoch*len(self.train_loader))
                    writer.add_image('simulated event map', (self.channel2to3(simulated_event_maps_cpu)).astype(np.uint8),step+epoch*len(self.train_loader))
                    writer.add_image('real event map', (self.channel2to3(real_event_maps_cpu)).astype(np.uint8),step+epoch*len(self.train_loader))
                    #threshold_normal = [real_threshold_C_pos.mean().item(), fake_threshold_C_pos.mean().item(), 
                    #                    real_threshold_C_pos.std().item(), fake_threshold_C_pos.std().item(),
                    #                    real_threshold_C_neg.mean().item(), fake_threshold_C_neg.mean().item(), 
                    #                    real_threshold_C_neg.std().item(), fake_threshold_C_neg.std().item()]
                    threshold_normal = [0.2, fake_threshold_C_pos.mean().item(), 
                                        0.00, fake_threshold_C_pos.std().item(),
                                        0.05, fake_threshold_C_neg.mean().item(), 
                                        0.00, fake_threshold_C_neg.std().item()]
                    writer.add_scalars('threshold pos mean', {'real pos mean':threshold_normal[0], 'fake pos mean':threshold_normal[1]}, step+epoch*len(self.train_loader))
                    writer.add_scalars('threshold pos std', {'real pos std':threshold_normal[2], 'fake pos std':threshold_normal[3]}, step+epoch*len(self.train_loader))
                    writer.add_scalars('threshold neg mean', {'real neg mean':threshold_normal[4], 'fake neg mean':threshold_normal[5]}, step+epoch*len(self.train_loader))
                    writer.add_scalars('threshold neg std', {'real neg std':threshold_normal[6], 'fake neg std':threshold_normal[7]}, step+epoch*len(self.train_loader))
                    #print(grey_scale_map)
                    #print('simulated_event_maps',(self.channel2to3(simulated_event_maps_cpu)).mean().mean())
                    #print('real_event_images',(self.channel2to3(real_event_maps_cpu)).mean().mean())
                    #print('regularization',math.log((self.args.log_eps+255)/self.args.log_eps)/self.args.minimum_threshold)
                    
                    writer.add_image('simulated event map patch', (self.channel2to3(simulated_event_maps_cpu[0:1,:,0:0+self.patch_size, 0:0+self.patch_size])).astype(np.uint8),step+epoch*len(self.train_loader))
                    writer.add_image('real event map patch', (self.channel2to3(real_event_maps_cpu[0:1,:,0:0+self.patch_size, 0:0+self.patch_size])).astype(np.uint8),step+epoch*len(self.train_loader))
                    
            
            #self.lr_step(epoch)

            average_epoch_train_loss_d_f = sum(epoch_loss_d_f) / len(epoch_loss_d_f)
            average_epoch_train_loss_d_r = sum(epoch_loss_d_r) / len(epoch_loss_d_r)
            average_epoch_train_loss_g = sum(epoch_loss_g) / len(epoch_loss_g)
            average_epoch_train_gp = sum(epoch_gp) / len(epoch_gp)

            #save model every X epoch
            if  (epoch+1) % self.args.epoch_save==0:
                torch.save(self.simulator.state_dict(), '{}_{}.pth'.format(self.savedir,str(epoch)))

            #save log
            with open(self.automated_log_path, "a") as myfile:
                myfile.write('\n%d\t\t%.5f\t\t%.5f\t\t%.5f\t\t%.3f\t\t%.8f\t\t%.8f' % (epoch, average_epoch_train_loss_d_f, average_epoch_train_loss_d_r, average_epoch_train_loss_g, average_epoch_train_gp, used_lr_d, used_lr_g ))
