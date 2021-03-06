import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

from torch.utils import model_zoo
from torchvision import models

import imageio


class DiscriminatorNet(nn.Module):

    def __init__(self, patch_size):
        super().__init__()
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 7, stride=2, padding=3),
            nn.LayerNorm((64,56,56)),
            #nn.BatchNorm1d(6),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LayerNorm((128,28,28)),
            #nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.LayerNorm((128,28,28)),
            #nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LayerNorm((256,14,14)),
            #nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.LayerNorm((256,14,14)),
            #nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.LayerNorm((512,7,7)),
            #nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.LayerNorm((512,7,7)),
            #nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        #self.downsample = nn.AvgPool2d(4)

        self.sigmoid = nn.Sigmoid()
        #self.tanh = nn.Tanh()
        self.tanh = nn.Tanh()

        self.linear1 = nn.Sequential(
            nn.Linear(512*7*7, 256),
            nn.LeakyReLU(0.2, inplace=True))
        self.linear2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True))
        self.linear3 = nn.Sequential(
            nn.Linear(128, 1),
            nn.LeakyReLU(0.2, inplace=True))
        
        #'''
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 7, stride=4, padding=3),
            nn.LayerNorm((64,56,56)),
            #nn.BatchNorm1d(6),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LayerNorm((128,28,28)),
            #nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.LayerNorm((128,28,28)),
            #nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LayerNorm((256,14,14)),
            #nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.LayerNorm((256,14,14)),
            #nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.LayerNorm((512,7,7)),
            #nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.LayerNorm((512,7,7)),
            #nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        #self.downsample = nn.AvgPool2d(4)

        self.sigmoid = nn.Sigmoid()
        #self.tanh = nn.Tanh()
        self.tanh = nn.Tanh()

        self.linear1 = nn.Sequential(
            nn.Linear(512*7*7, 256),
            nn.LeakyReLU(0.2, inplace=True))
        self.linear2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True))
        self.linear3 = nn.Sequential(
            nn.Linear(128, 1),
            nn.LeakyReLU(0.2, inplace=True))
        
        #'''
        #'''
        #self.mean = torch.tensor([[[[0.4932]],[[0.1147]]]]).cuda()
        #self.std = torch.tensor([[[[1.3564]], [[0.4777]]]]).cuda()
        #self.sigmoid = nn.Sigmoid()
        #self.tanh = nn.Tanh()
        #'''
        self.linear1 = nn.Sequential(
            nn.Linear(4*40*40, 256),
            nn.LeakyReLU(0.2, inplace=True))
        self.linear2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True))
        self.linear3 = nn.Sequential(
            nn.Linear(128, 1),
            nn.LeakyReLU(0.2, inplace=True))
        #'''
        '''
        self.conv1 = nn.Sequential(
            #nn.Conv2d(4, 32, 5, stride=2, padding=2),
            nn.Conv2d(4, 32, 7, stride=4, padding=3),
            nn.LayerNorm((32,40,40)),
            #nn.BatchNorm1d(6),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LayerNorm((64,20,20)),
            #nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LayerNorm((64,20,20)),
            #nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LayerNorm((128,10,10)),
            #nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.LayerNorm((128,10,10)),
            #nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LayerNorm((256,5,5)),
            #nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 32, 3, stride=1, padding=1),
            nn.LayerNorm((32,5,5)),
            #nn.BatchNorm1d(1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        #self.downsample = nn.AvgPool2d(4)

        #self.sigmoid = nn.Sigmoid()
        #self.tanh = nn.Tanh()
        #self.tanh = nn.Tanh()

        self.linear1 = nn.Sequential(
            nn.Linear(32*5*5, 256),
            nn.LeakyReLU(0.2, inplace=True))
        self.linear2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True))
        self.linear3 = nn.Sequential(
            nn.Linear(128, 1),
            nn.LeakyReLU(0.2, inplace=True))
        
        #'''
    def forward(self, x):
        #print(x.size())
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #'''
        
        #x = self.downsample(x)
        #x = self.sigmoid(x)
        #x = x.view(x.size()[0], -1)
        
        x = x.view(x.size()[0], -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        
        #x = self.tanh(x)
        #x = self.sigmoid(x)
        return x


