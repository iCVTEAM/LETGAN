import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils import model_zoo
from torchvision import models

class SequeezeNet(nn.Module):
    def __init__(self, class_number=5):
        super(SequeezeNet, self).__init__()
        encoders = list(models.squeezenet1_1(pretrained=False).features.children())
        self.input = nn.Sequential(*encoders[:3])
        self.enc1 = nn.Sequential(*encoders[3:6])
        self.enc2 = nn.Sequential(*encoders[6:])
        self.dec1 = nn.Sequential(nn.Conv2d(512,256,7,1,padding=12,dilation=4),nn.ReLU(inplace=True))
        self.fuse1 = nn.Conv2d(128+512,512,1,1)# no relu
        self.dec2 = nn.Sequential(nn.Conv2d(256,256,3,1,padding=1),nn.ReLU(inplace=True))
        self.score=nn.Conv2d(256,class_number,1,1)# no relu
        

    def forward(self, x):
        size1 = x.shape[2:]
        x = self.input(x)
        xm = self.enc1(x)
        x = self.enc2(xm)
        size2=xm.shape[2:]

        x=F.upsample(x,size=size2,mode='bilinear',align_corners=True)
        #x=torch.cat([xm,x],dim=1)
        #x = self.fuse1(x)
        x=self.dec1(x)
        x=F.upsample(x,size=size1,mode='bilinear',align_corners=True)#偶然发现，upsample可以放在self.score后面，减少训练时显存使用
        ##建议以后凡是最后一层前面是upsample，最后一层为转类别概率层，将upsample层和最后一层互换位置，减少使用显存##
        x = self.dec2(x)
        x=self.score(x)

        return x