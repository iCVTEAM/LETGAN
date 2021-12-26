# -*- coding:utf-8 -*-

# -----------------model net defination--------------------------

import torch
import torchvision
import numpy as np
from torch import nn
import torch.nn.functional as F

'''
deeplab_v3+ : pytorch's resnet is not same with tensorflow's resnet,so we modify some params
deeplab_v3+ : pytorch resnet 18/34 Basicblock
                      resnet 50/101/152 Bottleneck
'''
class ASPP(nn.Module):
    def __init__(self,in_channel=2048,depth=256):
        super().__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean=nn.AdaptiveAvgPool2d((1,1))
        self.conv=nn.Conv2d(in_channel,depth,1,1)
        # k=1 s=1 no pad
        self.atrous_block1=nn.Conv2d(in_channel,depth,1,1)
        self.atrous_block6=nn.Conv2d(in_channel,depth,3,1,padding=6,dilation=6)
        self.atrous_block12=nn.Conv2d(in_channel,depth,3,1,padding=12,dilation=12)
        self.atrous_block18=nn.Conv2d(in_channel,depth,3,1,padding=18,dilation=18)
        
        self.conv_1x1_output=nn.Conv2d(depth*5,depth,1,1)
        
    def forward(self,x):
        size=x.shape[2:]

        image_features=self.mean(x)
        image_features=self.conv(image_features)
        image_features=F.upsample(image_features,size=size,mode='bilinear',align_corners=True)
        
        atrous_block1=self.atrous_block1(x)
        
        atrous_block6=self.atrous_block6(x)
        
        atrous_block12=self.atrous_block12(x)
        
        atrous_block18=self.atrous_block18(x)
        
        net=self.conv_1x1_output(torch.cat([image_features,atrous_block1,atrous_block6,
                                            atrous_block12,atrous_block18],dim=1))
        return net
        
class DeeplabV3Plus(nn.Module):
    def __init__(self):
        class_number = 1
        super().__init__()
        encoder=torchvision.models.resnet50(pretrained=False)
        #self.start=nn.Sequential(encoder.conv1,encoder.bn1,
        #                         encoder.relu)
        self.start=nn.Sequential(nn.Conv2d(1,64,7,2,padding=3),encoder.bn1,
                                 encoder.relu)
        self.maxpool=encoder.maxpool
        self.encoder_feature=nn.Sequential(nn.Conv2d(64,48,1,1),nn.ReLU(inplace=True))
        encoder.layer1[0].conv2.stride=(2,2)###########modify#########
        encoder.layer1[0].downsample[0].stride=(2,2)#######modify########
        self.layer1=encoder.layer1
        self.layer2=encoder.layer2
        self.aspp=ASPP()
        self.conv_a=nn.Sequential(nn.Conv2d(256,256,1,1),nn.ReLU(inplace=True))
        self.conv_cat=nn.Sequential(nn.Conv2d(256+48,256,3,1,padding=1),nn.ReLU(inplace=True))
        self.conv_cat1=nn.Sequential(nn.Conv2d(256,256,3,1,padding=1),nn.ReLU(inplace=True))
        self.score=nn.Conv2d(256,class_number,1,1)# no relu
        
        self.conv_cat_1=nn.Sequential(nn.Conv2d(256+48,256,3,1,padding=1),nn.ReLU(inplace=True))
        self.conv_cat1_1=nn.Sequential(nn.Conv2d(256,256,3,1,padding=1),nn.ReLU(inplace=True))
        self.score_1=nn.Conv2d(256,class_number,1,1)# no relu
        self.layer3=encoder.layer3
        self.layer4=encoder.layer4
        
    def forward(self,x):
        size1=x.shape[2:]# need upsample input size
        x=self.start(x)
        xm=self.maxpool(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.aspp(x)
        x=self.conv_a(x)
        encoder_feature=self.encoder_feature(xm)
        size2=encoder_feature.shape[2:]
        decoder_feature=F.upsample(x,size=size2,mode='bilinear',align_corners=True)
        
        conv_cat=self.conv_cat(torch.cat([encoder_feature,decoder_feature],dim=1))
        conv_cat1=self.conv_cat1(conv_cat)
        score_conv=F.upsample(conv_cat1,size=size1,mode='bilinear',align_corners=True)
        score=self.score(score_conv)
        
        conv_cat_1=self.conv_cat_1(torch.cat([encoder_feature,decoder_feature],dim=1))
        conv_cat1_1=self.conv_cat1_1(conv_cat_1)
        score_conv_1=F.upsample(conv_cat1_1,size=size1,mode='bilinear',align_corners=True)
        score_1=self.score_1(score_conv_1)
        return score, score_1
        
        
def deeplabV3Plus():
    model=DeeplabV3Plus()
    return model


''' tensorflow resnet block
blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
]
'''
