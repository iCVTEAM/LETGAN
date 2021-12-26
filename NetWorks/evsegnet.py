import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

from torch.utils import model_zoo
from torchvision import models
from .resample2d_package.resample2d import Resample2d

import imageio

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


class EvSegNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        
        self.minimumThreshold = 0.01

        self.debugMode = args.debug
        self.isCudaAvailable = args.cuda
        self.resizeShape = args.size

        decoders = list(models.vgg16(pretrained=False).features.children())

        self.dec1 = nn.Sequential(*decoders[:5])
        self.dec2 = nn.Sequential(*decoders[5:10])
        self.dec3 = nn.Sequential(*decoders[10:17])
        self.dec4 = nn.Sequential(*decoders[17:24])
        self.dec5 = nn.Sequential(*decoders[24:])

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        m.requires_grad = False

        self.enc5 = SegNetEnc(512, 512, 1)
        self.enc4 = SegNetEnc(1024, 256, 1)
        self.enc3 = SegNetEnc(512, 128, 1)
        self.enc2 = SegNetEnc(256, 64, 0)
        self.enc1 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, 2, 3, padding=1)
        
        self.warp = Resample2d()
        #self.channelnorm = ChannelNorm()

    def forward(self, images, irradianceMaps, opticFlows):    
        '''
            Attention, input size should be the 32x. 
        '''
        #x = images
        x = F.interpolate(images, self.resizeShape, mode='nearest')
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        dec5 = self.dec5(dec4)
        enc5 = self.enc5(dec5) 

        enc4 = self.enc4(torch.cat([dec4, enc5], 1))
        enc3 = self.enc3(torch.cat([dec3, enc4], 1))
        enc2 = self.enc2(torch.cat([dec2, enc3], 1))
        enc1 = self.enc1(torch.cat([dec1, enc2], 1))
        
        #normal = F.interpolate(self.final(enc1), x.size()[2:])
        normal = enc1

        #mean = normal[:,0,:]
        #std = normal[:,1,:]
        
        normalSample = torch.randn(normal.size())
        if self.isCudaAvailable:
            normalSample = normalSample.cuda()
        thresholdCPos = normalSample[:,0,:] * normal[:,1,:] + normal[:,0,:]
        thresholdCNeg = normalSample[:,1,:] * normal[:,1,:] + normal[:,0,:]
        #thresholdCPos = torch.zeros_like(thresholdCPos)+0.5
        #thresholdCNeg = torch.zeros_like(thresholdCNeg)+0.5
        print(thresholdCPos[0,23,24])
        print(thresholdCNeg[0,23,24])
        thresholdCPos = F.relu(thresholdCPos - self.minimumThreshold) + self.minimumThreshold
        thresholdCNeg = F.relu(thresholdCNeg - self.minimumThreshold) + self.minimumThreshold
        print(thresholdCPos[0,23,24])
        print(thresholdCNeg[0,23,24])
        #print(irradianceMaps.size())
        #print(opticFlows.size())
        resample = self.warp(irradianceMaps, opticFlows)
        print(irradianceMaps[0,0,23,24])
        print(resample[0,0,23,24])
        #print(resample.size())
        diffImg = irradianceMaps - resample 
        print(diffImg[0,0,23,24])
        #normDiffImg = self.channelnorm(diffImg)
        diffImg = F.interpolate(diffImg, x.size()[2:])
        print(diffImg[0,0,23,24])
        resamplePos = F.relu(diffImg) / thresholdCPos
        resampleNeg = F.relu(-diffImg) / thresholdCNeg
        print(resamplePos[0,0,23,24])
        print(resampleNeg[0,0,23,24])
        resampleAll = resamplePos - resampleNeg
        print(resampleAll[0,0,23,24])
        
        #randSample = torch.rand_like(resampleAll)
        #if self.isCudaAvailable:
        #    randSample = randSample.cuda()
        #resampleAll = resampleAll - torch.log(-torch.log(randSample))
        #resampleAll = F.softmax(resampleAll, dim=1)

        if self.debugMode:
            imageio.imwrite('image.bmp', images.squeeze().cpu().data.numpy().transpose(1,2,0))
            imageio.imwrite('irradianceMaps.bmp', irradianceMaps.squeeze().cpu().data.numpy())
            imageio.imwrite('thresholdCPos.bmp', thresholdCPos.squeeze().cpu().data.numpy())
            imageio.imwrite('thresholdCNeg.bmp', thresholdCNeg.squeeze().cpu().data.numpy())
            imageio.imwrite('resample.bmp',resample.squeeze().cpu().data.numpy())
            imageio.imwrite('diffImg.bmp', diffImg.squeeze().cpu().data.numpy())
            imageio.imwrite('resamplePos.bmp', resamplePos.squeeze().cpu().data.numpy())
            imageio.imwrite('resampleNeg.bmp', resampleNeg.squeeze().cpu().data.numpy())
            imageio.imwrite('resampleAll.bmp', resampleAll.squeeze().cpu().data.numpy())
            resampleAll3 = torch.cat((resamplePos,resampleNeg),dim=1)
            resampleAll3 = torch.cat((resampleAll3,resampleNeg),dim=1)
            imageio.imwrite('resampleAll3.bmp',
                            np.trunc(resampleAll3.squeeze().cpu().data.numpy().transpose(1,2,0)))

        return resampleAll
