import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastNet(nn.Module):
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(ContrastNet, self).__init__()

    def forward(self, x):
        pass
        
    def inference(self, x):
        x_ud = x.clone().detach()
        x_lr = x.clone().detach()
        N,C,H,W = x.size()
        x_ud[:,:,1:H,:] = x_ud[:,:,1:H,:]-x_ud[:,:,0:H-1,:]
        x_ud[:,:,0:1,:] = x_ud[:,:,0:1,:]-x_ud[:,:,0:1,:]
        #print(x_ud)
        
        x_lr[:,:,:,1:W] = x_lr[:,:,:,1:W]-x_lr[:,:,:,0:W-1]
        x_lr[:,:,:,0:1] = x_lr[:,:,:,0:1]-x_lr[:,:,:,0:1]
        
        x_ud_2 = x_ud.mul(x_ud)
        x_lr_2 = x_lr.mul(x_lr)
        cg = x_ud_2 + x_lr_2
        cg[:,:,0:H-1,:] += x_ud_2[:,:,1:H,:]
        cg[:,:,:,0:W-1] += x_lr_2[:,:,:,1:W]
        #cg = (x_ud.mul(x_ud) + x_lr.mul(x_lr))#/(2*H*W-H-W)
        #print(cg)
        #p = (4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+2*4)
        return cg