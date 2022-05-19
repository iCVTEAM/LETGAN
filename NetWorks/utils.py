'''
initial net
'''

import math
import torch.nn.init as init

def weightsInit1(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif classname.find('Linear') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)

def weightsInit2(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        #init.xavier_normal_(m.weight)
        #init.kaiming_uniform_(m.weight)
        init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        #init.xavier_normal_(m.weight)
        #init.kaiming_uniform_(m.weight)
        init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        #init.constant_(m.weight, 1)
        #init.xavier_normal_(m.weight)
        #init.kaiming_uniform_(m.weight)
        init.normal_(m.weight, mean=1.0, std=0.01)
        if m.bias is not None:
            init.constant_(m.bias, 0)

#weights_init = weightsInit1#no use torch.nn.init
weights_init = weightsInit2#use torch.nn.init
