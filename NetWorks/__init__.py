from .generatornet import EvSegNet
from .discriminatornet import DiscriminatorNet
from .utils import *
import torch
from Debug.functional import showMessage, outputMapWrite
import os

NETsDIRECTORY = {'discriminator' : DiscriminatorNet, 'generator' : EvSegNet}

def loadParameter(args, model):
    new_params = model.state_dict()
    pretrain_dict = torch.load(args.model_dir)
    # remove module.
    pretrain_dict = {k[7:]:v for k, v in pretrain_dict.items() if k[7:] in new_params and v.size() == new_params[k[7:]].size()}# default k in m m.keys
    #pretrain_dict = {k:v for k, v in pretrain_dict.items() if k in new_params and v.size() == new_params[k].size()}# default k in m m.keys
    if args.debug:
        '''
        print("INFO: debug")
        for k in new_params:
            print(k)
        print('-------')
        print(pretrain_dict)
        for k, v in pretrain_dict.items():
            print(k)
        '''
        pass
    for k in pretrain_dict:
        showMessage(fileName='__init__.py', functionName='loadPertrainedParameter', lineNumber=18, variableName='pretrain_dict',
                    variableValue=k)
    new_params.update(pretrain_dict)
    model.load_state_dict(new_params)
    #model = torch.load('xx.pth')
    #model.load_state_dict(torch.load('xx.pth'), strict=True)
    return model

def loadPertrainedParameter(args, model):
    new_params = model.state_dict()
    '''
    with open('net.txt','w') as f:
        for k, v in new_params.items():
            f.write(str(k)+' '+str(v.size())+'\n')
        f.close()
    '''
    pretrain_dict = torch.load(args.model_dir)
    '''
    map_dict = {'features.0.weight':'dec1.0.weight',
            'features.0.bias':'dec1.0.bias',
            'features.2.weight':'dec1.2.weight',
            'features.2.bias':'dec1.2.bias',
            'features.5.weight':'dec2.0.weight',
            'features.5.bias':'dec2.0.bias',
            'features.7.weight':'dec2.2.weight',
            'features.7.bias':'dec2.2.bias',
            'features.10.weight':'dec3.0.weight',
            'features.10.bias':'dec3.0.bias',
            'features.12.weight':'dec3.2.weight',
            'features.12.bias':'dec3.2.bias',
            'features.14.weight':'dec3.4.weight',
            'features.14.bias':'dec3.4.bias',
            'features.17.weight':'dec4.0.weight',
            'features.17.bias':'dec4.0.bias',
            'features.19.weight':'dec4.2.weight',
            'features.19.bias':'dec4.2.bias',
            'features.21.weight':'dec4.4.weight',
            'features.21.bias':'dec4.4.bias',
            'features.24.weight':'dec5.0.weight',
            'features.24.bias':'dec5.0.bias',
            'features.26.weight':'dec5.2.weight',
            'features.26.bias':'dec5.2.bias',
            'features.28.weight':'dec5.4.weight',
            'features.28.bias':'dec5.4.bias'}
    pretrain_dict = {map_dict[k]:v for k, v in pretrain_dict.items() if k in map_dict and map_dict[k] in new_params and v.size() == new_params[map_dict[k]].size()}# default k in m m.keys
    '''
    
    #'''
    map_dict = {'layer1':'encoder3',
            'layer2':'encoder4',
            'layer3':'encoder5',
            'layer4':'encoder6'}
    pretrain_dict_new = {}
    for k, v in pretrain_dict.items():
        for key in map_dict:
            if key in k:
                new_k = map_dict[key]+k[6:]
                if new_k in new_params and v.size() == new_params[new_k].size():
                    pretrain_dict_new[new_k] = v
    pretrain_dict = pretrain_dict_new
    #'''
    #pretrain_dict = {'d3p.'+k:v for k, v in pretrain_dict.items() if 'd3p.'+k in new_params and v.size() == new_params['d3p.'+k].size()}# default k in m m.keys
    #'''
    
    #for k, v in new_params.items():
    #    print(k, v.size())
    #pretrain_dict = {k:v for k, v in pretrain_dict.items() if k in new_params and v.size() == new_params[k].size()}# default k in m m.keys
    if args.debug:
        print("INFO debug")
        #for k in new_params:
        #    print(k)
        #print('-------')
        #print(pretrain_dict)
        #for k, v in pretrain_dict.items():
        #    print(k)
    for k in pretrain_dict:
        showMessage(fileName='__init__.py', functionName='loadPertrainedParameter', lineNumber=18, variableName='pretrain_dict',
                    variableValue=k)
    new_params.update(pretrain_dict)
    model.load_state_dict(new_params)
    #model = torch.load('xx.pth')
    #model.load_state_dict(torch.load('xx.pth'), strict=True)
    return model

def existArgs(args):
    try:
        args.model_dir
    except:
        return False
    else:
        return True

def getModel(args):
    Net = NETsDIRECTORY[args.model]
    model = Net(args)
    model.apply(weights_init)
    #if existArgs(args):
    #    model = loadPertrainedParameter(args, model)
    return model
