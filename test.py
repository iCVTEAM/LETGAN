import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from Options.testOptions import TestOptions
from torch.autograd import Variable
from torch.utils.data import DataLoader
from DataLoader.transform import MyTransform
from DataLoader.dataset import MyDataSet
from NetWorks import getModel, loadParameter
#from Debug.functional import showMessage, outputMapWrite
from Criterion.criterion import Criterion

import imageio

def main(args):
    criter = Criterion('CategoricalCrossEntropyLoss')
    desPath = os.path.join(args.save_dir, args.model)
    if not os.path.exists(desPath):
        os.mkdir(desPath)
        
    images_list = []
    with open(args.data_images, 'r') as f:
        for line in f:
            images_list.append(line.strip())
                                         
    transform = MyTransform(args, reshape_size=(346,260),crop_size=args.size,debug_mode=args.debug,train_mode=False)
    datasetTest = MyDataSet(args, transform, debug_mode=args.debug,train_mode=False)
    loader = DataLoader(datasetTest, num_workers=0, batch_size=1,shuffle=False) #test data loader
    
    model = getModel(args)
    model = loadParameter(args, model)
    if args.cuda:
        model = model.cuda()
    #model.load_state_dict(torch.load(args.model_dir))
    model.eval()

    epochTime = []
    for step, (images_cpu, irradiance_maps_cpu, optic_flows_cpu) in enumerate(loader):
        if (args.debug):
            pass
        startTime = time.time()

        images, irradiance_maps, optic_flows = \
        images_cpu, irradiance_maps_cpu, optic_flows_cpu
        if args.cuda:
            images, irradiance_maps, optic_flows = \
            images_cpu.cuda(), irradiance_maps_cpu.cuda(), optic_flows_cpu.cuda()
        #imageio.imwrite('image_fake_big.bmp',irradiance_maps[0].squeeze().cpu().data.numpy())
        #images = Variable(images)
        model(images, irradiance_maps, optic_flows, images_list[step])
        #images, irradiance_maps, irradiance_maps_next, simulated_event_maps, fake_threshold_C_pos, fake_threshold_C_neg  = model(images, irradiance_maps, optic_flows)
        '''
        imageio.imwrite('image_next.bmp',irradiance_maps_next[0].squeeze().cpu().data.numpy())
        
        print('fake max', simulated_event_maps[0].max(),simulated_event_maps[0].mean())
        print('fake pos mean std, neg mean std', fake_threshold_C_pos.mean(), fake_threshold_C_pos.std(),fake_threshold_C_neg.mean(), fake_threshold_C_neg.std())
        
        imageio.imwrite('image.bmp', images[0].squeeze().cpu().data.numpy())#.transpose(1,2,0)
        
        zero = torch.zeros_like(real_event_images)
        zero = zero.cuda()
        output3 = torch.cat((real_event_maps[:,0:1],zero),dim=1)
        output3 = torch.cat((output3,real_event_maps[:,1:2]),dim=1)
        imageio.imwrite('event_real.bmp', np.trunc((output3[0]).squeeze().cpu().data.numpy().transpose(1,2,0)))
        
        output3 = torch.cat((simulated_event_maps[:,0:1],zero),dim=1)
        output3 = torch.cat((output3,simulated_event_maps[:,1:2]),dim=1)
        imageio.imwrite('event_fake.bmp', np.trunc((output3[0]).squeeze().cpu().data.numpy().transpose(1,2,0)))
        
        imageio.imwrite('image_real.bmp', real_event_images[0].squeeze().cpu().data.numpy())
        
        imageio.imwrite('image_fake.bmp', irradiance_maps[0].squeeze().cpu().data.numpy())

        
        '''
        if (args.debug):
            return
        epochTime.append(time.time() - startTime)
        
        #outputs = criter.getActivateFunction()(outputs, dim=1)
        #outputs = outputs.max(1)[1].squeeze().cpu().data.numpy() #index of max-channel 
        
        #print(type(outputs))

        #saveToFile(outputs, step)
        
        print('TEST %s: [%d/%d] fps-1: %.4fs'%(
            images_list[step], (step+1), len(loader), epochTime[step]))
        
    averageEpochTime = sum(epochTime) / len(epochTime)
    print('TEST num: %d fps-1: %.4fs'%(len(loader), averageEpochTime))
    

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser = TestOptions().parse()
    main(parser)


