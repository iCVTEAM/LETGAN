'''
read and handle data set
'''

import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from Debug.functional import showMessage
import random
import math
import imageio

class DatasetFunction(Dataset):
    def __init__(self):
        #super().__init__()
        self.extensions = ['.jpg', '.png', '.JPG', '.PNG']
    
    def loadImage(self, fileName):
        return Image.open(fileName)
        
    def loadNpy(self, fileName):
        return np.load(fileName)
    
    def getImagePath(self, root, baseName, extension=None):
        if(extension is not None):
            return os.path.join(root, '{}{}'.format(baseName, extension))
        else:
            return os.path.join(root, '{}'.format(baseName))
    
    def getImageBaseName(self, fileName):
        return os.path.basename(os.path.splitext(fileName)[0])

class MyDataSet(DatasetFunction):
    def __init__(self, args, transform, debug_mode=False, train_mode=False):
        self.args = args
        
        self.reshape_size = args.size
        
        self.transform = transform
        self.debug_mode = debug_mode
        self.train_mode = train_mode
        '''old
        self.images = args.data_images
                      
        assert os.path.exists(self.images), "{} not exists !".format(self.images)
                       
        if self.train_mode:
            self.real_images = args.data_real_images
            self.real_event_maps = args.data_event_maps
            
            assert os.path.exists(self.real_images), "{} not exists !".format(self.real_images)
            assert os.path.exists(self.real_event_maps), "{} not exists !".format(self.real_event_maps)
        
        
        self.images_list = []
                
        with open(self.images, 'r') as f:
            for line in f:
                self.images_list.append(line.strip())
                
        #for test faster
        #if self.debug_mode:
        #    self.images_list = self.images_list[0:64]

        if self.train_mode:
            self.real_event_images_list = []
            self.real_event_maps_list = []
            
            with open(self.real_images, 'r') as f:
                for line in f:
                    self.real_event_images_list.append(line.strip())
            with open(self.real_event_maps, 'r') as f:
                for line in f:
                    self.real_event_maps_list.append(line.strip())
            
            assert len(self.real_event_images_list) == len(self.real_event_maps_list), "{}".format(len(self.real_event_images_list), len(self.real_event_maps_list))
        '''
        self.simu_data_path = args.data_simu
        assert os.path.exists(self.simu_data_path), "{} not exists !".format(self.simu_data_path)
                       
        if self.train_mode:
            self.real_data_path = args.data_real
            assert os.path.exists(self.real_data_path), "{} not exists !".format(self.real_data_path)
        
        simu_list = [os.path.join(self.simu_data_path, scenefile) for scenefile in os.listdir(self.simu_data_path)]
        simu_index = [random.randint(0,len(simu_list)-1) for i in range(self.args.data_size)]
        self.simu_list = [simu_list[i] for i in simu_index]
        
        #self.simu_list = [os.path.join(self.simu_data_path, scenefile) for scenefile in os.listdir(self.simu_data_path)]
        #random.shuffle(self.simu_list)
        
        if self.train_mode:
            #real_list = [os.path.join(self.real_data_path, scenefile) for scenefile in os.listdir(self.real_data_path)]
            #real_index = [random.randint(0,len(real_list)-1) for i in range(self.args.data_size)]
            #random.shuffle(real_list)
            #self.real_list = [real_list[i] for i in real_index]
            
            self.real_list = [os.path.join(self.real_data_path, scenefile) for scenefile in os.listdir(self.real_data_path)]
            random.shuffle(self.real_list)
            
    def __getitem__(self, index):
        imageName = self.simu_list[index]
        irradiance_map = self.loadNpy(imageName)[0]

        #image = self.loadImage(imageName).convert('RGB')
        #image = image.resize(self.reshape_size,Image.BILINEAR)
        
        #irradiance_map = self.loadImage(imageName).convert('L')#gray
        
        #print('aaaaa',imageName,irradiance_map.getpixel((197,220)))
        #foreground = self.loadImage(imageName.replace('JPEGImages', 'Annotations').replace('jpg', 'png')).convert('L')#gray
        
        if self.train_mode:
            #optic_flow = self.loadNpy(imageName.replace('Img', 'Flo').replace('/00000.png', '.npy'))
            #real_event_image_name = self.real_event_images_list[index]
            #real_event_map_name = self.real_event_maps_list[index]
            
            #real_event_image = self.loadImage(real_event_image_name).convert('L')
            #real_event_map = self.loadNpy(real_event_map_name)
            #print(index, imageName, real_event_image_name,real_event_map_name)
            
            real_event_name = self.real_list[index]
            real_event = self.loadNpy(real_event_name)[0]
            real_event_image = real_event[0:1]
            real_event_map = real_event[1:3]
            real_event_image2 = real_event[3:4]
            #image, irradiance_map, optic_flow, real_event_image, real_event_map = self.transform(irradiance_map,optic_flow,real_event_image, real_event_map)
            #return image, irradiance_map, optic_flow, real_event_image, real_event_map
            irradiance_map, real_event_image, real_event_map,real_event_image2 = self.transform(irradiance_map,real_event_image=real_event_image, real_event_map=real_event_map,real_event_image2=real_event_image2)
            return irradiance_map, real_event_image, real_event_map,real_event_image2
        else:
            image, irradiance_map = self.transform(irradiance_map)
            return image, irradiance_map

    def __len__(self):
        return self.args.data_size
