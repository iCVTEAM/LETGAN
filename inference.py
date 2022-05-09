import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from Options.inferenceOptions import InferenceOptions
from NetWorks import getModel, loadParameter

from utils.event_packagers import *
import random
import cv2

import math

class ESIM():
    def __init__(self, args, model):
        self.args = args
        self.threshold = {'contrast_threshold_pos': 0, 'contrast_threshold_neg': 0, 'contrast_threshold_sigma_pos': 0, 'contrast_threshold_sigma_neg': 0}
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.minimum_threshold = args.minimum_threshold
        self.update_th = True
        self.random_th = 0.1

    def ln_map(self, map):
        new_map = map.clone()
        new_map[map < self.args.log_threshold] = map[map < self.args.log_threshold]/self.args.log_threshold*math.log(self.args.log_threshold)
        new_map[map >= self.args.log_threshold] = torch.log(map[map >= self.args.log_threshold])
        #new_map = torch.log(self.args.log_eps + map/255.0)
        return new_map
    
    def update(self):
        self.update_th = True
        
    def get_th(self, images, irradiance_maps):
        real_normal_sample = torch.randn(images.size()).to(self.device)
        if self.update_th:
            self.update_th = False
            with torch.no_grad():
                self.threshold = self.model.get_th(images)
        threshold_C_pos_origin = real_normal_sample * self.threshold['contrast_threshold_sigma_pos'] + self.threshold['contrast_threshold_pos']
        threshold_C_neg_origin = real_normal_sample * self.threshold['contrast_threshold_sigma_neg'] + self.threshold['contrast_threshold_neg']

        minimum_threshold = irradiance_maps*0.1
        maximum_threshold = irradiance_maps*0.5
        tmp = (threshold_C_pos_origin>maximum_threshold)# & random_sample
        threshold_C_pos_origin[tmp] = maximum_threshold[tmp]
        tmp = (threshold_C_pos_origin<minimum_threshold)# & random_sample
        threshold_C_pos_origin[tmp] = minimum_threshold[tmp]

        minimum_threshold = self.ref_values*0.1
        maximum_threshold = self.ref_values*0.5
        tmp = (threshold_C_neg_origin>maximum_threshold)# & random_sample
        threshold_C_neg_origin[tmp] = maximum_threshold[tmp]
        tmp = (threshold_C_neg_origin<minimum_threshold)# & random_sample
        threshold_C_neg_origin[tmp] = minimum_threshold[tmp]

        self.threshold_C_pos = F.relu(threshold_C_pos_origin - self.minimum_threshold) + self.minimum_threshold
        self.threshold_C_neg = F.relu(threshold_C_neg_origin - self.minimum_threshold) + self.minimum_threshold

    def inferenceBefore(self, images):
        self.irradiance_maps_0 = self.ln_map(images)
        
        self.last_img = self.irradiance_maps_0.clone()
        self.ref_values = self.irradiance_maps_0.clone()
        self.current_time = 0
        
        self.delta_min = 1e-8
        self.refractory_period_ns = 0
        
    def inference(self, images, now_time):
        delta_time = now_time - self.current_time
        irradiance_maps = self.ln_map(images)
        self.get_th(images, irradiance_maps)
        
        irradiance_values_pos = F.relu(irradiance_maps-self.ref_values) // self.threshold_C_pos
        irradiance_values_neg = F.relu(-(irradiance_maps-self.ref_values)) // self.threshold_C_neg
        
        irradiance_values_pos[self.ref_values+self.threshold_C_pos < self.last_img] = 0
        irradiance_values_neg[self.ref_values-self.threshold_C_neg > self.last_img] = 0
        
        #st = time.time()
        iter_pos_max = int(irradiance_values_pos.max().item())
        iter_neg_max = int(irradiance_values_neg.max().item())
        iter_max = iter_pos_max if iter_pos_max > iter_neg_max else iter_neg_max

        shotNoiseFactor = 0
        shotNoiseNumberPerIter = 0 if iter_max == 0 else int(shotNoiseFactor*delta_time*self.args.reshape_size[0]*self.args.reshape_size[1]//iter_max)
        shotNoiseNumber = shotNoiseNumberPerIter*iter_max*2
        shotNoiseZero = torch.zeros((shotNoiseNumberPerIter), dtype=torch.long).to(self.device)
        
        length = int(irradiance_values_pos.sum().item())+int(irradiance_values_neg.sum().item())+shotNoiseNumber
        time_stamps = torch.zeros((length), dtype=torch.float).to(self.device)
        xs  = torch.zeros((length), dtype=torch.long).to(self.device)
        ys  = torch.zeros((length), dtype=torch.long).to(self.device)
        values  = torch.zeros((length), dtype=torch.int).to(self.device)
        point = 0
        num_pos = 0
        num_neg = 0
        
        t_pos = self.current_time+(self.ref_values-self.last_img)/(irradiance_maps-self.last_img + self.delta_min)*delta_time
        t_pos_iter = self.threshold_C_pos/(irradiance_maps-self.last_img + self.delta_min)*delta_time
        t_pos_ori = t_pos.clone().detach()
        t_pos_iter_ori = t_pos_iter.clone().detach()
        
        t_neg = self.current_time+(self.ref_values-self.last_img)/(irradiance_maps-self.last_img + self.delta_min)*delta_time
        t_neg_iter = self.threshold_C_neg/(irradiance_maps-self.last_img + self.delta_min)*delta_time
        t_neg_ori = t_neg.clone().detach()
        t_neg_iter_ori = t_neg_iter.clone().detach()
        
        irradiance_values_pos_ori = irradiance_values_pos.clone().detach()
        pos_event_xy = torch.where(irradiance_values_pos>self.delta_min)
        (pos_event_xy_0,pos_event_xy_1,pos_event_xy_2,pos_event_xy_3) = pos_event_xy
        pos_event_value = irradiance_values_pos[pos_event_xy_0,pos_event_xy_1,pos_event_xy_2,pos_event_xy_3]
        num_pos_events = len(pos_event_value)
        t_pos = t_pos[pos_event_xy_0,pos_event_xy_1,pos_event_xy_2,pos_event_xy_3]
        t_pos_iter = t_pos_iter[pos_event_xy_0,pos_event_xy_1,pos_event_xy_2,pos_event_xy_3]
        
        irradiance_values_neg_ori = irradiance_values_neg.clone().detach()
        neg_event_xy = torch.where(irradiance_values_neg>self.delta_min)
        (neg_event_xy_0,neg_event_xy_1,neg_event_xy_2,neg_event_xy_3) = neg_event_xy
        neg_event_value = irradiance_values_neg[neg_event_xy_0,neg_event_xy_1,neg_event_xy_2,neg_event_xy_3]
        num_neg_events = len(neg_event_value)
        t_neg = t_neg[neg_event_xy_0,neg_event_xy_1,neg_event_xy_2,neg_event_xy_3]
        t_neg_iter = t_neg_iter[neg_event_xy_0,neg_event_xy_1,neg_event_xy_2,neg_event_xy_3]
        
        #print('itpr:', iter_max, time.time()-st)
        for iter in range(iter_max):
            #st11 = time.time()
            
            if num_pos_events > 0:
                #pos_event_xy = torch.where(irradiance_values_pos>iter+self.delta_min)
                pos_event_true = pos_event_value>iter+self.delta_min
                pos_event_value = pos_event_value[pos_event_true]
                tmp = len(pos_event_value)
                if tmp < num_pos_events:
                    num_pos_events = tmp
                    t_pos = t_pos[pos_event_true]
                    t_pos_iter = t_pos_iter[pos_event_true]
                #    pos_event_xy_0 = pos_event_xy_0[pos_event_true]
                #    pos_event_xy_1 = pos_event_xy_1[pos_event_true]
                    pos_event_xy_2 = pos_event_xy_2[pos_event_true]
                    pos_event_xy_3 = pos_event_xy_3[pos_event_true]
                #t = self.current_time+(self.ref_values+self.threshold_C_pos*(iter+1)-self.last_img)/(irradiance_maps-self.last_img + self.delta_min)*delta_time
                t_pos += t_pos_iter
                
            if num_pos_events > 0:
                time_stamps[point:point+num_pos_events] = t_pos
                xs[point:point+num_pos_events] = pos_event_xy_3
                ys[point:point+num_pos_events] = pos_event_xy_2
                #values[point:point+num_pos_events] = torch.ones((num_pos_events), dtype=torch.int).to(self.device)
                values[point:point+num_pos_events] += 1
                num_pos += num_pos_events
                point += num_pos_events

            if num_neg_events > 0:
                #neg_event_xy = torch.where(irradiance_values_neg>iter+self.delta_min)
                neg_event_true = neg_event_value>iter+self.delta_min
                neg_event_value = neg_event_value[neg_event_true]
                tmp = len(neg_event_value)
                if tmp < num_neg_events:
                    num_neg_events = tmp
                    t_neg = t_neg[neg_event_true]
                    t_neg_iter = t_neg_iter[neg_event_true]
                #    neg_event_xy_0 = neg_event_xy_0[neg_event_true]
                #    neg_event_xy_1 = neg_event_xy_1[neg_event_true]
                    neg_event_xy_2 = neg_event_xy_2[neg_event_true]
                    neg_event_xy_3 = neg_event_xy_3[neg_event_true]

                t_neg -= t_neg_iter
            
            if num_neg_events > 0:
                #time_stamps[point:point+num_neg_events] = t_neg[neg_event_xy_0,neg_event_xy_1,neg_event_xy_2,neg_event_xy_3]
                time_stamps[point:point+num_neg_events] = t_neg
                xs[point:point+num_neg_events] = neg_event_xy_3
                ys[point:point+num_neg_events] = neg_event_xy_2
                #no need
                #values[point:point+num_neg_events] = torch.zeros((num_neg_events), dtype=torch.int).to(self.device)
                num_neg += num_neg_events
                point += num_neg_events

        time_stamps = time_stamps.cpu().data.numpy().tolist()
        xs = xs.cpu().data.numpy().tolist()
        ys = ys.cpu().data.numpy().tolist()
        values = values.cpu().data.numpy().tolist()

        self.ref_values += (irradiance_values_pos_ori * self.threshold_C_pos - irradiance_values_neg_ori * self.threshold_C_neg)
        self.current_time = now_time
        self.last_img = irradiance_maps.clone()
        
        #assert sum(np.array(time_stamps)<0)==0
        index = sorted(range(len(time_stamps)), key=lambda k: time_stamps[k])
        return {'time_stamps':np.array(time_stamps)[index].tolist(),
                'xs':np.array(xs)[index].tolist(),
                'ys':np.array(ys)[index].tolist(),
                'values':np.array(values)[index].tolist(),
                'num_pos':num_pos,'num_neg':num_neg}

def channel2to3(output2):
    output3 = np.zeros((3,output2.shape[2], output2.shape[3]))
    output3[0,:] = output2[0,0,:]
    output3[2,:] = output2[0,1,:]
    
    #regularization = math.log((self.args.log_eps+255)/self.args.log_eps)/self.args.minimum_threshold
    output3[output3>0.1] = output3[output3>0.1]+125
    output3[output3>255]=255
    return np.trunc(output3)

def readconfig(file):
    imglist = []
    timelist = []

    with open(file, 'r') as f:
        for line in f:
            tmp = line.strip().split(' ')
            imglist.append(tmp[0])
            timelist.append(float(tmp[1]))
        f.close()
    return imglist, timelist



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    desPath = args.save_dir
    if not os.path.exists(desPath):
        os.mkdir(desPath)

    images_list, time_list = readconfig(args.data_images)
    
    model = getModel(args)
    model = loadParameter(args, model).to(device)
    model = model.to(device)
    model.eval()

    esim = ESIM(args, model)

    ep = hdf5_packager(os.path.join(desPath, 'output.h5'))
    ep.set_data_available(1, 1)
    num_pos, num_neg, last_ts, img_cnt, flow_cnt = 0, 0, 0, 0, 0
    time_stamps = []
    xs = []
    ys = []
    values = []
    tnow = 0
    dt = 1.0/60
    i = -1
    while(True):
        i += 1

        if i >= len(images_list):
            break

        image = torch.from_numpy(cv2.imread(images_list[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        time_stamp = time_list[i]

        if time_stamp >= tnow+dt:
            tnow = time_stamp
            ep.package_image(image[0].squeeze().cpu().data.numpy(), time_stamp, img_cnt)
            img_cnt += 1
            esim.update()

        if i == 0:
            ep.package_image(image[0].squeeze().cpu().data.numpy(), time_stamp, img_cnt)
            img_cnt += 1
            esim.inferenceBefore(image)
            esim.update()
            continue

        event = esim.inference(image, time_stamp)

        time_stamps.extend(event['time_stamps'])
        xs.extend(event['xs'])
        ys.extend(event['ys'])
        values.extend(event['values'])
        num_pos += event['num_pos']
        num_neg += event['num_neg']

    ep.package_events(xs,ys,time_stamps,values)
    t0 = time_stamps[0]
    last_ts = time_stamps[-1]
    sensor_size = [args.reshape_size[1], args.reshape_size[0]]
    ep.add_metadata(num_pos, num_neg, last_ts-t0, t0, last_ts, img_cnt, flow_cnt, sensor_size)
    ep.set_data_true(img_cnt, flow_cnt)
    print(num_pos, num_neg, last_ts-t0, t0, last_ts, img_cnt, flow_cnt, sensor_size)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = InferenceOptions().parse()
    main(parser)


