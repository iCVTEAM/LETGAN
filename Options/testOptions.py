#-*- coding:utf-8 -*-
import argparse
import os

class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--cuda', action='store_true', default=True)
        self.parser.add_argument('--model', default="generator", help='model to train,options:fcn8,segnet...')  
        self.parser.add_argument('--model-dir', default="./Model/TrainingModels/20200813/generator_92.pth", help='path to stored-model')   
        self.parser.add_argument('--data-images', default="./Data/V2E/images5000.txt",help='path where image.txt lies')
        self.parser.add_argument('--data-real-images', default="-",help='path where image.txt lies')
        self.parser.add_argument('--data-event-maps', default="-",help='path where image.txt lies')
        self.parser.add_argument('--size', default=(224,224), help='resize the test image')
        self.parser.add_argument('--stored',default=True, help='whether or not store the result')
        self.parser.add_argument('--save-dir', type=str, default='./Results/',help='options. visualize the result of segmented picture, not just show IoU')
        self.parser.add_argument('--debug', action='store_true', default=False) # debug mode 
        
        self.parser.add_argument('--minimum-threshold', type=float, default=0.01)
        
        self.parser.add_argument("--max-angle-x", type=int, default=1, help="rotate angle")
        self.parser.add_argument("--max-angle-y", type=int, default=1, help="rotate angle")
        self.parser.add_argument("--max-angle-z", type=int, default=1, help="rotate angle")
        self.parser.add_argument("--max-translate-x", type=int, default=10, help="translate value")
        self.parser.add_argument("--max-translate-y", type=int, default=10, help="translate value")
        self.parser.add_argument("--max-translate-z", type=int, default=0, help="translate value")
        
        self.parser.add_argument("--use-log-images", default=True, help="irradiance maps = log(gray map)")
        self.parser.add_argument("--log-eps", type=int, default=0.1, help="E = log(L+eps)")
        self.parser.add_argument("--log-threshold", type=int, default=5, help="x>=T E = log(L) or x<T E = L/Tlog(T)")
        
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt
