#-*- coding:utf-8 -*-
import argparse
import os

class InferenceOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--model', default="generator", help='model to train,options:fcn8,segnet...')  
        self.parser.add_argument('--model-dir', default="./Model/TrainingModels/16/generator_94.pth", help='path to stored-model')   #204c fenzhi
        #self.parser.add_argument('--model-dir', default="./Model/generatorIJRR.pth", help='path to stored-model')   #204c fenzhi
        #self.parser.add_argument('--model-dir', default="./Model/generatorMVSEC.pth", help='path to stored-model')   
        self.parser.add_argument('--data-images', default="./Data/images.txt",help='path where image.txt lies')
        #self.parser.add_argument('--reshape-size', default=(346,260), help='resize the image')
        #self.parser.add_argument('--reshape-size', default=(256,256), help='resize the image')
        self.parser.add_argument('--reshape-size', default=(240,180), help='resize the image')
        #self.parser.add_argument('--reshape-size', default=(346,260), help='resize the image')
        self.parser.add_argument('--size', default=(256,256), help='crop the image')
        self.parser.add_argument('--save-dir', type=str, default='./Results/',help='options. visualize the result of segmented picture, not just show IoU')
        self.parser.add_argument('--debug', action='store_true', default=False) # debug mode 
        self.parser.add_argument('--dl', type=int, default=0, help='crop the image')
        self.parser.add_argument('--dr', type=int, default=281, help='crop the image')
        self.parser.add_argument('--minimum-threshold', type=float, default=0.01)
        '''
        self.parser.add_argument("--max-angle-x", type=int, default=0, help="rotate angle")
        self.parser.add_argument("--max-angle-y", type=int, default=0, help="rotate angle")
        self.parser.add_argument("--max-angle-z", type=int, default=360, help="rotate angle")
        self.parser.add_argument("--max-translate-x", type=int, default=400, help="translate value")
        self.parser.add_argument("--max-translate-y", type=int, default=400, help="translate value")
        self.parser.add_argument("--max-translate-z", type=int, default=0, help="translate value")
        '''
        self.parser.add_argument("--use-log-images", default=True, help="irradiance maps = log(gray map)")
        self.parser.add_argument("--log-eps", type=int, default=0.001, help="E = log(L/255+eps)")
        self.parser.add_argument("--log-threshold", type=int, default=20, help="x>=T E = log(L) or x<T E = L/Tlog(T)")
        
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
