#-*- coding:utf-8 -*-
import argparse
import os

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--model', default="generator", help='model to train,options:fcn8,segnet...')  
        self.parser.add_argument('--model-dir', default="/media/localdisk1/usr/gdx/project/Pytorch/PyTorchModel/resnet34-333f7ec4.pth", help='path to stored-model')# vgg16-397923af.pth resnet34-333f7ec4.pth resnet50-19c8e357.pth
        #self.parser.add_argument('--model-dir', default="/media/localdisk1/usr/gdx/code/20201006_ED/code_20201018/Model/TrainingModels/20201011/generator_139.pth", help='path to stored-model')#resnet50-19c8e357.pth vgg16-397923af.pth
        '''old
        self.parser.add_argument('--data-real-images', default="./Data/V2E/image.txt",help='path where image.txt lies')
        self.parser.add_argument('--data-event-maps', default="./Data/V2E/event.txt",help='path where image.txt lies')
        self.parser.add_argument('--data-images', default="./Data/V2E/image.txt",help='path where image.txt lies')
        self.parser.add_argument("--max-angle-x", type=int, default=0.5, help="rotate angle")
        self.parser.add_argument("--max-angle-y", type=int, default=0.5, help="rotate angle")
        self.parser.add_argument("--max-angle-z", type=int, default=0.5, help="rotate angle")
        self.parser.add_argument("--max-translate-x", type=int, default=5, help="translate value")
        self.parser.add_argument("--max-translate-y", type=int, default=5, help="translate value")
        self.parser.add_argument("--max-translate-z", type=int, default=0, help="translate value")
        '''
        #new
        self.parser.add_argument('--data-size', type=int, default=1000,help='path where image.txt lies')
        #self.parser.add_argument('--data-size', type=int, default=5000,help='path where image.txt lies')
        #self.parser.add_argument('--data-size', type=int, default=9000,help='path where image.txt lies')
        #self.parser.add_argument('--data-real', default="./Data/NEW/real",help='path wherere image.txt lies')
        self.parser.add_argument('--data-real', default="./Data/ACMMM/real2",help='path wherere image.txt lies')
        #self.parser.add_argument('--data-real', default="./Data/TPAMI/real",help='path wherere image.txt lies')
        #self.parser.add_argument('--data-real', default="../ED/Data/NEW/mvsec",help='path where image.txt lies')
        self.parser.add_argument('--data-simu', default="./Data/NEW/simu",help='path where image.txt lies')
        
        self.parser.add_argument('--size', default=(160,160), help='resize the test image')
        self.parser.add_argument('--stored',default=True, help='whether or not store the result')
        self.parser.add_argument('--save-dir', type=str,default='./Model/TrainingModels/23/',help='savedir for models')#old20201126
        self.parser.add_argument('--tb-path', type=str,default='./Model/TrainingModels/23/',help='savedir for tensorboardX')#old20201126
        self.parser.add_argument('--debug', action='store_true', default=False) # debug mode 
        
        self.parser.add_argument("--patch", type=int, default=4, help="image patch size")
        self.parser.add_argument("--Qangle", type=int, default=24, help="Training Qangle size")
        self.parser.add_argument("--Qstrength", type=int, default=3, help="Training Qstrength size")
        self.parser.add_argument("--Qcoherence", type=int, default=3, help="Training Qcoherence size")
        
        self.parser.add_argument('--g-lr', type=float, default=4e-4)
        self.parser.add_argument('--d-lr', type=float, default=4e-4)
        
        self.parser.add_argument('--train-g-interval', type=int, default=5)
        self.parser.add_argument('--num-epochs', type=int, default=500)
        self.parser.add_argument('--num-workers', type=int, default=16)
        self.parser.add_argument('--batch-size', type=int, default=12)
        self.parser.add_argument('--d-batch-size', type=int, default=256)
        self.parser.add_argument('--epoch-save', type=int, default=5)    #You can use this value to save model every X epochs
        self.parser.add_argument('--show-interval', type=int, default=1)
        self.parser.add_argument('--steps-loss', type=int, default=20)
        self.parser.add_argument('--warmup', type=int, default=0)
        self.parser.add_argument('--lr-end', type=int, default=1e-5)
        self.parser.add_argument('--trainstep', type=int, default=100)
        
        self.parser.add_argument('--minimum-threshold', type=float, default=0.01)
        
        #self.parser.add_argument("--patch", type=int, default=8, help="image patch size")
        #self.parser.add_argument("--Qangle", type=int, default=12, help="Training Qangle size")
        #self.parser.add_argument("--Qstrength", type=int, default=3, help="Training Qstrength size")
        #self.parser.add_argument("--Qcoherence", type=int, default=3, help="Training Qcoherence size")
        

        self.parser.add_argument("--use-log-images", default=True, help="irradiance maps = log(gray map)")
        self.parser.add_argument("--log-eps", type=int, default=0.001, help="E = log(L/255+eps)")
        self.parser.add_argument("--log-threshold", type=int, default=20, help="x>=T E = log(L) or x<T E = L/Tlog(T)")
        
        self.parser.add_argument("--clamp-num", type=float, default=0.01, help="WGAN clip gradient")
        self.parser.add_argument("--gp-lambda", type=float, default=1, help="WGAN gradient penalty")
        self.parser.add_argument("--min-event-num", type=float, default=10, help="avoid all black")
        
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
