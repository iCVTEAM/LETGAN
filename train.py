import os
from evgannet import EvGanNet
from Options.trainOptions import TrainOptions
    
def main(args):
    '''
        Train the model and record training options.
    '''
    savedir = os.path.join(args.save_dir, args.model)
    modeltxtpath = os.path.join(savedir,'model.txt') 

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    with open(savedir + '/opts.txt', "w") as myfile: #record options
        myfile.write(str(args))
        
    #model = getModel(args)     #load model
    
    #with open(modeltxtpath, "w") as myfile:  #record model 
    #    myfile.write(str(model))
        
    #if args.cuda:
    #    model = model.cuda() 
    print("========== TRAINING ===========")
    evGanNet = EvGanNet(args,savedir)
    evGanNet.train()
    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    parser = TrainOptions().parse()
    main(parser)
