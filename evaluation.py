# import torch modeuls
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import inside modeuls
import warnings
from argparse import ArgumentParser
import numpy as np
import os
# import my modeuls
from util.utilFunctions import outcode2MALU, Evaluation_calc
from data_perpare.DatasetGen import DatasetGenerator
from network.EfficientNet import EfficientNet
from encoding.EncodProcess import decode


def run_eval(model, data_loader, device, pca_inverse, pca_mean, sigatureMean):
    # switch to evaluate mode
    model.eval()
    print("Running validation...")
    
    dice_lumen, JI_lumen = [],[]
    dice_media, JI_media = [],[]
    hd_lumen, hd_media = [],[]
    with torch.no_grad():
        for [img, mask, _, _, _] in data_loader:
            img = img.to(device)
            outcode = model(img)
            outcode = decode(outcode, pca_inverse[:128, :], pca_mean, sigatureMean)  #[:512,:]
            media, lumen = outcode2MALU(outcode)
            ldice, lJI, mdice, mJI, lhd, mhd = Evaluation_calc(lumen, media, mask)
            dice_lumen.append(ldice)
            JI_lumen.append(lJI)
            hd_lumen.append(lhd)
            dice_media.append(mdice)
            JI_media.append(mJI)
            hd_media.append(mhd)
    JI_lumen = [i for j in JI_lumen for i in j]
    JI_media = [i for j in JI_media for i in j]
    dice_lumen = [i for j in dice_lumen for i in j]
    dice_media = [i for j in dice_media for i in j]
    hd_lumen = [i for j in hd_lumen for i in j]
    hd_media = [i for j in hd_media for i in j]
    print('MEAN:')
    print('lumen dice: %.4f, \t media dice: %.4f' % (np.mean(dice_lumen), np.mean(dice_media)))
    print('lumen AMCD: %.4f, \t media AMCD: %.4f' % (np.mean(JI_lumen), np.mean(JI_media)))
    print('lumen HD:   %.4f, \t media HD:   %.4f' % (np.mean(hd_lumen), np.mean(hd_media)))
    print('STD:')
    print('lumen dice: %.4f, \t media dice: %.4f' % (np.std(dice_lumen), np.std(dice_media)))
    print('lumen AMCD: %.4f, \t media AMCD: %.4f' % (np.std(JI_lumen), np.std(JI_media)))
    print('lumen HD:   %.4f, \t media HD:   %.4f' % (np.std(hd_lumen), np.std(hd_media)))

def main(args):
    # set the running device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if torch.cuda.is_available() and args.gpu != '-1' and args.gpu != 'cpu':
        device = torch.device('cuda')
    if args.gpu == '-1' or args.gpu == 'cpu':
        device = torch.device('cpu')
    
    # make the train&eval&test dataloader
    testpath = '/home/NeverDie/data/testimg/'
    testmaskpath = '/home/NeverDie/data/testmask/'
    datasetTest = DatasetGenerator(testpath, testmaskpath, if_train=False)
    test_loader = DataLoader(dataset=datasetTest, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    # Build the network
    model = EfficientNet.from_name('efficientnet-b0')
    model = nn.DataParallel(model)  # if multiGPU
    
    # Resume fine-tuning if we find a saved model.
    checkpoint = torch.load(args.modeldir, map_location="cpu")
    step = checkpoint["step"]
    model.load_state_dict(checkpoint["model"])
    print(f"Resumed at step {step}")
    model = model.to(device)
    
    # perpare the decoding matfile
    cwdir = '/home/NeverDie/code/HMCCRNet/encoding/'
    pca_inverse = np.load(os.path.join(cwdir,'pca_inverse.npy'))
    pca_mean = np.load(os.path.join(cwdir,'pca_mean.npy'))
    sigatureMean = np.load(os.path.join(cwdir, 'sigatureMean.npy'))

    run_eval(model, test_loader, device, pca_inverse, pca_mean, sigatureMean)
    print('Finish evaluating.')


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = ArgumentParser()

    parser.add_argument("--gpu", type=str, default='0', help="gpu id")
    parser.add_argument("--batch_size", type=int, dest="batch_size", 
                        default=16, help="batch_size")
    parser.add_argument("--num_workers", type=int, dest="num_workers",
                        default=2, help="num_workers")
    parser.add_argument("--modeldir", type=str, dest="modeldir", 
                        default=None,
                        help="models folder")

    main(parser.parse_args())
