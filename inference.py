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
from network.EfficientNet import EfficientNet
from encoding.EncodProcess import decode

def load_image(args):
    imagedata = np.load(args.img_path)[6:9, :, :]

    imagedata = imagedata.astype(np.float32)
    for i in range(3):
        imagedata[i] -= np.mean(imagedata[i])
        imagedata[i] /= np.std(imagedata[i])

    return torch.from_numpy(imagedata.copy())

def run_eval(model, img, device, pca_inverse, pca_mean, sigatureMean):
    # switch to evaluate mode
    model.eval()
    print("Running validation...")
    
    dice_lumen, JI_lumen = [],[]
    dice_media, JI_media = [],[]
    hd_lumen, hd_media = [],[]
    with torch.no_grad():
        img = img.to(device)
        outcode = model(img)
        outcode = decode(outcode, pca_inverse[:128, :], pca_mean, sigatureMean)
        media, lumen = outcode2MALU(outcode)
    return media, lumen

def main(args):
    # set the running device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if torch.cuda.is_available() and args.gpu != '-1' and args.gpu != 'cpu':
        device = torch.device('cuda')
    if args.gpu == '-1' or args.gpu == 'cpu':
        device = torch.device('cpu')
    
    # make the test dataloader
    img = load_image(args)
    
    # Build the network
    model = EfficientNet.from_name('efficientnet-b0')
    model = nn.DataParallel(model)  # if multiGPU
    
    # load the model
    checkpoint = torch.load(args.modeldir, map_location="cpu")
    step = checkpoint["step"]
    model.load_state_dict(checkpoint["model"])
    print(f"Resumed at step {step}")
    model = model.to(device)
    
    # perpare the decoding matfile
    cwdir = './encoding/'
    pca_inverse = np.load(os.path.join(cwdir,'pca_inverse.npy'))
    pca_mean = np.load(os.path.join(cwdir,'pca_mean.npy'))
    sigatureMean = np.load(os.path.join(cwdir, 'sigatureMean.npy'))

    media, lumen = run_eval(model, img, device, pca_inverse, pca_mean, sigatureMean)
    print('Finish inferencing.')


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = ArgumentParser()

    parser.add_argument("--gpu", type=str, default='0', help="gpu id")
    parser.add_argument("--img_path", type=str, dest="img_path", 
                        default=None,
                        help="img path")
    parser.add_argument("--modeldir", type=str, dest="modeldir", 
                        default=None,
                        help="models folder")

    main(parser.parse_args())
