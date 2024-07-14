import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
from argparse import ArgumentParser
import numpy as np
import os
from data_perpare.DatasetGen import DatasetGenerator
from torch.utils.data.distributed import DistributedSampler
from network.EfficientNet import EfficientNet
from util.earlystopping import EarlyStopping
from encoding.EncodProcess import decode
from util.losses import Weighted_L2_loss, L2_loss


def fix_seed(seed):
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True 
    torch.manual_seed(seed)


def main(args):
    # 1- set the random seed
    if args.fix_seed == None:
        torch.backends.cudnn.benchmark = True
    else:
        fix_seed(args.fix_seed)
    # 2- set the running device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda')
    # 3- set all the datafolders
    trainpath='./trainimg/'
    trainmaskpath='./trainmask/'
    valpath='./valimg/'
    valmaskpath='./valmask/'
    # 4- make the train and eval dataloader
    datasetTrain = DatasetGenerator(trainpath, trainmaskpath, if_train=True)
    train_loader = DataLoader(dataset=datasetTrain, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    datasetVal = DatasetGenerator(valpath, valmaskpath, if_train=False)
    val_loader = DataLoader(dataset=datasetVal, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    # 5- Build the network
    model = EfficientNet.from_name('efficientnet-b0', dim=128)
    model = nn.DataParallel(model)
    if args.pretrained:
        # load the pretrained model
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        step = checkpoint["step"]
        model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    # 6- Set optimizer and losses
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[200, 400, 600, 800], gamma=0.1)
    early_stop = EarlyStopping(model_path=args.models_dir+args.name, persistence=40)
    # 7- decoder perpare
    cwdir = './encoding/'
    pca_inverse = np.load(os.path.join(cwdir, 'pca_inverse.npy'))
    pca_mean = np.load(os.path.join(cwdir, 'pca_mean.npy'))
    sigatureMean = np.load(os.path.join(cwdir, 'sigatureMean.npy'))
    # 8- loss function
    sim_loss_fn = Weighted_L1_loss
    sim_loss_fn2 = Weighted_PIoU_loss
    # 9- Training loop.
    alpha = 0
    for epoch in range(args.n_epoch):
        if epoch == 0:
            datasetTrain.set_prob(alpha)
        elif epoch % 10 == 0 and alpha < 2:
            alpha += 0.1
        model.train()
        total_train_loss = 0
        for [img, _, ContourLumen, ContourMedia, prob] in train_loader:
            img = img.to(device)
            ContourLumen, ContourMedia = ContourLumen.to(device), ContourMedia.to(device)

            # Run the data through the model to produce warp and flow field
            outcode = model(img)
            outcode = decode(outcode, pca_inverse[:128, :], pca_mean, sigatureMean)
            
            # Calculate loss
            loss1 = sim_loss_fn(outcode, ContourLumen, ContourMedia, prob)
            loss2 = sim_loss_fn2(outcode, ContourLumen, ContourMedia, prob)
            loss = loss1 + loss2
            total_train_loss += loss.item()
            
            # Backwards and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_train_loss /= len(train_loader) 
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for [img, _, ContourLumen, ContourMedia, _] in val_loader:
                img = img.to(device)
                ContourLumen, ContourMedia = ContourLumen.to(device), ContourMedia.to(device)
                outcode = model(img)
                outcode = decode(outcode, pca_inverse[:128, :], pca_mean, sigatureMean)
                loss1 = sim_loss_fn(outcode, ContourLumen, ContourMedia, prob)
                loss2 = sim_loss_fn2(outcode, ContourLumen, ContourMedia, prob)
                val_loss = loss1 + loss2
                total_val_loss += val_loss.item()
        total_val_loss /= len(val_loader)
        model_dict = {"step": epoch, "model": model.state_dict(),
                      "optim" : opt.state_dict()}
        early_stop(model_dict, total_val_loss, epoch)
        scheduler.step()
        if early_stop.criterion:
            break
    print('finished training')

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = ArgumentParser()
    parser.add_argument("--fix_seed", type=int, dest="fix_seed", default=None,
                        help="If set seed number, then fix the seed. Default[None]: without random fixing seed.")
    parser.add_argument("--models_dir", type=str, dest="models_dir", default='./results/',
                        help="Models folder for saving all the results.")
    parser.add_argument("--name", default='ccrnetplus', help="Name of this run. Used for monitoring and checkpointing.")
    parser.add_argument("--gpu", type=str, default='0', help="gpu id")
    parser.add_argument("--batch_size", type=int, dest="batch_size",
                        default=16, help="batch_size")
    parser.add_argument("--num_workers", type=int, dest="num_workers",
                        default=4, help="num_workers")
    parser.add_argument("--lr", type=float, dest="lr", default=1e-4,
                        help="The training learning rate")
    parser.add_argument("--n_epoch", type=int, dest="n_epoch", default=1000)
    parser.add_argument("--pretrained", type=str, dest="pretrained", default=None,
                        help="The resume model or checkpoint path.")

    main(parser.parse_args())
