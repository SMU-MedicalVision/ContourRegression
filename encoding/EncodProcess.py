import torch


def decode(outcode,pca_inverse, pca_mean, sigatureMean):
    '''
    param outcode: the output of network encoder part. size: [batchsize,2048]
    '''
    pca_inverse = torch.tensor(pca_inverse).float().to('cuda')
    sigatureMean = torch.tensor(sigatureMean).float().to('cuda')
    pca_mean = torch.tensor(pca_mean).float().to('cuda')
    
    result = torch.matmul(outcode,pca_inverse) + pca_mean
    result = result + sigatureMean
    return result
