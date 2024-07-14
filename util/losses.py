# import all needed modules
import math
import torch
import torch.nn.functional as F
import numpy as np
from shapely.geometry import Polygon
from typing import Optional
import torch.distributed as dist


def Weighted_L2_loss(outcode, ContourLumen, ContourMedia, prob):
    '''
    param outcode: output of model, which the size is [batchsize, 2048]
    param ContourLumen: the true encoded lumen signatures, size is [batchsize,512,2]
    param ContourMedia: the true encoded media signatures, size is [batchsize,512,2]
    '''
    lumen = torch.cat([ContourLumen[:, :, 0], ContourLumen[:, :, 1]], dim=1)
    media = torch.cat([ContourMedia[:, :, 0], ContourMedia[:, :, 1]], dim=1)
    incode = torch.cat([media, lumen], dim=1)
    prob = torch.from_numpy(np.squeeze(normalize(np.expand_dims(prob, 1), axis=0, norm='l1'))).to('cuda')
    mse = torch.mean((outcode - incode) ** 2, dim=1)
    return torch.sum(mse * prob)


def Weighted_PIOU_loss(outcode, ContourLumen, ContourMedia, prob):
    '''
    param outcode: output of model, which the size is [batchsize, 2048]
    param ContourLumen: the true encoded lumen signatures, size is [batchsize,512,2]
    param ContourMedia: the true encoded media signatures, size is [batchsize,512,2]
    '''
    media, lumen = outcode[:, :1024], outcode[:, 1024:]
    lumen1, lumen2 = lumen.chunk(2,dim = 1)
    lumen = torch.stack([lumen1,lumen2],dim = 2)
    media1, media2 = media.chunk(2,dim = 1)
    media = torch.stack([media1,media2],dim = 2)
    lumen_iouloss = polygon_iou_loss(lumen, ContourLumen, prob)
    media_iouloss = polygon_iou_loss(media, ContourMedia, prob)
    iou_loss = lumen_iouloss + media_iouloss
    return iou_loss / 2


def polygon_iou_loss(pred_poly_batch, gt_poly_batch, prob):
    eps = 1e-2
    b, n, c = pred_poly_batch.shape
    iou_loss = 0.
    for i in range(b):
        tensor_poly1 = pred_poly_batch[i, :, :].squeeze() #n x 2
        tensor_poly2 = gt_poly_batch[i, :, :].squeeze()   #n x 2

        area1 = polyarea(tensor_poly1)
        area2 = polyarea(tensor_poly2)

        inter_poly, vertice_idx = poly_intersection(tensor_poly1, tensor_poly2)
        if inter_poly is None:
            iou_loss = iou_loss + 1.
        else:
            inter_poly = inter_poly.detach()
            vertice_idx = vertice_idx.detach()

            is_not_located = vertice_idx[:, 0] < 1
            assert is_not_located.sum() == 0

            inter_mask = vertice_idx[:, 0] == 3
            if inter_mask.sum() > 0:
                idx1 = vertice_idx[inter_mask, 1]
                idx2 = vertice_idx[inter_mask, 2]
                # idx1, idx2 to 'cuda'

                tensor_poly11 = torch.roll(tensor_poly1, -1, 0)
                tensor_poly21 = torch.roll(tensor_poly2, -1, 0)

                tensor_pts1 = torch.index_select(tensor_poly1, 0, idx1.to('cuda'))
                tensor_pts2 = torch.index_select(tensor_poly11, 0, idx1.to('cuda'))
                tensor_pts3 = torch.index_select(tensor_poly2, 0, idx2.to('cuda'))
                tensor_pts4 = torch.index_select(tensor_poly21, 0, idx2.to('cuda'))

                inter_pts = pl_intersect(tensor_pts1, tensor_pts2, tensor_pts3, tensor_pts4)
                tensor_cat_poly = torch.cat((tensor_poly1, tensor_poly2, inter_pts), dim=0)
            else:
                tensor_cat_poly = torch.cat((tensor_poly1, tensor_poly2), dim=0)

            n1, c = tensor_poly1.shape
            n2, c = tensor_poly2.shape

            sorted_idx = vertice_idx[:, 1]
            n3, c = vertice_idx.shape
            inter_count = 0
            for k in range(n3):
                if vertice_idx[k, 0] == 2:
                    sorted_idx[k] = sorted_idx[k] + n1
                if vertice_idx[k, 0] == 3:
                    sorted_idx[k] = n1 + n2 + inter_count
                    inter_count = inter_count + 1

            tensor_inter_poly = torch.index_select(tensor_cat_poly, 0, sorted_idx.to('cuda'))

            inter_area = polyarea(tensor_inter_poly)
            iou_loss = (iou_loss + 1. - inter_area / (area1 + area2 - inter_area + eps)) * prob[i]

    iou_loss = iou_loss / b

    return iou_loss


def polyarea(poly:torch.Tensor):
    x = poly[:, 0]
    y = poly[:, 1]
    # x = x - x.mean()
    # y = y - y.mean()
    area = 0.5 * (torch.dot(x, torch.roll(y, 1)) - torch.dot(y, torch.roll(x, 1)))
    area = torch.abs(area)
    return area


def poly_intersection(tensor_poly1, tensor_poly2):
    np_poly1 = tensor_poly1.cpu().detach().numpy()
    np_poly2 = tensor_poly2.cpu().detach().numpy()
    np_poly1 = np.squeeze(np_poly1)
    np_poly2 = np.squeeze(np_poly2)

    poly1 = Polygon(np_poly1)
    poly2 = Polygon(np_poly2)
    if not poly1.intersects(poly2):
        return None, None

    try:
        poly1.intersection(poly1)
    except:
        return None, None
    try:
        poly3 = poly2.intersection(poly1)
    except:
        return None, None

    if not poly3.type == 'Polygon':
        return None, None

    np_poly3 = np.asarray(poly3.exterior.coords)
    tensor_inter_poly3 = torch.from_numpy(np_poly3)

    vertice_idx = lookfor_vertices_line_indices(np_poly3, np_poly1, np_poly2)
    vertice_idx = torch.from_numpy(vertice_idx).long()

    return tensor_inter_poly3, vertice_idx

def lookfor_vertices_line_indices(np_inter_poly, np_poly1, np_poly2):
    eps = 1e-4

    x1 = np_poly1[:, 0]
    y1 = np_poly1[:, 1]
    x2 = np_poly2[:, 0]
    y2 = np_poly2[:, 1]

    n = np_inter_poly.shape[0]
    vertice_idx = np.zeros((n, 3), dtype=np.int64)

    for i in range(n):
        xp = np_inter_poly[i, 0]
        yp = np_inter_poly[i, 1]

        equal_vec = np.bitwise_and(abs(x1-xp) <= 1e-5, abs(y1-yp) <= 1e-5)
        if equal_vec.sum() > 0:
            ind = np.where(equal_vec)
            vertice_idx[i, 0] = 1
            vertice_idx[i, 1] = ind[0].item()
            vertice_idx[i, 2] = ind[0].item()
        else:
            equal_vec = np.bitwise_and(abs(x2-xp) <= 1e-5, abs(y2-yp) <= 1e-5)
            if equal_vec.sum() > 0:
                ind = np.where(equal_vec)
                vertice_idx[i, 0] = 2
                vertice_idx[i, 1] = ind[0].item()
                vertice_idx[i, 2] = ind[0].item()

    np_poly11 = np.roll(np_poly1, -1, axis=0)
    x1 = np_poly1[:, 0]
    y1 = np_poly1[:, 1]
    x2 = np_poly11[:, 0]
    y2 = np_poly11[:, 1]

    x_min = np.minimum(x1, x2)
    x_max = np.maximum(x1, x2)
    y_min = np.minimum(y1, y2)
    y_max = np.maximum(y1, y2)

    is_not_rep = np.bitwise_and(x1 == x2, y1 == y2)
    is_not_rep = np.bitwise_not(is_not_rep)

    for i in range(n):
        if vertice_idx[i, 0] < 1:
            xp = np_inter_poly[i, 0]
            yp = np_inter_poly[i, 1]

            # tmp = (xp - x1) * (y2 - y1) - (x2 - x1) * (yp - y1)
            is_cross_vec = np.abs((xp - x1) * (y2 - y1) - (x2 - x1) * (yp - y1)) <= eps

            is_in_xbounds = np.bitwise_and(xp >= x_min, xp <= x_max)
            is_in_ybounds = np.bitwise_and(yp >= y_min, yp <= y_max)
            is_in_bounds = np.bitwise_and(is_in_xbounds, is_in_ybounds)

            is_on_line = np.bitwise_and(is_cross_vec, is_in_bounds)
            is_on_line = np.bitwise_and(is_on_line, is_not_rep)

            if np.sum(is_on_line) >= 1:
                ind = np.where(is_on_line)
                vertice_idx[i, 0] = 3
                if ind[0].shape[0] > 1:
                    vertice_idx[i, 1] = ind[0][1].item()
                else:
                    vertice_idx[i, 1] = ind[0].item()

    np_poly21 = np.roll(np_poly2, -1, axis=0)
    x1 = np_poly2[:, 0]
    y1 = np_poly2[:, 1]
    x2 = np_poly21[:, 0]
    y2 = np_poly21[:, 1]

    x_min = np.minimum(x1, x2)
    x_max = np.maximum(x1, x2)
    y_min = np.minimum(y1, y2)
    y_max = np.maximum(y1, y2)

    is_not_rep = np.bitwise_and(x1 == x2, y1 == y2)
    is_not_rep = np.bitwise_not(is_not_rep)

    for i in range(n):
        if vertice_idx[i, 0] == 3:
            xp = np_inter_poly[i, 0]
            yp = np_inter_poly[i, 1]

            # tmp = (xp - x1) * (y2 - y1) - (x2 - x1) * (yp - y1)
            is_cross_vec = np.abs((xp - x1) * (y2 - y1) - (x2 - x1) * (yp - y1)) <= eps

            is_in_xbounds = np.bitwise_and(xp >= x_min, xp <= x_max)
            is_in_ybounds = np.bitwise_and(yp >= y_min, yp <= y_max)
            is_in_bounds = np.bitwise_and(is_in_xbounds, is_in_ybounds)

            is_on_line = np.bitwise_and(is_cross_vec, is_in_bounds)
            is_on_line = np.bitwise_and(is_on_line, is_not_rep)

            if np.sum(is_on_line) >= 1:
                ind = np.where(is_on_line)
                vertice_idx[i, 0] = 3
                if ind[0].shape[0] > 1:
                    vertice_idx[i, 2] = ind[0][1].item()
                else:
                    vertice_idx[i, 2] = ind[0].item()

    return vertice_idx

def pl_intersect(tensor_pts1, tensor_pts2, tensor_pts3, tensor_pts4):
    eps = 1e-6
    x1 = tensor_pts1[:, 0]
    y1 = tensor_pts1[:, 1]
    x2 = tensor_pts2[:, 0]
    y2 = tensor_pts2[:, 1]
    x3 = tensor_pts3[:, 0]
    y3 = tensor_pts3[:, 1]
    x4 = tensor_pts4[:, 0]
    y4 = tensor_pts4[:, 1]

    num = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    den_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)

    tmp_t = den_t / num
    cross_flag = (tmp_t > 0) * (tmp_t < 1)
    cross_flag = ~ cross_flag
    assert cross_flag.sum() == 0

    t = den_t / (num + eps)

    inter_pts = torch.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dim=-1)

    return inter_pts

def poly_interp(tensor_poly, scale_factor=10):
    n, c = tensor_poly.shape
    tensor_poly1 = torch.roll(tensor_poly, -1, 0)
    ploy_list = []
    ploy_list.append(tensor_poly)
    for i in range(1, scale_factor, 1):
        weight = i / scale_factor
        tmp_poly = (1. - weight) * tensor_poly + weight * tensor_poly1
        ploy_list.append(tmp_poly)

    tensor_poly1 = torch.stack(ploy_list, dim=1)
    tensor_poly1 = tensor_poly1.reshape(n * scale_factor, c)

    return tensor_poly1

def _assert_no_grad(variables):
    for var in variables:
        assert not var.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"

def cdist(x, y):
    '''
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances
