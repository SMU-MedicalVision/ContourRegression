import cv2
import torch 
import numpy as np
import torch.nn.functional as F
from shapely.geometry import Polygon


def outcode2MALU(outcode):
    media, lumen = outcode.chunk(2, dim=1)
    lumen1, lumen2 = lumen.chunk(2, dim=1)
    lumen = torch.stack([lumen1, lumen2], dim = 2)
    media1, media2 = media.chunk(2,dim = 1)
    media = torch.stack([media1,media2],dim = 2)
    return media+255.5, lumen+255.5


def polyarea(poly:torch.Tensor):
    x = poly[:, 0]
    y = poly[:, 1]
    # x = x - x.mean()
    # y = y - y.mean()
    area = 0.5 * (torch.dot(x, torch.roll(y, 1)) - torch.dot(y, torch.roll(x, 1)))
    area = torch.abs(area)
    return area

def poly_intersection(tensor_poly1, tensor_poly2):
    np_poly1 = tensor_poly1.detach().numpy()
    np_poly2 = tensor_poly2.detach().numpy()
    np_poly1 = np.squeeze(np_poly1)
    np_poly2 = np.squeeze(np_poly2)

    poly1 = Polygon(np_poly1)
    poly2 = Polygon(np_poly2)
    if not poly1.intersects(poly2):
        return None, None

    poly3 = poly2.intersection(poly1)
    # print(poly3.area)
    # poly3 = poly2.union(poly1)

    np_poly3 = np.asarray(poly3.exterior.coords)
    tensor_inter_poly3 = torch.from_numpy(np_poly3)

    vertice_idx = lookfor_vertices_line_indices(np_poly3, np_poly1, np_poly2)
    vertice_idx = torch.from_numpy(vertice_idx).long()

    return tensor_inter_poly3, vertice_idx


def lookfor_vertices_line_indices(np_inter_poly, np_poly1, np_poly2):
    eps = 1e-6

    x1 = np_poly1[:, 0]
    y1 = np_poly1[:, 1]
    x2 = np_poly2[:, 0]
    y2 = np_poly2[:, 1]

    n = np_inter_poly.shape[0]
    vertice_idx = np.zeros((n, 3), dtype=np.int64)

    for i in range(n):
        xp = np_inter_poly[i, 0]
        yp = np_inter_poly[i, 1]

        equal_vec = np.bitwise_and(x1 == xp, y1 == yp)
        if equal_vec.sum() > 0:
            ind = np.where(equal_vec)
            vertice_idx[i, 0] = 1
            vertice_idx[i, 1] = ind[0].item()
            vertice_idx[i, 2] = ind[0].item()
        else:
            equal_vec = np.bitwise_and(x2 == xp, y2 == yp)
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


def iou_dice_calc(pred_poly, gt_poly):
    tensor_poly1 = pred_poly.detach().double().cpu()  # n x 2
    tensor_poly2 = gt_poly.detach().double().cpu()

    area1 = polyarea(tensor_poly1)
    area2 = polyarea(tensor_poly2)

    np_poly1 = tensor_poly1.detach().numpy()
    np_poly2 = tensor_poly2.detach().numpy()
    np_poly1 = np.squeeze(np_poly1)
    np_poly2 = np.squeeze(np_poly2)

    poly1 = Polygon(np_poly1)
    poly2 = Polygon(np_poly2)

    if not poly1.intersects(poly2):
        return None, None
    if poly1.is_valid == False:
        poly1 = poly1.buffer(0)
    if poly2.is_valid == False:
        poly2 = poly2.buffer(0)

    poly3 = poly2.intersection(poly1)
    np_poly3 = np.asarray(poly3.exterior.coords)
    inter_poly = torch.from_numpy(np_poly3)

    if inter_poly is None:
        # iou_area = 0.
        iou = 0
        dice = 0
    else:
        inter_area = polyarea(inter_poly)
        dice = 2 * inter_area / (area1 + area2)
    return dice


def Evaluation_calc(lumen, media, mask):
    batch = lumen.shape[0]
    lDice, lmcd, mDice, mmcd = [], [], [], []
    lhd, mhd = [], []
    for i in range(batch):
        mask_lumen, mask_media = np.zeros((512,512)).astype(np.uint8), np.zeros((512,512)).astype(np.uint8)
        mask_lumen[mask[i,:,:] == 2], mask_media[mask[i,:,:] != 0] = 255, 255
        _, pointslumen, _ = cv2.findContours(mask_lumen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        _, pointsmedia, _ = cv2.findContours(mask_media, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        index_lumen = [i.shape[0] for i in pointslumen]
        index_lumen = index_lumen.index(max(index_lumen))
        index_media = [i.shape[0] for i in pointsmedia]
        index_media = index_media.index(max(index_media))
        pointslumen, pointsmedia = pointslumen[index_lumen].squeeze(), pointsmedia[index_media].squeeze()

        pointslumen, pointsmedia = torch.tensor(pointslumen).double(), torch.tensor(pointsmedia).double()
        dice = iou_dice_calc(lumen[i,:,:].detach(), pointslumen)  #
        lDice.append(dice)
        dice = iou_dice_calc(media[i, :, :].detach(), pointsmedia)
        mDice.append(dice)
        pointslumen = F.interpolate(pointslumen.detach().permute(1, 0).unsqueeze(0), (2048), mode='linear',
                                    align_corners=True)  #
        pointslumen = pointslumen.squeeze().permute(1, 0)
        pointsmedia = F.interpolate(pointsmedia.detach().permute(1, 0).unsqueeze(0), (2048), mode='linear',
                                    align_corners=True)  #
        pointsmedia = pointsmedia.squeeze().permute(1, 0)
        lhd.append(hausdorff_distance(lumen[i, :, :], pointslumen))
        mhd.append(hausdorff_distance(media[i, :, :], pointsmedia))

        lasd = asd(lumen[i, :, :].detach().cpu(), pointslumen)  #
        lmcd.append(lasd)
        masd = asd(media[i, :, :].detach().cpu(), pointsmedia)
        mmcd.append(masd)
    return lDice, lmcd, mDice, mmcd, lhd, mhd

def hausdorff_distance(lumensig, pointslumen):
    A, B = lumensig.detach().cpu(), pointslumen.detach().cpu()
    xa, ya = A[:,0], A[:,1]
    xb, yb = B[:,0], B[:,1]
    xa = torch.stack([xa for i in range(B.shape[0])], dim=-1)
    ya = torch.stack([ya for i in range(B.shape[0])], dim=-1)
    xb = torch.stack([xb for i in range(A.shape[0])], dim=0)
    yb = torch.stack([yb for i in range(A.shape[0])], dim=0)
    ab = torch.sqrt((xa-xb)**2 + (ya-yb)**2)
    hab, _ = torch.min(ab, dim=0)
    hab = torch.max(hab)
    hba, _ = torch.min(ab, dim=-1)
    hba = torch.max(hba)
    return max(hab, hba)


def bresenham(x0, y0, x1, y1):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    Input coordinates should be integers.
    The result will contain both the start and the end point.
    """
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(int(dx) + 1):
        yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy


def FillBdryGaps(inBdry):
    n = inBdry.shape[0]
    outBdry = np.empty(shape=[0,2])
    for i in range(n-1):
        if abs(inBdry[i+1,0]-inBdry[i,0])>1 or abs(inBdry[i+1,1]-inBdry[i,1])>1:
            rc = np.array(list(bresenham(inBdry[i,0], inBdry[i,1], inBdry[i+1,0], inBdry[i+1,1])))
            outBdry = np.append(outBdry,rc[:-1,:],axis=0)
        else:
            temp = inBdry[i,:]; temp = np.transpose(temp[:,np.newaxis])
            outBdry = np.append(outBdry,temp,axis=0)
    if abs(inBdry[0,0]-inBdry[n-1,0])>1 or abs(inBdry[0,1]-inBdry[n-1,1])>1:
        rc = np.array(list(bresenham(inBdry[n-1,0], inBdry[n-1,1], inBdry[0,0], inBdry[0,1])))
        outBdry = np.append(outBdry,rc[:-1,:],axis=0)
    else:
        temp = inBdry[n-1,:]; temp = np.transpose(temp[:,np.newaxis])
        outBdry = np.append(outBdry,temp,axis=0)
    return outBdry


def asd(predpoints, gtpoints):
    gt_to_pred = gtpoints.unsqueeze(1)
    gt_to_pred = gt_to_pred.repeat(1, (predpoints.shape[0]), 1)
    temp = predpoints.unsqueeze(0)
    temp = temp.repeat((gtpoints.shape[0]), 1, 1)
    gt_to_pred = torch.sqrt(torch.sum((gt_to_pred - temp) ** 2, dim=2))
    temp, _ = torch.min(gt_to_pred, dim=0)
    pred_to_gt = max(temp)
    temp, _ = torch.min(gt_to_pred, dim=1)
    gt_to_pred = max(temp) 
    return (pred_to_gt + gt_to_pred) / 2
