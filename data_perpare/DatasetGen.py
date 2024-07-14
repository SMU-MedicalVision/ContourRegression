# import torch modeuls
import torch
import torch.nn as nn
from torch.utils.data import Dataset
# import inside modeuls
import os
import cv2
import random
import numpy as np
import numpy.random
import torch.nn.functional as F
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from math import pi, sqrt, exp, radians, cos, sin
import matplotlib.pyplot as plt


def _rect_inter_inner(x1, x2):
    n1 = x1.shape[0] - 1
    n2 = x2.shape[0] - 1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4
  

def _rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj


def intersection(x1, y1, x2, y2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.Inf

    T = T.astype(np.float32)  #####

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
            T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    return xy0[:, 0], xy0[:, 1]


def bresenham(x0, y0, x1, y1):
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

    D = 2 * dy - dx
    y = 0

    for x in range(int(dx) + 1):
        yield x0 + x * xx + y * yx, y0 + x * xy + y * yy
        if D >= 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy


def FillBdryGaps(inBdry):
    n = inBdry.shape[0]
    outBdry = np.empty(shape=[0, 2])
    for i in range(n - 1):
        if abs(inBdry[i + 1, 0] - inBdry[i, 0]) > 1 or abs(inBdry[i + 1, 1] - inBdry[i, 1]) > 1:
            rc = np.array(list(bresenham(inBdry[i, 0], inBdry[i, 1], inBdry[i + 1, 0], inBdry[i + 1, 1])))
            outBdry = np.append(outBdry, rc[:-1, :], axis=0)
        else:
            temp = inBdry[i, :]
            temp = np.transpose(temp[:, np.newaxis])
            outBdry = np.append(outBdry, temp, axis=0)
    if abs(inBdry[0, 0] - inBdry[n - 1, 0]) > 1 or abs(inBdry[0, 1] - inBdry[n - 1, 1]) > 1:
        rc = np.array(list(bresenham(inBdry[n - 1, 0], inBdry[n - 1, 1], inBdry[0, 0], inBdry[0, 1])))
        outBdry = np.append(outBdry, rc[:-1, :], axis=0)
    else:
        temp = inBdry[n - 1, :]
        temp = np.transpose(temp[:, np.newaxis])
        outBdry = np.append(outBdry, temp, axis=0)
    return outBdry


def gauss(n=17, sigma=2):
    r = range(-int(n / 2), int(n / 2) + 1)
    return [1 / (sigma * sqrt(2 * pi)) * exp(-float(x) ** 2 / (2 * sigma ** 2)) for x in r]
  

def Bdry2RadialArcCords(mask, radialN, arcN):
    # extract the contour
    mask_lumen, mask_media = np.zeros_like(mask).astype(np.uint8), np.zeros_like(mask).astype(np.uint8)
    mask_lumen[mask == 2], mask_media[mask != 0] = 255, 255
    pointslumen, _ = cv2.findContours(mask_lumen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pointsmedia, _ = cv2.findContours(mask_media, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    index_lumen = [i.shape[0] for i in pointslumen]
    index_lumen = index_lumen.index(max(index_lumen))
    index_media = [i.shape[0] for i in pointsmedia]
    index_media = index_media.index(max(index_media))
    pointslumen, pointsmedia = pointslumen[index_lumen].squeeze(), pointsmedia[index_media].squeeze()
    # 1: fill the gaps
    lumen_bdry = FillBdryGaps(pointslumen)
    media_bdry = FillBdryGaps(pointsmedia)
    # 2: smooth the contour
    f = gauss(17, 1)
    lumen_bdry = np.concatenate((lumen_bdry[-64:, :], lumen_bdry, lumen_bdry[:64, :]))
    lumen_bdry[:, 0] = np.convolve(lumen_bdry[:, 0], f, 'same')
    lumen_bdry[:, 1] = np.convolve(lumen_bdry[:, 1], f, 'same')
    lumen_bdry = lumen_bdry[64:-63, :]
    media_bdry = np.concatenate((media_bdry[-64:, :], media_bdry, media_bdry[:64, :]))
    media_bdry[:, 0] = np.convolve(media_bdry[:, 0], f, 'same')
    media_bdry[:, 1] = np.convolve(media_bdry[:, 1], f, 'same')
    media_bdry = media_bdry[64:-63, :]
    # 3: set the mess center as the sample center
    # ----------------------- lumen part sampling -------------------------
    # 1: translate according to the center
    lumen_bdry[:, 0] = lumen_bdry[:, 0] - 255.5
    lumen_bdry[:, 1] = lumen_bdry[:, 1] - 255.5

    # 2: first sample the landmarks
    sampled_thetas = np.linspace(-pi, pi - 2 * pi / radialN, radialN)
    intersected_idx = np.zeros((len(sampled_thetas), 2))
    intersection_pts = np.zeros((len(sampled_thetas), 2))
    split_lumen_bdry = []
    for k in range(len(sampled_thetas)):
        x0, y0 = intersection([0, 742 * cos(sampled_thetas[k])],
                              [0, 742 * sin(sampled_thetas[k])],
                              np.concatenate((lumen_bdry[:, 1], np.expand_dims(lumen_bdry[0, 1], axis=0)), axis=0),
                              np.concatenate((lumen_bdry[:, 0], np.expand_dims(lumen_bdry[0, 0], axis=0)), axis=0)
                              )
        if len(x0) > 1:
            min_ind = np.argmin(x0 ** 2 + y0 ** 2)
            x0 = x0[min_ind]
            y0 = y0[min_ind]

        tmp_dist = (lumen_bdry[:, 1] - x0) ** 2 + (lumen_bdry[:, 0] - y0) ** 2
        sorted_idx = np.argsort(tmp_dist)
        sorted_idx = sorted_idx[:2]
        sorted_idx = np.sort(sorted_idx)
        intersection_pts[k, :] = np.array([y0.item(), x0.item()])
        intersected_idx[k, :] = sorted_idx
    intersection_pts = np.concatenate((intersection_pts, np.expand_dims(intersection_pts[0, :], axis=0)), axis=0)
    intersected_idx = np.concatenate((intersected_idx, np.expand_dims(intersected_idx[0, :], axis=0)), axis=0)
    # 3: split the contour into N sections
    for k in range(len(sampled_thetas)):
        if intersected_idx[k, 0] == 0 and intersected_idx[k, 1] == (lumen_bdry.shape[0] - 1):
            intersected_idx[k, 0], intersected_idx[k, 1] = (lumen_bdry.shape[0] - 1), 0
            if k == 0:
                intersected_idx[len(sampled_thetas), 0], intersected_idx[len(sampled_thetas), 1] = (lumen_bdry.shape[
                                                                                                        0] - 1), 0
        if intersected_idx[k, 1] > intersected_idx[k + 1, 0]:
            temp = np.concatenate((lumen_bdry[int(intersected_idx[k, 1]):, :],
                                   lumen_bdry[:int(intersected_idx[k + 1, 0] + 1), :]), axis=0)
            if intersected_idx[k, 1] == intersected_idx[k + 1, 1]:
                temp = np.empty_like(intersection_pts[np.newaxis, k, :])
        else:
            temp = lumen_bdry[int(intersected_idx[k, 1]):int(intersected_idx[k + 1, 0] + 1), :]
        temp = np.concatenate([intersection_pts[np.newaxis, k, :], temp, intersection_pts[np.newaxis, k + 1, :]],
                              axis=0)
        if intersected_idx[k, 1] == intersected_idx[k + 1, 1]:
            temp = np.concatenate([intersection_pts[np.newaxis, k, :], intersection_pts[np.newaxis, k + 1, :]],
                                  axis=0)
        split_lumen_bdry.append(temp)
    # 4: equal-arc-length sample the contour sections
    lumen_sig = np.empty((0, 2))
    for k in range(len(split_lumen_bdry)):
        bdryk = split_lumen_bdry[k]

        arcs = np.sqrt(np.sum((bdryk[1:, :] - bdryk[:-1, :]) ** 2, axis=1))
        arcs = np.cumsum(arcs)
        total_arc = arcs[-1]
        arcs = np.insert(arcs, 0, 0)

        arcs, IA = np.unique(arcs, return_index=True)
        bdryk = bdryk[IA, :]

        xi = [i * total_arc / arcN for i in range(arcN + 1)]
        xi = np.array(xi, dtype=np.float32)
        set_interp = interp1d(arcs, bdryk, kind='linear', axis=0)
        sigk = set_interp(xi)

        lumen_sig = np.concatenate((lumen_sig, sigk[:-1, :]), axis=0)
    # ----------------------- media part sampling -------------------------
    # 1: translate according to the center
    media_bdry[:, 0] = media_bdry[:, 0] - 255.5
    media_bdry[:, 1] = media_bdry[:, 1] - 255.5
    # 2: first sample the landmarks
    sampled_thetas = np.linspace(-pi, pi - 2 * pi / radialN, radialN)
    intersected_idx = np.zeros((len(sampled_thetas), 2))
    intersection_pts = np.zeros((len(sampled_thetas), 2))
    split_media_bdry = []
    for k in range(len(sampled_thetas)):
        x0, y0 = intersection([0, 742 * cos(sampled_thetas[k])],
                              [0, 742 * sin(sampled_thetas[k])],
                              np.concatenate((media_bdry[:, 1], np.expand_dims(media_bdry[0, 1], axis=0)), axis=0),
                              np.concatenate((media_bdry[:, 0], np.expand_dims(media_bdry[0, 0], axis=0)), axis=0)
                              )
        if len(x0) > 1:
            min_ind = np.argmin(x0 ** 2 + y0 ** 2)
            x0 = x0[min_ind]
            y0 = y0[min_ind]

        tmp_dist = (media_bdry[:, 1] - x0) ** 2 + (media_bdry[:, 0] - y0) ** 2
        sorted_idx = np.argsort(tmp_dist)
        sorted_idx = sorted_idx[:2]
        sorted_idx = np.sort(sorted_idx)
        intersection_pts[k, :] = np.array([y0.item(), x0.item()])
        intersected_idx[k, :] = sorted_idx
    intersection_pts = np.concatenate((intersection_pts, np.expand_dims(intersection_pts[0, :], axis=0)), axis=0)
    intersected_idx = np.concatenate((intersected_idx, np.expand_dims(intersected_idx[0, :], axis=0)), axis=0)
    # 3: split the contour into N sections
    for k in range(len(sampled_thetas)):
        if intersected_idx[k, 0] == 0 and intersected_idx[k, 1] == (media_bdry.shape[0] - 1):
            intersected_idx[k, 0], intersected_idx[k, 1] = (media_bdry.shape[0] - 1), 0
            if k == 0:
                intersected_idx[len(sampled_thetas), 0], intersected_idx[len(sampled_thetas), 1] = (media_bdry.shape[
                                                                                                        0] - 1), 0
        if intersected_idx[k, 1] > intersected_idx[k + 1, 0]:
            temp = np.concatenate((media_bdry[int(intersected_idx[k, 1]):, :],
                                   media_bdry[:int(intersected_idx[k + 1, 0] + 1), :]), axis=0)
            if intersected_idx[k, 1] == intersected_idx[k + 1, 1]:
                temp = np.empty_like(intersection_pts[np.newaxis, k, :])
        else:
            temp = media_bdry[int(intersected_idx[k, 1]):int(intersected_idx[k + 1, 0] + 1), :]
        temp = np.concatenate([intersection_pts[np.newaxis, k, :], temp, intersection_pts[np.newaxis, k + 1, :]],
                              axis=0)
        if intersected_idx[k, 1] == intersected_idx[k + 1, 1]:
            temp = np.concatenate([intersection_pts[np.newaxis, k, :], intersection_pts[np.newaxis, k + 1, :]],
                                  axis=0)
        split_media_bdry.append(temp)
    # 4: equal-arc-length sample the contour sections
    media_sig = np.empty((0, 2))
    for k in range(len(split_media_bdry)):
        bdryk = split_media_bdry[k]

        arcs = np.sqrt(np.sum((bdryk[1:, :] - bdryk[:-1, :]) ** 2, axis=1))
        arcs = np.cumsum(arcs)
        total_arc = arcs[-1]
        arcs = np.insert(arcs, 0, 0)

        arcs, IA = np.unique(arcs, return_index=True)
        bdryk = bdryk[IA, :]

        xi = [i * total_arc / arcN for i in range(arcN + 1)]
        xi = np.array(xi, dtype=np.float32)
        set_interp = interp1d(arcs, bdryk, kind='linear', axis=0)
        sigk = set_interp(xi)

        media_sig = np.concatenate((media_sig, sigk[:-1, :]), axis=0)
    return lumen_sig, media_sig


class DataEnhance(nn.Module):
    def __init__(self, degrees=[0, 0], scale=[1, 1]):
        super(DataEnhance, self).__init__()

        self.degrees = degrees
        self.scale = scale

    def forward(self, imagedata, maskdata):
        a = random.randint(6, 14) / 10
        b = random.randrange(0, 40, 10) - 20
        image = np.uint8(np.clip((a * imagedata + b), 0, 255))

        channels = image.shape[0]
        rows = image.shape[1]
        cols = image.shape[2]

        angle = np.random.random() * (self.degrees[1] - self.degrees[0]) + self.degrees[0]
        scale = np.random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        matrix1 = cv2.getRotationMatrix2D(((rows-1) / 2, (cols-1) / 2), angle, scale)

        for j in range(channels):
            temp = image[j].copy()
            temp = cv2.warpAffine(temp, matrix1, (rows, cols))
            image[j] = temp

        tempmask = maskdata.copy()
        tempmask = cv2.warpAffine(tempmask, matrix1, (rows, cols))
        mask = tempmask

        hudu = radians(angle)
        R = np.array([[cos(hudu), sin(hudu)],[sin(hudu), cos(hudu)]])
        
        p = random.random()
        code = random.randint(0, 2) - 1
        if p < 0.5:
            for j in range(channels):
                image[j] = cv2.flip(image[j], code)
            mask = cv2.flip(mask, code)
        else:
            image = image
            mask = mask

        # extract the contours
        lumen_sig, media_sig = Bdry2RadialArcCords(mask, 16, 32)

        return image, mask, lumen_sig, media_sig


class DatasetGenerator(Dataset):
    def __init__ (self, pathimagedata, pathmask, if_train=False):
        self.listMaskpath = []
        self.listdatapath = []
        self.if_train = if_train
        self.enhancer = DataEnhance(degrees=[-30, 30],  scale=[0.9, 1.1])
        self.listmiou = []

        allmasklist = sorted(os.listdir(pathmask))
        for maskname in allmasklist:
            maskpath = os.path.join(pathmask, maskname)
            datapath = os.path.join(pathimagedata, maskname[:-8]+'image.npy')
            if self.if_train:
                miou = np.load(os.path.join('./trainiou/'+maskname[:-8]+'iou.npy')).item()
            else:
                miou = 0
            self.listMaskpath.append(maskpath)
            self.listdatapath.append(datapath)
            self.listmiou.append(miou)
        self.prob = np.ones(len(self.listmiou))

    def set_prob(self, alpha=0):
        temp = np.maximum(1 - alpha * np.copy(self.listmiou), 1e-8)
        self.prob = temp
        
    def __getitem__(self, index):
        imagepath = self.listdatapath[index]
        imgs = np.load(imagepath)
        maskpath = self.listMaskpath[index]
        mask = np.load(maskpath)
        mask = mask.astype(np.uint8)
        miou_prob = self.prob[index]

        if self.if_train == False:
            imagedata, imagemask = imgs, mask
            lumen_sig, media_sig = Bdry2RadialArcCords(mask, 16, 32)
        else:
            imagedata, imagemask, lumen_sig, media_sig = self.enhancer(imgs, mask)

        imagedata=imagedata.astype(np.float32)
        for i in range(3):
            imagedata[i] -= np.mean(imagedata[i])
            imagedata[i] /= np.std(imagedata[i])

        return torch.from_numpy(imagedata.copy()), torch.from_numpy(imagemask.copy()).long(), \
            torch.from_numpy(lumen_sig.copy()), torch.from_numpy(media_sig.copy()), miou_prob
    
    def __len__(self):
        return len(self.listMaskpath)
