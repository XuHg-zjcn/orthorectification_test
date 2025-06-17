#!/usr/bin/env python3
#############################################################################
# 遥感影像预处理程序
# Copyright (C) 2025  Xu Ruijun
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#############################################################################
import math
from abc import ABC, abstractmethod
import numpy as np
import cv2
import shapely
from imgview import ImgView
from transform import KeepTransform, MoveZoomTransform
from common import CropZoom2D


class Preprocess:
    def process(self, dict_):
        return dict_


class PreprocessSingle(Preprocess):
    def __init__(self, imgname):
        self.imgname = imgname

    def process(self, dict_):
        if self.imgname not in {'A', 'B'}:
            raise ValueError(f'invaild imgname {self.imgname}')
        s_imgX = 'img'+self.imgname
        s_tX = 't'+self.imgname
        imgX, t_ = self.process_img(dict_[s_imgX])
        tX = dict_[s_tX].fog(t_)
        dict_[s_imgX] = imgX
        dict_[s_tX] = tX
        return dict_

    def process_img(self, img):
        return img, KeepTransform()


class PreprocessWithEstTf(Preprocess):
    def __init__(self, imgname):
        self.imgname = imgname

    def process(self, dict_):
        imgA = dict_['imgA']
        imgB = dict_['imgB']
        tA = dict_['tA']
        tB = dict_['tB']
        estT = dict_['estT']
        estT_curr = tA.inv().fog(estT).fog(tB)
        if self.imgname == 'A':  #裁剪当前的imgA
            imgA, t_ = self.process_img(imgA, imgB.shape, estT_curr)
            tA = tA.fog(t_)
            dict_['imgA'] = imgA
            dict_['tA'] = tA
        elif self.imgname == 'B':  #裁剪当前的imgB
            imgB, t_ = self.process_img(imgB, imgA.shape, estT_curr.inv())
            tB = tB.fog(t_)
            dict_['imgB'] = imgB
            dict_['tB'] = tB
        return dict_


# 计算图像B透视投影到图像A中与图像A的交集在图像A的坐标系下的包围框
def perspective_boundingbox(H, widthA, heightA, widthB, heightB):
    conrerB = np.array([[0, 0, 1],
                        [0, heightB, 1],
                        [widthB, heightB, 1],
                        [widthB, 0, 1]])
    conrerB_at_coordA = np.matmul(H, conrerB.transpose())
    conrerB_at_coordA /= conrerB_at_coordA[-1]
    xmin_A = 0
    xMax_A = widthA
    ymin_A = 0
    yMax_A = heightA
    xmin_B = np.min(conrerB_at_coordA[0])
    xMax_B = np.max(conrerB_at_coordA[0])
    ymin_B = np.min(conrerB_at_coordA[1])
    yMax_B = np.max(conrerB_at_coordA[1])
    xmin = max(xmin_A, xmin_B)
    xMax = min(xMax_A, xMax_B)
    ymin = max(ymin_A, ymin_B)
    yMax = min(yMax_A, yMax_B)
    return int(xmin), int(ymin), int(xMax), int(yMax)

# axis=0 for top and bottom for axis=1 for left and right
def detect_edge_black(img, axis, alpha=1.0/3):
    def scan(index_above, min_count, inc):
        begin = index_above[0]
        i = 0
        count = 0
        while i+1 < len(index_above) and count < min_count:
            if index_above[i+1] == index_above[i]+inc:
                count += 1
            else:
                count = 1
                begin = index_above[i+1]
            i += 1
        return begin
    axis_other = [i for i in range(img.ndim)]
    axis_other.remove(axis)
    line_mean = np.mean(img, axis=tuple(axis_other))
    threshold = np.median(line_mean)*alpha
    index_above = np.where(line_mean > threshold)[0]
    min_count = line_mean.shape[0]//4
    a = scan(index_above, min_count, 1)
    b = scan(index_above[::-1], min_count, -1)
    return a, b

# 以合适的整数倍缩小图像
def _auto_zoom(img, maxpixel=1e6, predown=None):
    if predown is not None and predown != 0:
        n = predown
    elif img.shape[0]*img.shape[1] <= maxpixel:
        n = 1
    else:
        n = math.ceil(math.sqrt(img.shape[0]*img.shape[1]/maxpixel))
    if isinstance(img, ImgView):
        img = img[::n, ::n].trim_scale()
        img_ = img.get_array()
        return img_, n
    elif isinstance(img, np.ndarray):
        if n != 1:
            img_ = cv2.resize(img, None, None, 1.0/n, 1.0/n, cv2.INTER_AREA)
            return img_, n
        else:
            return img, n
    else:
        raise TypeError(f'unknown type {type(img)}')


class ApplyTransform(PreprocessSingle):
    def __init__(self, imgname, transform):
        super().__init__(imgname)
        self.transform = transform

    def process_img(self, img):
        return img, self.transform


class CutImg(PreprocessSingle):
    def __init__(self, imgname, cz2d):
        super().__init__(imgname)
        self.cz2d = cz2d

    def process_img(self, img):
        img = img[self.cz2d.to_slice()]
        mt = MoveZoomTransform(x0=self.cz2d.x0, y0=self.cz2d.y0, nz=self.cz2d.nz)
        return img, mt


class AutoZoom(PreprocessSingle):
    def __init__(self, imgname, maxpixel=1e6, predown=None):
        super().__init__(imgname)
        self.maxpixel = maxpixel
        self.predown = predown

    def process_img(self, img):
        img, n = _auto_zoom(img, self.maxpixel, self.predown)
        mt = MoveZoomTransform(nz=n)
        return img, mt


class LaplacianAndDilate(PreprocessSingle):
    def __init__(self, imgname, nz=8):
        super().__init__(imgname)
        self.nz = nz

    def process_img(self, img):
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE if self.nz >= 5 else cv2.MORPH_RECT, (self.nz, self.nz))
        img = cv2.Laplacian(img, cv2.CV_8U)
        img = cv2.dilate(img, kern)
        img = cv2.resize(img, None, None, 1.0/self.nz, 1.0/self.nz, cv2.INTER_AREA)
        return img, MoveZoomTransform(nz=self.nz)


class EdgeDetection(PreprocessSingle):
    kern_ra = np.array([[-1, 0], [0, 1]])
    kern_rb = np.array([[0, -1], [1, 0]])
    def __init__(self, imgname,
                 w_Laplace=1,
                 w_Roberts=1.414,
                 w_Sobel=0.53,
                 use_abs=True):
        super().__init__(imgname)
        self.w_Laplace = w_Laplace
        self.w_Roberts = w_Roberts
        self.w_Sobel = w_Sobel
        self.use_abs = use_abs

    def process_img_float(self, img):
        G_all = np.zeros_like(img, dtype=np.float32)
        if self.w_Laplace != 0:
            L = np.abs(cv2.Laplacian(img, cv2.CV_32F))
            G_all += self.w_Laplace*L
        if self.w_Roberts != 0:
            Gra = cv2.filter2D(img, cv2.CV_32F, self.kern_ra)
            Grb = cv2.filter2D(img, cv2.CV_32F, self.kern_rb)
            Gr = cv2.magnitude(Gra, Grb)
            G_all += self.w_Roberts*Gr
        if self.w_Sobel != 0:
            Gsx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
            Gsy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
            Gs = cv2.magnitude(Gsx, Gsy)
            G_all += self.w_Sobel*Gs
        G_all *= 255/(0.7*G_all.max() + 0.3*np.sqrt(np.mean(G_all**2)))
        G_all_u8 = np.clip(G_all, 0, 255).astype(np.uint8)
        return G_all_u8, KeepTransform()

    def process_img_abs(self, img):
        G_all = np.zeros_like(img, dtype=np.uint64)
        if self.w_Laplace != 0:
            L = np.abs(cv2.Laplacian(img, cv2.CV_16S))
            w_Laplace_ = int(round(self.w_Laplace*100))
            G_all += w_Laplace_ * (L.astype(np.uint64))
        if self.w_Roberts != 0:
            Gra = cv2.filter2D(img, cv2.CV_16S, self.kern_ra)
            Grb = cv2.filter2D(img, cv2.CV_16S, self.kern_rb)
            Gr = np.abs(Gra)+np.abs(Grb)
            w_Roberts_ = int(round(self.w_Roberts*100))
            G_all += w_Roberts_ * (Gr.astype(np.uint64))
        if self.w_Sobel != 0:
            Gsx = cv2.Sobel(img, cv2.CV_16S, 1, 0)
            Gsy = cv2.Sobel(img, cv2.CV_16S, 0, 1)
            Gs = np.abs(Gsx)+np.abs(Gsy)
            w_Sobel_ = int(round(self.w_Sobel*100))
            G_all += w_Sobel_ * (Gs.astype(np.uint64))
        scale = (0.7*float(G_all.max()) + 0.3*1.2*float(np.mean(G_all)))/255
        G_all //= int(round(scale))
        G_all_u8 = np.clip(G_all, 0, 255).astype(np.uint8)
        return G_all_u8, KeepTransform()

    def process_img(self, img):
        if self.use_abs:
            return self.process_img_abs(img)
        else:
            return self.process_img_float(img)


class DilateAndDownsamp(PreprocessSingle):
    def __init__(self, imgname, nz):
        super().__init__(imgname)
        self.nz = nz
        self.kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE if self.nz >= 5 else cv2.MORPH_RECT, (self.nz, self.nz))

    def process_img(self, img):
        img = cv2.dilate(img, self.kern)
        img = cv2.resize(img, None, None, 1.0/self.nz, 1.0/self.nz, cv2.INTER_AREA)
        mt = MoveZoomTransform(nz=self.nz)
        return img, mt


class CutBlackTopBottom(PreprocessSingle):
    def process_img(self, img):
        height_in = img.shape[0]
        t, b = detect_edge_black(img, axis=0)
        img = img[t:b+1]
        percent = (height_in-(b-t+1))/height_in
        name = 'img'+self.imgname
        print(f'{name} cropoff black edge top-bottom in x:[{t},{b+1}), drop {percent:.2%}')
        return img, MoveZoomTransform(y0=t)


class CutBlackLeftRight(PreprocessSingle):
    def process_img(self, img):
        width_in = img.shape[1]
        l, r = detect_edge_black(img, axis=1)
        img = img[:, l:r+1]
        percent = (width_in-(r-l+1))/width_in
        name = 'img'+self.imgname
        print(f'{name} cropoff black edge left-right in y:[{l},{r+1}), drop {percent:.2%}')
        return img, MoveZoomTransform(x0=l)


class AutoCutEstTf(PreprocessWithEstTf):
    def __init__(self, imgname, extCoef=0, extMin=0):
        super().__init__(imgname)
        self.extCoef = extCoef  # 扩展系数
        self.extMin = extMin    # 最小扩展宽度

    def process_img(self, img, other_shape, estT):
        other_height, other_width = other_shape
        box_other = shapely.box(0, 0, other_width, other_height)
        box_other_in_img = estT(box_other)
        xmin, ymin, xmax, ymax = box_other_in_img.bounds
        if self.extCoef != 0 or self.extMin != 0:
            xdelta = max(self.extMin, self.extCoef*(xmax-xmin))
            ydelta = max(self.extMin, self.extCoef*(ymax-ymin))
            xmin -= xdelta
            xmax += xdelta
            ymin -= ydelta
            ymax += ydelta
        xmin = int(min(max(0, xmin), img.shape[1]))
        xmax = int(min(max(0, xmax), img.shape[1]))
        ymin = int(min(max(0, ymin), img.shape[0]))
        ymax = int(min(max(0, ymax), img.shape[0]))
        cz2d = CropZoom2D(x0=xmin, y0=ymin, x1=xmax, y1=ymax)
        mt = MoveZoomTransform(x0=xmin, y0=ymin)
        if isinstance(img, ImgView):
            img = img[cz2d]
        else:
            img = img[cz2d.to_slice()]
        return img, mt


class AutoZoomEstTf(PreprocessWithEstTf):
    def __init__(self, imgname, nX=1):
        super().__init__(imgname)
        self.nX = nX  # 为后续变换预留

    def process_img(self, img, other_shape, estT):
        other_height, other_width = other_shape
        if hasattr(estT, 'nz_at'):
            nz_ = estT.nz_at(other_width/2, other_height/2)
        else:
            nz_ = estT.nz
        nz = max(1, int(round(nz_/self.nX)))
        img = img[::nz, ::nz]
        mt = MoveZoomTransform(nz=nz)
        return img, mt
