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
import numpy as np
import cv2
from imgview import ImgView


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
def auto_zoom(img, maxpixel=1e6, predown=None):
    if predown is not None and predown != 0:
        n = predown
    elif img.shape[0]*img.shape[1] <= maxpixel:
        n = 1
    else:
        n = math.ceil(math.sqrt(img.shape[0]*img.shape[1]/maxpixel))
    if isinstance(img, ImgView):
        img = img[::n, ::n].trim_scale()
        n = img.scale
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

def laplacian_and_dilate(img, nz):
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE if nz >= 5 else cv2.MORPH_RECT, (nz, nz))
    img = cv2.Laplacian(img, cv2.CV_8U)
    img = cv2.dilate(img, kern)
    img = cv2.resize(img, None, None, 1.0/nz, 1.0/nz, cv2.INTER_AREA)
    return img

def preprocess(img, name='',
               maxpixel_out=None, predown=None,
               laplace=False, dilsize=8,
               cutblack_topbottom=False, cutblack_leftright=False):
    ndown = 1
    x0 = 0
    y0 = 0
    if  predown is not None and predown != 0:  # 直接指定了下采样倍率
        img, n = auto_zoom(img, None, predown)
        ndown *= n
    elif maxpixel_out is not None:             # 没有指定倍率，但指定了SIFT最大分辨率
        if laplace:
            maxpix1 = maxpixel_out*dilsize**2
        else:
            maxpix1 = maxpixel_out
        img, n = auto_zoom(img, maxpix1, predown)
        ndown *= n
    elif isinstance(img, ImgView):
        img = img.get_array()
    shape_down = img.shape
    if cutblack_topbottom:
        t, b = detect_edge_black(img, axis=0)
        img = img[t:b+1]
        y0 += t
        percent = (shape_down[0]-(b-t+1))/shape_down[0]
        print(f'{name} cropoff black edge top-bottom in x:[{t},{b+1}), drop {percent:.2%}')
    if cutblack_leftright:
        l, r = detect_edge_black(img, axis=1)
        img = img[:, l:r+1]
        x0 += l
        percent = (shape_down[1]-(r-l+1))/shape_down[1]
        print(f'{name} cropoff black edge left-right in y:[{l},{r+1}), drop {percent:.2%}')
    if laplace:
        img = laplacian_and_dilate(img, dilsize)
        ndown *= dilsize
    return img, ndown, (x0, y0)
