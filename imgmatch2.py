#!/usr/bin/env python3
#############################################################################
# 遥感影像对匹配（封装成类）
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
import numpy as np
from preprocess_single import preprocess
from imgview import ImgView
from imgmatch import compare, H_transpose
from common import CropZoom2D


class ImgMatch:
    def __init__(self, imgA, imgB):
        self.imgA = imgA  # imgA, imgB可以是np.ndarray或ImgView类型
        self.imgB = imgB
        self.ppA = None   # 预处理参数(Preprocess Param)
        self.ppB = None
        self.cp = None    # 对比参数(Compare Param)
        self.cutA = None  # 估计区域
        self.cutB = None

    def set_cutA(self, cutA):
        self.cutA = cutA

    def set_cutB(self, cutB):
        self.cutB = cutB

    def setParam_preprocessA_empty(self):
        self.ppA = ([], {})

    def setParam_preprocessB_empty(self):
        self.ppB = ([], {})

    def setParam_preprocessA(self, *args, **kwargs):
        self.ppA = (args, kwargs)

    def setParam_preprocessB(self, *args, **kwargs):
        self.ppB = (args, kwargs)

    def setParam_compare(self, *args, **kwargs):
        self.cp = (args, kwargs)

    def match(self):
        def preprocess_and_transfrom(imgX, cutX, ppX, name):
            cutX = cutX if cutX is not None else CropZoom2D.with_shape(imgX.shape)
            if isinstance(imgX, ImgView):
                imgX = imgX[cutX]
            elif isinstance(imgX, np.ndarray):
                imgX = imgX[cutX.to_slice()]
            imgX_, nX, xyX = preprocess(imgX, name, *ppX[0], **ppX[1])
            czpX = CropZoom2D(x0=xyX[0], y0=xyX[1], nz=nX, wo=imgX_.shape[1], ho=imgX_.shape[0])
            czoX = cutX.fog(czpX)
            return imgX_, czoX
        # TODO: 暂存预处理后的数据
        imgA_, czoA = preprocess_and_transfrom(self.imgA, self.cutA, self.ppA, 'imgA')
        imgB_, czoB = preprocess_and_transfrom(self.imgB, self.cutB, self.ppB, 'imgB')
        H_, matchs = compare(imgA_, imgB_, *self.cp[0], **self.cp[1])
        if H_ is None:
            return None, 0
        H = H_transpose(
            H_,
            x0_d=czoA.x0, y0_d=czoA.y0, zoom_d=czoA.nz,
            x0_s=czoB.x0, y0_s=czoB.y0, zoom_s=czoB.nz
        )
        # TODO: 更进一步，使用高分辨率的图像分块处理，进行更高精度的对齐
        # TODO: 在此处生成B_in_A的shapely.Polygon对象
        return H, len(matchs)

    def match_with_estH(self, H_est):
        # TODO: 实现此方法
        pass