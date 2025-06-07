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
from preprocess_single import preprocess
from imgmatch import compare, H_transpose
from common import CropZoom2D

class ImgMatch:
    def __init__(self, imgA, imgB):
        self.imgA = imgA
        self.imgB = imgB
        self.ppA = None
        self.ppB = None
        self.cp = None
        self.estA = None
        self.estB = None

    def set_estA(self, estA):
        self.estA = estA

    def set_estB(self, estB):
        self.estB = estB

    def setParam_preprocessA(self, *args, **kwargs):
        self.ppA = (args, kwargs)

    def setParam_preprocessB(self, *args, **kwargs):
        self.ppB = (args, kwargs)

    def setParam_compare(self, *args, **kwargs):
        self.cp = (args, kwargs)

    def match(self):
        # TODO: 暂存预处理后的数据
        imgA = self.imgA
        imgB = self.imgB
        estA = self.estA if self.estA is not None else CropZoom2D.with_shape(self.imgA.shape)
        estB = self.estB if self.estB is not None else CropZoom2D.with_shape(self.imgB.shape)
        imgA = imgA[estA]
        imgB = imgB[estB]
        imgA_, nA, xyA = preprocess(imgA, 'imgA', *self.ppA[0], **self.ppA[1])
        imgB_, nB, xyB = preprocess(imgB, 'imgB', *self.ppB[0], **self.ppB[1])
        H_, matchs = compare(imgA_, imgB_, *self.cp[0], **self.cp[1])
        if H_ is None:
            return None, 0
        czpA = CropZoom2D(x0=xyA[0], y0=xyA[1], nz=nA, wo=imgA_.shape[1], ho=imgA_.shape[0])
        czpB = CropZoom2D(x0=xyB[0], y0=xyB[1], nz=nB, wo=imgB_.shape[1], ho=imgB_.shape[0])
        czoA = estA.fog(czpA)
        czoB = estB.fog(czpB)
        H = H_transpose(
            H_,
            x0_d=czoA.x0, y0_d=czoA.y0, zoom_d=czoA.nz,
            x0_s=czoB.x0, y0_s=czoB.y0, zoom_s=czoB.nz
        )
        # TODO: 更进一步，使用高分辨率的图像分块处理，进行更高精度的对齐
        return H, len(matchs)