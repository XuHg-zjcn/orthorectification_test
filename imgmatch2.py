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
import shapely
from imgview import ImgView
from imgmatch import compare
from common import CropZoom2D
from transform import KeepTransform, PerspectiveTransform


class ImgMatch:
    def __init__(self, imgA, imgB):
        self.imgA = imgA  # imgA, imgB可以是np.ndarray或ImgView类型
        self.imgB = imgB
        self.pseq = []    # 预处理序列(PreprocessSeq子类的对象)
        self.cp = None    # 对比参数(Compare Param)
        self.estT = None
        self.H = None
        self.n_match = 0

    def append_pobj(self, x):
        self.pol.append(x)

    def setPreprocessSeq(self, pseq):
        self.pseq = pseq

    def set_estT(self, estT):
        self.estT = estT

    def setParam_compare(self, *args, **kwargs):
        self.cp = (args, kwargs)

    def match(self):
        # TODO: 暂存预处理后的数据
        # TODO: 不应该用字典作为预处理参数，而是self对象
        dict_ = {'imgA':self.imgA,
                 'imgB':self.imgB,
                 'tA':KeepTransform(),
                 'tB':KeepTransform(),
                 'estT':self.estT}
        dict_ = self.pseq.process(dict_)
        imgA_ = dict_['imgA']
        imgB_ = dict_['imgB']
        tA = dict_['tA']
        tB = dict_['tB']
        H_, matchs = compare(imgA_, imgB_, *self.cp[0], **self.cp[1])
        if H_ is None:
            return None, 0
        H_ = PerspectiveTransform(H_)
        H = tA.fog(H_).fog(tB.inv())
        self.H = H
        self.n_match = len(matchs)
        # TODO: 更进一步，使用高分辨率的图像分块处理，进行更高精度的对齐
        return H, len(matchs)

    def get_poly_B_in_A(self):
        # TODO: 应该取图像的有效区域
        height_B, width_B = self.imgB.shape
        box_B = shapely.box(0, 0, width_B, height_B)
        return self.H(box_B)

    def get_poly_A_in_B(self):
        height_A, width_A = self.imgA.shape
        box_A = shapely.box(0, 0, width_A, height_A)
        return self.H.inv()(box_A)