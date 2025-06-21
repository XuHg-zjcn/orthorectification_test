#!/usr/bin/env python3
#############################################################################
# 遥感影像预处理序列
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
import inspect
from abc import ABC, abstractmethod
from preprocess import *


class PreprocessSeq(ABC):
    def __new__(cls, **kwargs):
        return cls.params_from_dict(kwargs)

    @classmethod
    def params_from_dict(cls, d, raise_unknown=True):
        obj = super().__new__(cls)
        signature = inspect.signature(obj._build)
        param_names = signature.parameters.keys()
        d_param = {}
        for key, value in d.items():
            if key in param_names:
                d_param[key] = value
            elif key[0] != 'A' and key[0] != 'B':
                A_key = 'A_' + key
                B_key = 'B_' + key
                if A_key in param_names:
                    d_param[A_key] = value
                if B_key in param_names:
                    d_param[B_key] = value
                if raise_unknown and A_key not in param_names and B_key not in param_names:
                    raise TypeError(f'unknown keyword {key}')
            elif raise_unknown:
                raise TypeError(f'unknown keyword {key}')
        obj._build(**d_param)
        return obj

    @classmethod
    def params_from_3dict(cls, d_A, d_B, d_com, raise_unknown=True):
        obj = super().__new__(cls)
        signature = inspect.signature(obj._build)
        param_names = signature.parameters.keys()
        d_param = {}
        for key, value in d_A.items():
            A_key = 'A_' + key
            if A_key in param_names:
                d_param[A_key] = value
            elif raise_unknown:
                raise TypeError(f'unknown keyword {key} in d_A')
        for key, value in d_B.items():
            B_key = 'B_' + key
            if B_key in param_names:
                d_param[B_key] = value
            elif raise_unknown:
                raise TypeError(f'unknown keyword {key} in d_B')
        for key, value in d_com.items():
            if key in param_names:
                d_param[key] = value
            elif raise_unknown:
                raise TypeError(f'unknown keyword {key} in d_com')
        obj._build(**d_param)
        return obj

    def process(self, dict_):
        for pobj in self.lst:
            dict_ = pobj.process(dict_)
        return dict_

    @abstractmethod
    def _build(self, **kwargs):
        pass


# 在`match_db.py`使用
class PreprocessWithEst(PreprocessSeq):
    def _build(self,
            A_extCoef=0.1, A_extMin=10, A_nX=8, A_maxpixel=1e6, A_predown=0,
            B_extCoef=0.1, B_extMin=10, B_nX=8, B_maxpixel=1e6, B_predown=0,
            A_cutblack_topbottom=True, A_cutblack_leftright=True,
            B_cutblack_topbottom=True, B_cutblack_leftright=True,
            w_Laplace=1.0, w_Roberts=1.414, w_Sobel=0.53):
        lst = []
        # 预先裁剪和缩放
        lst.append(AutoCutEstTf('A', extCoef=A_extCoef, extMin=A_extMin*A_nX))
        lst.append(AutoCutEstTf('B', extCoef=B_extCoef, extMin=B_extMin*B_nX))
        lst.append(AutoZoom('A', maxpixel=A_maxpixel*A_nX**2, predown=A_predown))  # 此处会转成numpy数组
        lst.append(AutoZoom('B', maxpixel=B_maxpixel*B_nX**2, predown=B_predown))
        lst.append(AutoZoomEstTf('A', nX=A_nX/B_nX))  # 此处nX是保留倍率
        lst.append(AutoZoomEstTf('B', nX=B_nX/A_nX))
        # TODO: 先估计缩放倍率再转换成numpy数组
        # 处理A
        if A_cutblack_topbottom:
            lst.append(CutBlackTopBottom('A'))
        if A_cutblack_leftright:
            lst.append(CutBlackLeftRight('A'))
        lst.append(EdgeDetection('A', w_Laplace=w_Laplace, w_Roberts=w_Roberts, w_Sobel=w_Sobel))
        lst.append(DilateAndDownsamp('A', nz=A_nX))
        # 处理B
        if B_cutblack_topbottom:
            lst.append(CutBlackTopBottom('B'))
        if B_cutblack_leftright:
            lst.append(CutBlackLeftRight('B'))
        lst.append(EdgeDetection('B', w_Laplace=w_Laplace, w_Roberts=w_Roberts, w_Sobel=w_Sobel))
        lst.append(DilateAndDownsamp('B', nz=B_nX))
        lst.append(AutoCutEstTf('A', extCoef=A_extCoef, extMin=A_extMin))
        lst.append(AutoCutEstTf('B', extCoef=B_extCoef, extMin=B_extMin))
        self.lst = lst


# 在`match_imgpair.py`使用
class PreprocessNoEst(PreprocessSeq):
    def _build(self,
            A_laplace=False, A_dilsize=8, A_maxpixel=1e7, A_predown=0,
            B_laplace=False, B_dilsize=8, B_maxpixel=1e7, B_predown=0,
            A_cutblack_topbottom=False, A_cutblack_leftright=False,
            B_cutblack_topbottom=False, B_cutblack_leftright=False):
        lst = []
        if A_laplace:
            A_maxpixel *= A_dilsize**2
        lst.append(AutoZoom('A', maxpixel=A_maxpixel, predown=A_predown))
        if A_cutblack_topbottom:
            lst.append(CutBlackTopBottom('A'))
        if A_cutblack_leftright:
            lst.append(CutBlackLeftRight('A'))
        if A_laplace:
            lst.append(LaplacianAndDilate('A', nz=A_dilsize))

        if B_laplace:
            B_maxpixel *= B_dilsize**2
        lst.append(AutoZoom('B', maxpixel=B_maxpixel, predown=B_predown))
        if B_cutblack_topbottom:
            lst.append(CutBlackTopBottom('B'))
        if B_cutblack_leftright:
            lst.append(CutBlackLeftRight('B'))
        if B_laplace:
            lst.append(LaplacianAndDilate('B', nz=B_dilsize))
        self.lst = lst


# `match_levels.py`中使用
class PreprocessLevels(PreprocessSeq):
    def _build(self,
            A_extCoef=0.1, A_extMin=10, B_nX=8,
            w_Laplace=1.0, w_Roberts=1.414, w_Sobel=0.53):
        lst = []
        # 预先裁剪和缩放
        lst.append(AutoCutEstTf('A', extCoef=A_extCoef, extMin=A_extMin))
        lst.append(AutoZoomEstTf('B', nX=B_nX))
        lst.append(AutoZoom('B', predown=1))
        lst.append(EdgeDetection('B', w_Laplace=w_Laplace, w_Roberts=w_Roberts, w_Sobel=w_Sobel))
        lst.append(DilateAndDownsamp('B', nz=8))
        lst.append(CutBlackTopBottom('B'))
        lst.append(CutBlackLeftRight('B'))
        self.lst = lst


# `match_part.py`中使用
class PreprocessPart(PreprocessSeq):
    def _build(self, A_cut=None, B_cut=None):
        lst = []
        # 预先裁剪和缩放
        lst.append(CutImg('A', A_cut))
        lst.append(AutoZoom('A', predown=1))
        lst.append(CutImg('B', B_cut))
        lst.append(AutoZoom('B', predown=1))
        self.lst = lst