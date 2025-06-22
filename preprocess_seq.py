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
    def get_param_names(cls):
        signature = inspect.signature(cls._build)
        param_names = set(signature.parameters.keys())
        cls_ = cls.__base__
        while True:  # 获取所有基类的`_build`方法签名
            if hasattr(cls_, '_build'):
                signature = inspect.signature(cls_._build)
                param_names.update(set(signature.parameters.keys()))
            else:
                break
            cls_ = cls_.__base__
        param_names.discard('self')
        return param_names

    @classmethod
    def params_from_dict(cls, d, raise_unknown=True):
        obj = super().__new__(cls)
        param_names = cls.get_param_names()
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
        param_names = cls.get_param_names()
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
    def _build(self):
        pass


# 在`match_imgpair.py`使用
class PreprocessNoEst(PreprocessSeq):
    def _build(self,
            A_detEdge=False, A_dilsize=8, A_maxpixel=1e7, A_predown=0,
            B_detEdge=False, B_dilsize=8, B_maxpixel=1e7, B_predown=0,
            A_cutblack_topbottom=True, A_cutblack_leftright=True,
            B_cutblack_topbottom=True, B_cutblack_leftright=True,
            w_Laplace=1.0, w_Roberts=1.414, w_Sobel=0.53):
        lst = []
        if A_detEdge:
            A_maxpixel *= A_dilsize**2
        if B_detEdge:
            B_maxpixel *= B_dilsize**2
        lst.append(AutoZoom('A', maxpixel=A_maxpixel, predown=A_predown))
        if A_cutblack_topbottom:
            lst.append(CutBlackTopBottom('A'))
        if A_cutblack_leftright:
            lst.append(CutBlackLeftRight('A'))
        if A_detEdge:
            lst.append(EdgeDetection('A', w_Laplace=w_Laplace, w_Roberts=w_Roberts, w_Sobel=w_Sobel))
            lst.append(DilateAndDownsamp('A', nz=A_dilsize))

        lst.append(AutoZoom('B', maxpixel=B_maxpixel, predown=B_predown))
        if B_cutblack_topbottom:
            lst.append(CutBlackTopBottom('B'))
        if B_cutblack_leftright:
            lst.append(CutBlackLeftRight('B'))
        if B_detEdge:
            lst.append(EdgeDetection('B', w_Laplace=w_Laplace, w_Roberts=w_Roberts, w_Sobel=w_Sobel))
            lst.append(DilateAndDownsamp('B', nz=B_dilsize))
        if hasattr(self, 'lst'):
            self.lst.extend(lst)
        else:
            self.lst = lst


# 在`match_db.py`使用
class PreprocessWithEst(PreprocessNoEst):
    def _build(self,
        A_extCoef=0.1, A_extMin=10, A_detEdge=True, A_dilsize=8, A_maxpixel=1e6, A_predown=0,
        B_extCoef=0.1, B_extMin=10, B_detEdge=True, B_dilsize=8, B_maxpixel=1e6, B_predown=0, **kwargs):
        if A_detEdge:
            A_extMin *= A_dilsize
            A_maxpixel *= A_dilsize**2
        if B_detEdge:
            B_extMin *= B_dilsize
            B_maxpixel *= B_dilsize**2
        lst = []
        # 预先裁剪和缩放
        lst.append(AutoCutEstTf('A', extCoef=A_extCoef, extMin=A_extMin))
        lst.append(AutoCutEstTf('B', extCoef=B_extCoef, extMin=B_extMin))
        lst.append(AutoZoom('A', maxpixel=A_maxpixel, predown=A_predown))  # 此处会转成numpy数组
        lst.append(AutoZoom('B', maxpixel=B_maxpixel, predown=B_predown))
        lst.append(AutoZoomEstTf('A', nX=A_dilsize/B_dilsize))  # 此处nX是保留倍率
        lst.append(AutoZoomEstTf('B', nX=B_dilsize/A_dilsize))
        # TODO: 先估计缩放倍率再转换成numpy数组
        self.lst = lst
        super()._build(A_detEdge=A_detEdge, A_dilsize=A_dilsize, A_predown=1,
                       B_detEdge=B_detEdge, B_dilsize=B_dilsize, B_predown=1, **kwargs)


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