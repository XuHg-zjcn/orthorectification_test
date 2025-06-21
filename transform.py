#!/usr/bin/env python3
#############################################################################
# 图像坐标变换
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
from abc import ABC, abstractmethod
from collections.abc import Sized
from functools import cached_property
import numpy as np
from PIL import Image
import cv2
import shapely


class Transform(ABC):
    compatibleLst = []
    @classmethod
    def isCompatible(cls, T):   # T类能被转换成当前类
        if issubclass(cls, T):  # 是当前类的父类
            return True
        if T == KeepTransform:
            return True
        for c in cls.compatibleLst:
            if c.isCompatible(T):
                return True
        return False

    @abstractmethod
    def apply(self, img, shape):
        # 对图像进行变换
        pass

    @abstractmethod
    def transpoints(self, points):
        pass

    def __call__(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.shape == (2,):
                return self.transpoints(obj.reshape((1,2)))[0]
            elif obj.ndim == 2 and obj.shape[1] == 2:
                return self.transpoints(obj)
        elif isinstance(obj, tuple) and len(obj) == 2:
            return self.transpoints(np.array([obj]))[0]
        elif isinstance(obj, shapely.geometry.base.BaseGeometry):
            return shapely.transform(obj, self.transpoints)
        else:
            raise TypeError(f'unsupport type {type(obj)}')

    @abstractmethod
    def inv(self):
        # 求逆变换
        pass

    @abstractmethod
    def keep(cls):
        pass

    @abstractmethod
    def _fog(self, inner):
        pass

    def fog(self, inner):
        # 两个变换的复合 newtranform(obj) = self(inner(obj))
        if self.__class__ == inner.__class__:
            return self._fog(inner)
        elif self.isCompatible(type(inner)):
            return self._fog(self.__class__(inner))
        elif inner.isCompatible(type(self)):
            return inner.__class__(self)._fog(inner)
        else:
            return ValueError(f'unsupport fog, type(outer)={type(self)}, type(inner)={type(inner)}')

    @abstractmethod
    def arr(self):
        pass


class KeepTransform(Transform):
    def __new__(cls):
        if not hasattr(cls, '_obj'):
            cls._obj = super().__new__(cls)
        return cls._obj

    def apply(self, img, shape):
        return img[:shape[0], :shape[1]]

    def transpoints(self, points):
        return points

    def inv(self):
        return self

    @classmethod
    def keep(cls):
        return cls()

    def _fog(self, inner):
        return inner

    def __str__(self):
        return 'KeepTransform'

    def __repr__(self):
        return 'KeepTransform()'

    @property
    def x0(self):
        return 0

    @property
    def y0(self):
        return 0

    @property
    def nz(self):
        return 1

    @property
    def arr(self):
        return np.array([[1, 0, 0],
                         [0, 1, 0]])


class MoveZoomTransform(Transform):  # 平移缩放变换
    def __init__(self, x0=0, y0=0, nz=1):
        if isinstance(x0, KeepTransform):
            x0 = 0
            y0 = 0
            nz = 1
        self.x0 = x0
        self.y0 = y0
        self.nz = nz

    @cached_property
    def arr(self):
        nz = self.nz
        return np.array([[nz, 0,  self.x0],
                         [0,  nz, self.y0]])

    def apply(self, img, shape):
        pimg = Image.fromarray(img)
        height, width = shape
        left, upper = self.x0, self.y0
        right, lower = self((width, height))
        pimg = pimg.crop((int(left), int(upper), int(right), int(lower)))
        pimg = pimg.resize((width, height))
        return np.array(pimg)

    def transpoints(self, points):
        return points*self.nz + (self.x0, self.y0)

    def inv(self):
        return MoveZoomTransform(
            x0=-self.x0/self.nz,
            y0=-self.y0/self.nz,
            nz=1/self.nz
        )

    @classmethod
    def keep(cls):
        return cls(x0=0, y0=0, nz=1)

    def _fog(self, inner):
        return MoveZoomTransform(
            x0=self.x0+inner.x0*self.nz,
            y0=self.y0+inner.y0*self.nz,
            nz=self.nz*inner.nz
        )

    @property
    def arr(self):
        nz = self.nz
        return np.array([[nz, 0,  self.x0],
                         [0,  nz, self.y0]])

    def __str__(self):
        return f'xy0=({self.x0}, {self.y0}), nz={self.nz}'

    def __repr__(self):
        return f'MoveZoomTransform(x0={self.x0}, y0={self.y0}, nz={self.nz})'


class CongruentTransform(Transform):  # 合同变换(全等变换)
    def __init__(self, x0=0, y0=0, radX=0, filp=False):
        if isinstance(x0, KeepTransform):
            x0 = 0
            y0 = 0
            radX = 0
            filp = False
        self.x0 = x0
        self.y0 = y0
        self.radX = radX  # 原X轴在新坐标系下的对应角度
        self.filp = filp

    @property
    def radY(self):
        if not self.filp:
            return self.radX + np.pi/2
        else:
            return self.radX - np.pi/2

    @cached_property
    def arr(self):
        cosX = np.cos(self.radX)
        sinX = np.sin(self.radX)
        if not self.filp:
            return np.array([[cosX, -sinX, self.x0],
                             [sinX,  cosX, self.y0]])
        else:
            return np.array([[cosX,  sinX, self.x0],
                             [sinX, -cosX, self.y0]])

    def apply(self, img, shape, *args, **kwargs):
        t_affine = AffineTransform(self)
        return cv2.warpAffine(img, t_affine, shape, *args, **kwargs)

    def transpoints(self, points):
        at = AffineTransform(self)
        return at.transpoints(points)

    @classmethod
    def keep(cls):
        return cls(x0=0, y0=0, radX=0, filp=False)

    def inv(self):
        if not self.filp:
            radX_ = -self.radX
        else:
            radX_ = self.radX
        arr = self.arr
        (x0_, y0_) = np.linalg.solve(self.arr[:2,:2], (-self.x0, -self.y0))
        return CongruentTransform(x0_, y0_, radX_, self.filp)

    def _fog(self, inner):
        if not self.filp:
            radX_ = self.radX + inner.radX
        else:
            radX_ = self.radX - inner.radX
        filp_ = self.filp != inner.filp
        x0_, y0_ = self((inner.x0, inner.y0))
        return CongruentTransform(x0_, y0_, radX_, filp_)

    def __str__(self):
        return f'xy0=({self.x0}, {self.y0}), degX={self.radX/np.pi*180:.3f}, filp={self.filp}'

    def __repr__(self):
        return f'CongruentTransform(x0={self.x0}, y0={self.y0}, radX={self.radX}, filp={self.filp})'


class SimilarTransform(CongruentTransform):  # 相似变换
    compatibleLst = [MoveZoomTransform, CongruentTransform]
    def __init__(self, x0=0, y0=0, nz=1, radX=0, filp=False):
        if isinstance(x0, CongruentTransform):
            t = x0
            super().__init__(t.x0, t.y0, t.radX, t.filp)
        elif isinstance(x0, MoveZoomTransform):
            t = x0
            super().__init__(t.x0, t.y0, 0, False)
            nz = t.nz
        elif isinstance(x0, KeepTransform):
            super().__init__(0, 0, 0, False)
            nz = 1
        else:
            super().__init__(x0, y0, radX, filp)
        self.nz = nz

    @cached_property
    def arr(self):
        nz = self.nz
        cosX = np.cos(self.radX)
        sinX = np.sin(self.radX)
        if not self.filp:
            return np.array([[nz*cosX, -nz*sinX, self.x0],
                             [nz*sinX,  nz*cosX, self.y0]])
        else:
            return np.array([[nz*cosX,  nz*sinX, self.x0],
                             [nz*sinX, -nz*cosX, self.y0]])

    def inv(self):
        return self.__class__(super().inv(), nz=1/self.nz)

    @classmethod
    def keep(cls):
        return cls(x0=0, y0=0, nz=1, radX=0, filp=False)

    def _fog(self, inner):
        return self.__class__(super()._fog(inner), nz=self.nz*inner.nz)

    def __str__(self):
        return f'xy0=({self.x0}, {self.y0}), degX={self.radX/np.pi*180:.3f}, filp={self.filp}'

    def __repr__(self):
        return f'SimilarTransform(x0={self.x0}, y0={self.y0}, nz={self.nz}, radX={self.radX}, filp={self.filp})'


class AffineTransform(Transform, np.ndarray):  # 仿射变换
    compatibleLst = [SimilarTransform]
    def __new__(cls, t):
        if hasattr(t, 'arr'):
            t = t.arr
        if isinstance(t, np.ndarray):
            assert t.shape == (2, 3)
            t_ = np.asarray(t)
        else:
            raise TypeError(f'unsupport type {type(t)}')
        return t_.view(cls)

    def apply(self, img, shape, *args, **kwargs):
        return cv2.warpAffine(img, self, shape, *args, **kwargs)

    def transpoints(self, points):
        x = np.matmul(points, np.array(self[:2,:2]).transpose())  #(A . x^T)^T = x . A^T
        x += self[:,2]
        return x

    def inv(self):
        arr9 = np.concatenate((t, [[0,0,1]]), axis=0)
        arr9inv = np.linalg.inv(arr9)
        return AffineTransform(arr9inv[:2])

    @classmethod
    def keep(cls):
        return cls(np.eye(3)[:2])

    def _fog(self, inner):
        arr_4 = np.matmul(np.array(self[:2,:2]), np.array(inner[:2,:2]))
        xy0_ = np.dot(np.array(self[:2,:2]), inner[:,2]) + self[:,2]
        arr_ = np.concatenate((arr_4, xy0_.reshape((2,1))), axis=1)
        return AffineTransform(arr_)

    @property
    def x0(self):
        return self[0, 2]

    @property
    def y0(self):
        return self[1, 2]

    @property
    def nz(self):
        return np.sqrt(np.sum(np.array(self[:2,:2])**2)/2)

    @property
    def arr(self):
        return np.array(self)


class PerspectiveTransform(AffineTransform):  # 透视变换
    compatibleLst = [AffineTransform]
    def __new__(cls, t):
        if hasattr(t, 'arr'):
            t = t.arr
        if isinstance(t, np.ndarray):
            if t.shape == (3, 3):
                t_ = np.array(t)
            elif t.shape == (2, 3):
                t_ = np.concatenate((t, [[0,0,1]]), axis=0)
            else:
                raise ValueError(f'invaild shape {t.shape}')
        else:
            raise TypeError(f'unsupport type {type(t)}')
        return t_.view(cls)

    def apply(self, img, shape, *args, **kwargs):
        return cv2.warpPerspective(img, self, shape, *args, **kwargs)

    def inv(self):
        t = np.linalg.inv(self)
        t /= t[-1, -1]
        return t

    @classmethod
    def keep(cls):
        return cls(np.eye(3))

    def transpoints(self, points):
        ones = np.ones((points.shape[0], 1))
        x = np.concatenate((points, ones), axis=1)
        x = np.matmul(x, np.array(self).transpose())  #(A . x^T)^T = x . A^T
        x /= x[:,2].reshape((-1,1))
        return x[:,:2]

    def _fog(self, inner):
        t = np.matmul(self, inner)
        t /= t[-1, -1]
        return t

    def nz_at(self, x, y):
        alpha = self[2,0]*x + self[2,1]*y + self[2,2]
        return np.sqrt(np.sum((np.array(self[:2,:2])/alpha)**2)/2)
