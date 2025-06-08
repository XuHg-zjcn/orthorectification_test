#!/usr/bin/env python3
#############################################################################
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
import functools
import numpy as np
import shapely
import cv2


def proj(H, x):
    ones = np.ones((x.shape[0], 1))
    x = np.concatenate((x, ones), axis=1)
    x = np.matmul(x, H.transpose())  #(A . x^T)^T = x . A^T
    x /= x[:,2].reshape((-1,1))
    return x[:,:2]

def shapely_perspective(x, H):
    return shapely.transform(x, functools.partial(proj, H))

def findPerspective(x1, y1, x2, y2, coord_lt, coord_rt, coord_rb, coord_lb):
    try:
        H, _ = cv2.findHomography(
            np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]),
            np.array([coord_lt, coord_rt, coord_rb, coord_lb]))
    except:
        return None
    else:
        return H

def try_func(f, *args, **kwargs):
    try:
        retval = f(*args, **kwargs)
    except Exception as e:
        print(e)
        return None
    else:
        return retval

def process_index_neg(i, isize):
    if i >= 0:
        return min(i, isize-1)
    elif i < 0:
        return max(0, isize+i)
    else:
        raise ValueError()

class CropZoom1D:
    def __init__(self, x0=None, x1=None, nz=1, xc=None, wi=None, wo=None):
        if wi is None and None not in {wo, nz}:
            wi = wo*nz
        if x0 is None:
            if None not in {xc, x1}:
                x0 = xc - (x1-xc)
            elif None not in {x1, wi}:
                x0 = x1 - wi
            elif None not in {xc, wi}:
                x0 = xc - wi//2
            else:
                raise ValueError("can't solve x0")
        if x1 is None:
            if None not in {x0, wi}:
                x1 = x0 + wi
            elif None not in {x0, xc}:
                x1 = xc + (xc-x0)
            else:
                raise ValueError("can't solve x1")
        assert None not in [x0, x1, nz]
        self._x0 = x0
        self._x1 = x1
        self._nz = nz

    @staticmethod
    def with_length(length, x0=None, x1=None, nz=1, xc=None, wi=None, wo=None):
        if x0 is not None:
            x0 = process_index_neg(x0, length)
        if x1 is not None:
            x1 = process_index_neg(x1, length)
        if x1 is not None and all(map(lambda x:x is None, [x0, xc, wi, wo])):
            x0 = 0
        if x0 is not None and all(map(lambda x:x is None, [x1, xc, wi, wo])):
            x1 = length
        if all(map(lambda x:x is None, [x0, x1, xc, wi, wo])):
            x0 = 0
            x1 = length
        return CropZoom1D(x0, x1, nz, xc, wi, wo)

    def fog(self, g):
        return CropZoom1D(x0=self.x0+g.x0*self.nz,
                          x1=self.x0+g.x1*self.nz,
                          nz=self.nz*g.nz)

    def __getitem__(self, key):
        return self.fog(CropZoom1D.with_length(self.wo, x0=key.start, x1=key.stop, nz=key.step))

    def to_slice(self):
        return slice(self.x0, self.x1, self.nz)

    def __str__(self):
        return f'x=[{self.x0}:{self.x1}], nz={self.nz}'

    def __repr__(self):
        return f'CropZoom2D(x0={self.x0}, x1={self.x1}, nz={self.nz})'

    @property
    def x0(self):
        return self._x0

    @property
    def nz(self):
        return self._nz

    @property
    def x1(self):
        return self._x1

    @property
    def wi(self):
        return self._x1 - self._x0

    @property
    def wo(self):
        return (self._x1 - self._x0)//self._nz


class CropZoom2D:
    def __init__(self,
                 x0=None, y0=None,
                 x1=None, y1=None,
                 nz=1,
                 xc=None, yc=None,
                 wi=None, hi=None,
                 wo=None, ho=None):
        if isinstance(x0, CropZoom1D):
            self.x = x0
        else:
            self.x = CropZoom1D(x0, x1, nz, xc, wi, wo)
        if isinstance(y0, CropZoom1D):
            self.y = y0
        else:
            self.y = CropZoom1D(y0, y1, nz, yc, hi, ho)

    @staticmethod
    def with_shape(shape,
                   x0=None, y0=None,
                   x1=None, y1=None,
                   nz=1,
                   xc=None, yc=None,
                   wi=None, hi=None,
                   wo=None, ho=None):
        x_cz = CropZoom1D.with_length(shape[1], x0, x1, nz, xc, wi, wo)
        y_cz = CropZoom1D.with_length(shape[0], y0, y1, nz, yc, hi, ho)
        return CropZoom2D(x_cz, y_cz)

    def fog(self, g):
        x_cz = self.x.fog(g.x)
        y_cz = self.y.fog(g.y)
        return CropZoom2D(x_cz, y_cz)

    def __getitem__(self, key):
        slice_x = key[1]
        slice_y = key[0]
        return CropZoom2D(self.x[slice_x], self.y[slice_y])

    def to_slice(self):
        return (slice(self.y0, self.y1, self.nz),
                slice(self.x0, self.x1, self.nz))

    def __str__(self):
        return f'x=[{self.x0}:{self.x1}], y=[{self.y0}:{self.y1}], nz={self.nz}'

    def __repr__(self):
        return f'CropZoom2D(x0={self.x0}, y0={self.y0}, x1={self.x1}, y1={self.y1}, nz={self.nz})'

    @property
    def x0(self):
        return self.x.x0

    @property
    def x1(self):
        return self.x.x1

    @property
    def nz(self):
        return self.x.nz

    @property
    def wi(self):
        return self.x.wi

    @property
    def wo(self):
        return self.x.wo

    @property
    def y0(self):
        return self.y.x0

    @property
    def y1(self):
        return self.y.x1

    @property
    def hi(self):
        return self.y.wi

    @property
    def ho(self):
        return self.y.wo
