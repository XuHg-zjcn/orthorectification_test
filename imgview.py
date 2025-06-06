#!/usr/bin/env python3
#############################################################################
# GDAL遥感影像封装
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
from osgeo import gdal
import cv2


def pow2_count(x):
    count = 0
    while True:
        if x % 2 == 0:
            x //= 2
            count += 1
        else:
            break
    return count


def process_index_neg_none(i, isize, inone):
    if i is None:
        i = inone
    if i >= 0:
        return min(i, isize-1)
    elif i < 0:
        return max(0, isize+i)
    else:
        raise ValueError()


def apply_slice(x0, xsize, scale, slice_x):
    xstart_in_old = process_index_neg_none(slice_x.start, xsize, 0)
    xstop_in_old = process_index_neg_none(slice_x.stop, xsize, -1)
    xstep_in_old = slice_x.step if slice_x.step is not None else 1
    x0_ = x0 + xstart_in_old * scale
    xsize_ = (xstop_in_old-xstart_in_old)//xstep_in_old
    scale_ = scale*xstep_in_old
    return x0_, xsize_, scale_


class ImgView:
    def __init__(self, band: gdal.Band, x0=0, y0=0, scale=1, xsize=None, ysize=None):
        self.band = band
        self.x0 = x0     # x0,y0是在原图中的左上角坐标
        self.y0 = y0
        self.scale = scale  # 当前视图相对于原图的缩小倍率
        self.xsize = xsize or band.XSize  # 当前视图的大小
        self.ysize = ysize or band.YSize

    @property
    def shape(self):
        return (self.ysize, self.xsize)

    # using like numpy array
    def __getitem__(self, key):
        slice_x = key[1]
        slice_y = key[0]
        assert isinstance(slice_x, slice)
        assert isinstance(slice_y, slice)
        assert slice_x.step == slice_y.step
        x0_, xsize_, xscale_ = apply_slice(self.x0, self.xsize, self.scale, slice_x)
        y0_, ysize_, yscale_ = apply_slice(self.y0, self.ysize, self.scale, slice_y)
        assert xscale_ == yscale_
        scale_ = xscale_
        return ImgView(self.band, x0_, y0_, scale_, xsize_, ysize_)

    def trim_scale(self):
        if self.band.GetOverviewCount() == 0:
            return self  # 没有金字塔，不用调整了
        width_orig = self.xsize*self.scale
        height_orig = self.ysize*self.scale
        scale = self.scale
        if scale < 1.5:
            scale = 1
        n2 = round(math.log2(scale)*2)
        if n2 == 1:
            n2 = 2  # 不可缩小1.5倍，改为2倍
        if n2 % 2 == 0:
            # n2是偶数，缩小2**n倍
            n = n2 // 2
            scale_ = 2**n
        else:
            # n2是奇数，缩小3*2**n倍
            n = (n2-3) // 2
            scale_ = 3*2**n
        return ImgView(self.band, self.x0, self.y0, scale_, width_orig//scale_, height_orig//scale_)

    def _get_array_no_overview(self):
        width_orig = self.xsize*self.scale
        height_orig = self.ysize*self.scale
        img = self.band.ReadAsArray(
            xoff=self.x0, yoff=self.y0,
            win_xsize=width_orig, win_ysize=height_orig)
        print(f'need scale={self.scale}')
        print('read full resol')
        if self.scale != 1:
            return cv2.resize(img, dsize=(self.xsize, self.ysize), interpolation=cv2.INTER_AREA)
        else:
            return img

    def _get_array_with_overview(self):
        width_orig = self.xsize*self.scale
        height_orig = self.ysize*self.scale
        n = pow2_count(self.scale)
        print(f'need scale {self.scale}')
        if n == 0:
            img = self.band.ReadAsArray(
                xoff=self.x0, yoff=self.y0,
                win_xsize=width_orig, win_ysize=height_orig)
            scale2 = self.scale
            print(f'read full resol')
        else:
            ov_count = self.band.GetOverviewCount()
            ov_i = min(n-1, ov_count-1)
            assert ov_i >= 0
            ov_scale = 2**(ov_i+1)
            img = self.band.GetOverview(ov_i).ReadAsArray(
                xoff=self.x0//ov_scale, yoff=self.y0//ov_scale,
                win_xsize=width_orig//ov_scale, win_ysize=height_orig//ov_scale
            )
            print(f'read overview scale={ov_scale}')
            scale2 = self.scale//ov_scale
        if scale2 != 1:
            return cv2.resize(img, dsize=(self.xsize, self.ysize), interpolation=cv2.INTER_AREA)
        else:
            return img

    def get_array(self):
        if self.band.GetOverviewCount() == 0:
            return self._get_array_no_overview()
        else:
            return self._get_array_with_overview()
