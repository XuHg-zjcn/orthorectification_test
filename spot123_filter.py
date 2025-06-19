#!/usr/bin/env python3
#############################################################################
# 过滤SPOT 1-3影像条纹噪声
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
import sys
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal


def fix_heavy_errs(delta_x):
    def class_neighbour_num(lst):
        # 按相邻数字分类
        lst = sorted(lst)
        groups = []
        prv_err = None
        curr_group = []
        for i in lst:
            if prv_err is None or i-prv_err > 1:
                # 没有之前元素 或 当前元素与之前元素间隔超过1
                if len(curr_group) >= 1:
                    groups.append(curr_group)
                    curr_group = []
            curr_group.append(i)
            prv_err = i
        if len(curr_group) >= 1:
            groups.append(curr_group)
        return groups

    seg_std = []  # 分段标准差
    lst_err = []
    for i in range(4):
        begin_delta_x = i*1500  # 1500是经验值，大多数delta_x看起来都分成等长四段
        seg_delta_x = delta_x[i*1500:(i+1)*1500]
        seg_delta_x_noedge = delta_x[i*1500:(i+1)*1500-2]
        std = seg_delta_x_noedge.std()
        seg_std.append(std)
        where = np.where(np.abs(seg_delta_x)/std > 5)[0]
        for i in where:
            lst_err.append(i + begin_delta_x)
    groups = class_neighbour_num(lst_err) # 找出错误的列
    fix_i = []
    for g in groups:
        if len(g) == 0 or len(g) > 3:
            continue
        m = -1
        i_argmax = None
        for i in g:
            if np.abs(delta_x[i]) > m:
                m = np.abs(delta_x[i])
                i_argmax = i
        if i_argmax-1 >= 0:
            delta_x[i_argmax-1] = 0  # 左右两列不要修改
        if i_argmax+1 < delta_x.shape[0]:
            delta_x[i_argmax+1] = 0  # 左右两列不要修改
        delta_x[i_argmax] *= 2       # 该列使用全权重,以后会除以2
        fix_i.append(i_argmax)
    return fix_i


if __name__ == '__main__':
    path_in = sys.argv[1]
    path_out = sys.argv[2]
    ds = gdal.Open(path_in, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    img = band.ReadAsArray()
    print(img)
    img -= img.min()
    img_i16 = img.astype(np.int16)*2
    mean_y = np.mean(img_i16, axis=1) # 计算每一行的平均值
    mean_x = np.mean(img_i16, axis=0) # 计算每一列的平均值
    delta_y = np.convolve(mean_y, [-0.5, 1, -0.5], 'vaild')
    delta_x = np.convolve(mean_x, [-0.5, 1, -0.5], 'vaild')
    fix_i = fix_heavy_errs(delta_x)
    print(fix_i)
    #plt.plot(delta_x) # 看起来分四段方差不同的区间
    #plt.plot(np.convolve(np.mean(img_i16[1500], axis=1), [-0.5, 1, -0.5], 'vaild')) # 看起来X上各区间在Y没有差异
    #plt.plot(np.convolve(np.mean(img_i16[1500:3000], axis=1), [-0.5, 1, -0.5], 'vaild'))
    #plt.plot(np.convolve(np.mean(img_i16[3000:4500], axis=1), [-0.5, 1, -0.5], 'vaild'))
    #plt.plot(np.convolve(np.mean(img_i16[4500:], axis=1), [-0.5, 1, -0.5], 'vaild'))
    #plt.show()
    img_i16[1:-1] -= np.round(delta_y/2).astype(np.int16).reshape((-1, 1))
    img_i16[:, 1:-1] -= np.round(delta_x/2).astype(np.int16).reshape((1, -1))
    img_i16 -= img_i16.min()
    img_u8 = np.clip(img_i16, 0, 255).astype(np.uint8)

    # ref: https://gis.stackexchange.com/questions/164853/reading-modifying-and-writing-a-geotiff-with-gdal-in-python
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(path_out, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)
    outdata.SetGeoTransform(ds.GetGeoTransform())
    outdata.SetProjection(ds.GetProjection())
    outdata.SetGCPs(ds.GetGCPs(), ds.GetGCPProjection())
    outdata.GetRasterBand(1).WriteArray(img_u8)
    outdata.Close()
