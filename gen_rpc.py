#!/usr/bin/env python3
#############################################################################
# 生成遥感影像RPC文件
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
import database
import import_img
import numpy as np
import matplotlib.pyplot as plt
import shapely
import common
import transform
import os

#TODO: 应该将匹配点对储存到数据库，用点来配准

def write_rpc(fn, tf_img_to_geo, imgsize):
    f = open(fn, 'w')
    imgwidth = imgsize[0]
    imgheight = imgsize[1]
    img_xcen = imgwidth/2
    img_ycen = imgheight/2
    img_scale = max(imgwidth, imgheight)/2
    box_img = shapely.box(0, 0, imgwidth, imgheight)
    lon_min, lat_min, lon_max, lat_max, = tf_img_to_geo(box_img).bounds
    lon_cen = (lon_min+lon_max)/2
    lat_cen = (lat_min+lat_max)/2
    lon_delta = lon_max-lon_min
    lat_delta = lat_max-lat_min
    geo_scale = max(lon_delta, lat_delta)/2
    tf_normimg = transform.MoveZoomTransform(x0=img_xcen, y0=img_ycen, nz=img_scale)
    tf_normgeo = transform.MoveZoomTransform(x0=lon_cen, y0=lat_cen, nz=geo_scale)
    tf_norm_geo_to_img = (tf_normgeo.inv().fog(tf_img_to_geo).fog(tf_normimg)).inv()
    f.write('ERR_BIAS: 0.0\n'
            'ERR_RAND: 0.0\n'
            f'LINE_OFF: {img_ycen}\n'
            f'SAMP_OFF: {img_xcen}\n'
            f'LAT_OFF: {lat_cen}\n'
            f'LONG_OFF: {lon_cen}\n'
            'HEIGHT_OFF: 1\n'
            f'LINE_SCALE: {img_scale}\n'
            f'SAMP_SCALE: {img_scale}\n'
            f'LAT_SCALE: {geo_scale}\n'
            f'LONG_SCALE: {geo_scale}\n'
            'HEIGHT_SCALE: 1\n')
    line_num_coeff = np.zeros(20)
    line_num_coeff[0] = tf_norm_geo_to_img[1, 2]
    line_num_coeff[1] = tf_norm_geo_to_img[1, 0]
    line_num_coeff[2] = tf_norm_geo_to_img[1, 1]
    samp_num_coeff = np.zeros(20)
    samp_num_coeff[0] = tf_norm_geo_to_img[0, 2]
    samp_num_coeff[1] = tf_norm_geo_to_img[0, 0]
    samp_num_coeff[2] = tf_norm_geo_to_img[0, 1]
    line_den_coeff = np.zeros(20)
    line_den_coeff[0] = tf_norm_geo_to_img[2, 2]
    line_den_coeff[1] = tf_norm_geo_to_img[2, 0]
    line_den_coeff[2] = tf_norm_geo_to_img[2, 1]
    samp_den_coeff = line_den_coeff
    for i in range(20):
        f.write(f'LINE_NUM_COEFF_{i+1}: {line_num_coeff[i]}\n')
    for i in range(20):
        f.write(f'LINE_DEN_COEFF_{i+1}: {line_den_coeff[i]}\n')
    for i in range(20):
        f.write(f'SAMP_NUM_COEFF_{i+1}: {samp_num_coeff[i]}\n')
    for i in range(20):
        f.write(f'SAMP_DEN_COEFF_{i+1}: {samp_den_coeff[i]}\n')
    f.close()


if __name__ == '__main__':
    imgpath = sys.argv[1]
    db = database.Database('data/imagery.db')
    fid, iid = import_img.import_img(db, imgpath)
    imgfields = db.get_img_allfield(iid)
    lst = []
    weights = []
    results = db.get_matchways_more_tfgeo(iid, 2)
    print(f'found {len(results)} matchways to image with transform_geo')
    for path, weight, tfgeo in results:
        if weight < 20:
            continue
        weights.append(weight)
        lst.append(tfgeo.flatten()[:-1])
    print(f'filterd {len(lst)} matchways weight more than 20')
    tfs = np.array(lst)

    wmid = common.weight_median(tfs, weights, axis=0)
    wmad = common.weight_median(np.abs(tfs-wmid), weights, axis=0)
    wmad[wmad == 0] = float('inf')
    z_score_m = np.sqrt(np.nanmean(((tfs-wmid)/wmad)**2, axis=1))
    print(z_score_m)
    tfs_filted = tfs[z_score_m<3]
    weights_filted = tfs[z_score_m<3]
    print(f'filterd {len(tfs_filted)} matchways by z_score_m')
    tf_avg_f = np.average(tfs_filted, axis=0, weights=weights_filted)
    tf_img_to_geo = np.zeros(9)
    tf_img_to_geo[:8] = tf_avg_f
    tf_img_to_geo[-1] = 1
    tf_img_to_geo = tf_img_to_geo.reshape((3,3))
    tf_img_to_geo = transform.PerspectiveTransform(tf_img_to_geo)
    print('tf_img_to_geo:')
    print(tf_img_to_geo)
    rpc_fn = os.path.splitext(imgpath)[0]+'_RPC.TXT'
    write_rpc(rpc_fn, tf_img_to_geo, (imgfields['width'], imgfields['height']))
