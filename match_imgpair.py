#!/usr/bin/env python3
#############################################################################
# 遥感影像像对匹配
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
import getopt
import cv2
from osgeo import gdal
import numpy as np
import shapely
import preprocess
from metadata import read_metadata
from imgmatch import create_rb3dview
import database
import import_img
from common import shapely_perspective, CropZoom2D
from imgview import ImgView
from imgmatch2 import ImgMatch


def parser_options():
    def mycast(T, valstr):
        if not isinstance(T, type):
            raise TypeError('T is not a type')
        if T == bool:
            return True
        elif T == int:
            return int(valstr)
        elif T == float:
            return float(valstr)
        else:
            return str(valstr)

    short_options = 'a:b:m:3:'
    long_options = [
        'imgA=',
        'imgB=',
        'predown=',
        'laplace',
        'dilsize=',
        'maxpix_sift=',
        'maxpixel_sift=',
        'maxpoint_sift=',
        'threshold_m1m2_ratio=',
        'cutblack_topbottom',
        'cutblack_leftright',
        'addto_db',
        'update_db',
        'minmatch='
    ]
    short_to_long = {
        'a':'imgA',
        'b':'imgB',
        '3':'img3D',
        'm':'imgMatch',
        'maxpix_sift':'maxpixel_sift'
    }
    opts = {
        'imgA':None,
        'imgB':None,
        'img3D':None,
        'imgMatch':None,
        'threshold_m1m2_ratio':0.8,
        'addto_db':True,
        'update_db':False,
        'minmatch':15,
    }
    img_para_default = {
        'predown':0,
        'laplace':False,
        'dilsize':8,
        'maxpixel_sift':1e7,
        'maxpoint_sift':10000,
        'cutblack_topbottom':False,
        'cutblack_leftright':False}
    curr_img_para = None
    optlist, args = getopt.getopt(sys.argv[1:], short_options, long_options)
    for opt, valstr in optlist:
        optname = opt.lstrip('-')
        if optname in short_to_long:
            optname = short_to_long[optname]
        if optname == 'imgA' or optname == 'imgB':
            curr_img_para =  img_para_default.copy()
            curr_img_para['path'] = str(valstr)
            opts[optname] = curr_img_para
        elif optname in img_para_default:
            value = mycast(type(img_para_default[optname]), valstr)
            if curr_img_para is None:
                img_para_default[optname] = value
            else:
                curr_img_para[optname] = value
        elif optname in opts:
            value = mycast(type(opts[optname]), valstr)
            opts[optname] = value
        else:
            raise ValueError(f'unknown opt {optname}')
    return opts


if __name__ == '__main__':
    opts = parser_options()
    opt_A = opts['imgA']
    opt_B = opts['imgB']
    db = database.Database('data/imagery.db')
    fidA, iidA = import_img.import_img(db, opt_A['path'])
    fidB, iidB = import_img.import_img(db, opt_B['path'])
    result_in_db = db.get_match(iidA, iidB)
    if result_in_db is not None:
        print('found in database')
        H, area_B_in_A_wkt = result_in_db
        area_B_in_A = shapely.from_wkt(area_B_in_A_wkt)
        print(H)
        print(area_B_in_A.wkt)
        if not opts['update_db']:
            exit()

    dsA = gdal.Open(opt_A['path'], gdal.GA_ReadOnly)
    dsB = gdal.Open(opt_B['path'], gdal.GA_ReadOnly)
    ivA = ImgView(dsA.GetRasterBand(1))
    ivB = ImgView(dsB.GetRasterBand(1))

    paraA = read_metadata(opt_A['path'])
    paraB = read_metadata(opt_B['path'])
    hasMetadata = False
    im = ImgMatch(ivA, ivB)
    if paraA is not None and paraB is not None:
        hasMetadata = True
        Hx = np.matmul(np.linalg.inv(paraA[1]), paraB[1])
        print('estimated perspective matrix by metadata:\n', Hx)  # 估计原图的透视矩阵
        bbox_at_coordA = preprocess.perspective_boundingbox(
            Hx,
            ivA.shape[1], ivA.shape[0],
            ivB.shape[1], ivB.shape[0])
        print(bbox_at_coordA)
        bbox_at_coordB = preprocess.perspective_boundingbox(
            np.linalg.inv(Hx),
            ivB.shape[1], ivB.shape[0],
            ivA.shape[1], ivA.shape[0])
        print(bbox_at_coordB)

        czA = CropZoom2D.with_shape(ivA.shape,
            x0=bbox_at_coordA[0],y0=bbox_at_coordA[1],
            x1=bbox_at_coordA[2],y1=bbox_at_coordA[3])
        czB = CropZoom2D.with_shape(ivB.shape,
            x0=bbox_at_coordB[0],y0=bbox_at_coordB[1],
            x1=bbox_at_coordB[2],y1=bbox_at_coordB[3])
        im.append_pobj(preprocess.CutImg('A', czA))
        im.append_pobj(preprocess.CutImg('B', czB))

    maxpixelA = opt_A['maxpixel_sift']
    if opt_A['laplace']:
        maxpixelA *= opt_A['dilsize']**2
    im.append_pobj(preprocess.AutoZoom('A', maxpixel=maxpixelA))
    if opt_A['laplace']:
        im.append_pobj(preprocess.LaplacianAndDilate('A', nz=opt_A['dilsize']))
    if opt_A['cutblack_topbottom']:
        im.append_pobj(preprocess.CutBlackTopBottom('A'))
    if opt_A['cutblack_leftright']:
        im.append_pobj(preprocess.CutBlackLeftRight('A'))

    maxpixelB = opt_B['maxpixel_sift']
    if opt_B['laplace']:
        maxpixelB *= opt_B['dilsize']**2
    im.append_pobj(preprocess.AutoZoom('B', maxpixel=maxpixelB))
    if opt_B['laplace']:
        im.append_pobj(preprocess.LaplacianAndDilate('B', nz=opt_B['dilsize']))
    if opt_B['cutblack_topbottom']:
        im.append_pobj(preprocess.CutBlackTopBottom('B'))
    if opt_B['cutblack_leftright']:
        im.append_pobj(preprocess.CutBlackLeftRight('B'))
    im.setParam_compare(outpath_match=opts['imgMatch'],
                        maxpoints1=opt_A['maxpoint_sift'],
                        maxpoints2=opt_B['maxpoint_sift'],
                        threshold_m1m2_ratio=opts['threshold_m1m2_ratio'])
    H_orig, n_match = im.match()
    if H_orig is None:
        print('match failed')
    print('mearused perspective matrix:\n', H_orig)
    if n_match < opts['minmatch']:
        print('too few match keypoint pairs')
        exit()

    rect_A = shapely.box(0, 0, ivA.shape[1], ivA.shape[0])
    rect_B = shapely.box(0, 0, ivB.shape[1], ivB.shape[0])
    B_in_A = shapely_perspective(rect_B, H_orig)

    if opts['addto_db']:
        # TODO: 检查数据库里是否已存在，避免UNIQUE约束错误
        if result_in_db is None:
            db.insert_match(iidA, iidB, H_orig, B_in_A.wkt)
        else:
            db.update_match(iidA, iidB, H_orig, B_in_A.wkt)
        db.commit()
        db.close()

    # TODO: 修复create_rb3dview实现
    #if opts['img3D'] is not None:
    #    create_rb3dview(imgA[:32766, :32766], imgB[:32766,:32766], H_orig, opts['img3D']) #32766是由于SHRT_MAX限制
