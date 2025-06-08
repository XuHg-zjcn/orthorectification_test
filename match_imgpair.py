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
import functools
import cv2
from osgeo import gdal
import numpy as np
import shapely
from preprocess_single import preprocess, perspective_boundingbox
from metadata import read_metadata
from imgmatch import H_transpose, compare, create_rb3dview
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


# TODO: a,b 和 1,2 两种记号代表图片在代码中混用，需要进行统一
if __name__ == '__main__':
    opts = parser_options()
    opt_a = opts['imgA']
    opt_b = opts['imgB']
    db = database.Database('data/imagery.db')
    fid1, iid1 = import_img.import_img(db, opt_a['path'])
    fid2, iid2 = import_img.import_img(db, opt_b['path'])
    result_in_db = db.get_match(iid1, iid2)
    if result_in_db is not None:
        print('found in database')
        H, area_b_in_a_wkt = result_in_db
        area_b_in_a = shapely.from_wkt(area_b_in_a_wkt)
        print(H)
        print(area_b_in_a.wkt)
        if not opts['update_db']:
            exit()

    # TODO: 使用GDAL读取遥感影像
    ds1 = gdal.Open(opt_a['path'], gdal.GA_ReadOnly)
    ds2 = gdal.Open(opt_b['path'], gdal.GA_ReadOnly)
    iv1 = ImgView(ds1.GetRasterBand(1))
    iv2 = ImgView(ds2.GetRasterBand(1))

    paraA = read_metadata(opt_a['path'])
    paraB = read_metadata(opt_b['path'])
    hasMetadata = False
    im = ImgMatch(iv1, iv2)
    if paraA is not None and paraB is not None:
        hasMetadata = True
        Hx = np.matmul(np.linalg.inv(paraA[1]), paraB[1])
        print('estimated perspective matrix by metadata:\n', Hx)  # 估计原图的透视矩阵
        bbox_at_coordA = perspective_boundingbox(
            Hx,
            iv1.shape[1], iv1.shape[0],
            iv2.shape[1], iv2.shape[0])
        print(bbox_at_coordA)
        bbox_at_coordB = perspective_boundingbox(
            np.linalg.inv(Hx),
            iv2.shape[1], iv2.shape[0],
            iv1.shape[1], iv1.shape[0])
        print(bbox_at_coordB)

        cz1 = CropZoom2D.with_shape(iv1.shape,
            x0=bbox_at_coordA[0],y0=bbox_at_coordA[1],
            x1=bbox_at_coordA[2],y1=bbox_at_coordA[3])
        cz2 = CropZoom2D.with_shape(iv2.shape,
            x0=bbox_at_coordB[0],y0=bbox_at_coordB[1],
            x1=bbox_at_coordB[2],y1=bbox_at_coordB[3])
        im.set_estA(cz1)
        im.set_estB(cz2)

    im.setParam_preprocessA(maxpixel_out=opt_a['maxpixel_sift'],
                            predown=opt_a['predown'],
                            laplace=opt_a['laplace'],
                            dilsize=opt_a['dilsize'],
                            cutblack_topbottom=opt_a['cutblack_topbottom'],
                            cutblack_leftright=opt_a['cutblack_leftright'])
    im.setParam_preprocessB(maxpixel_out=opt_b['maxpixel_sift'],
                            predown=opt_b['predown'],
                            laplace=opt_b['laplace'],
                            dilsize=opt_b['dilsize'],
                            cutblack_topbottom=opt_b['cutblack_topbottom'],
                            cutblack_leftright=opt_b['cutblack_leftright'])
    im.setParam_compare(outpath_match=opts['imgMatch'],
                        maxpoints1=opt_a['maxpoint_sift'],
                        maxpoints2=opt_b['maxpoint_sift'],
                        threshold_m1m2_ratio=opts['threshold_m1m2_ratio'])
    H_orig, n_match = im.match()
    if H_orig is None:
        print('match failed')
    print('mearused perspective matrix:\n', H_orig)
    if n_match < opts['minmatch']:
        print('too few match keypoint pairs')
        exit()

    rect_a = shapely.box(0, 0, iv1.shape[1], iv1.shape[0])
    rect_b = shapely.box(0, 0, iv2.shape[1], iv2.shape[0])
    b_in_a = shapely_perspective(rect_b, H_orig)

    if opts['addto_db']:
        # TODO: 检查数据库里是否已存在，避免UNIQUE约束错误
        if result_in_db is None:
            db.insert_match(iid1, iid2, H_orig, b_in_a.wkt)
        else:
            db.update_match(iid1, iid2, H_orig, b_in_a.wkt)
        db.commit()
        db.close()

    # TODO: 修复create_rb3dview实现
    #if opts['img3D'] is not None:
    #    create_rb3dview(img1[:32766, :32766], img2[:32766,:32766], H_orig, opts['img3D']) #32766是由于SHRT_MAX限制
