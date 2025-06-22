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
from preprocess_seq import PreprocessNoEst
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
        'detEdge',
        'dilsize=',
        'maxpix_sift=',
        'maxpixel_sift=',
        'maxpixel=',
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
        'maxpix_sift':'maxpixel',
        'maxpixel_sift':'maxpixel',
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
        'detEdge':False,
        'dilsize':8,
        'maxpixel':1e7,
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
        H, area_B_in_A_wkt, n_points, last_update = result_in_db
        area_B_in_A = shapely.from_wkt(area_B_in_A_wkt)
        print(H)
        print(area_B_in_A.wkt)
        print(n_points, 'points')
        print('last_update', last_update)
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
        im.set_estT(Hx)  # TODO: 使用该估计

    pseq = PreprocessNoEst.params_from_3dict(opt_A, opt_B, {}, raise_unknown=False)
    im.setPreprocessSeq(pseq)
    im.setParam_compare(outpath_match=opts['imgMatch'],
                        maxpoints1=opt_A['maxpoint_sift'],
                        maxpoints2=opt_B['maxpoint_sift'],
                        threshold_m1m2_ratio=opts['threshold_m1m2_ratio'])
    H_orig, n_match, iters = im.match_iter()
    if H_orig is None:
        print('match failed')
    print(f'{n_match} match points, {iters} iters')
    print('mearused perspective matrix:\n', H_orig)
    if n_match < opts['minmatch']:
        print('too few match keypoint pairs')
        exit()

    if opts['addto_db']:
        # TODO: 检查数据库里是否已存在，避免UNIQUE约束错误
        B_in_A = im.get_poly_B_in_A()
        A_in_B = im.get_poly_A_in_B()
        if result_in_db is None:
            db.insert_replace_match(iidA, iidB, H_orig, B_in_A.wkt, n_match)
            db.insert_replace_match(iidB, iidA, H_orig.inv(), A_in_B.wkt, n_match)
        else:
            db.insert_replace_match(iidA, iidB, H_orig, B_in_A.wkt, n_match)
            db.insert_replace_match(iidB, iidA, H_orig.inv(), A_in_B.wkt, n_match)
        db.commit()
        db.close()

    # TODO: 修复create_rb3dview实现
    #if opts['img3D'] is not None:
    #    create_rb3dview(imgA[:32766, :32766], imgB[:32766,:32766], H_orig, opts['img3D']) #32766是由于SHRT_MAX限制
