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
import numpy as np
import shapely
from preprocess_single import preprocess, perspective_boundingbox
from metadata import read_metadata
from imgmatch import H_transpose, compare, create_rb3dview
import database
import import_img
from common import shaply_proj


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
        transform_blob, area_b_in_a_wkt = result_in_db
        area_b_in_a = shapely.from_wkt(area_b_in_a_wkt)
        H = np.frombuffer(transform_blob, dtype=np.float64).reshape((3, 3))
        print(H)
        print(area_b_in_a.wkt)
        if not opts['update_db']:
            exit()

    # TODO: 使用GDAL读取遥感影像
    img1 = cv2.imread(opt_a['path'], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(opt_b['path'], cv2.IMREAD_GRAYSCALE)

    paraA = read_metadata(opt_a['path'])
    paraB = read_metadata(opt_b['path'])
    hasMetadata = False
    x0a = 0
    y0a = 0
    x0b = 0
    y0b = 0
    na = 1
    nb = 1
    if paraA is not None and paraB is not None:
        hasMetadata = True
        Hx = np.matmul(np.linalg.inv(paraA[1]), paraB[1])
        print('estimated perspective matrix by metadata:\n', Hx)  # 估计原图的透视矩阵
        bbox_at_coordA = perspective_boundingbox(
            Hx,
            img1.shape[1], img1.shape[0],
            img2.shape[1], img2.shape[0])
        print(bbox_at_coordA)
        bbox_at_coordB = perspective_boundingbox(
            np.linalg.inv(Hx),
            img2.shape[1], img2.shape[0],
            img1.shape[1], img1.shape[0])
        print(bbox_at_coordB)

        img1 = img1[bbox_at_coordA[1]:bbox_at_coordA[3],
                    bbox_at_coordA[0]:bbox_at_coordA[2]]
        img2 = img2[bbox_at_coordB[1]:bbox_at_coordB[3],
                    bbox_at_coordB[0]:bbox_at_coordB[2]]
        x0a += bbox_at_coordA[0]
        y0a += bbox_at_coordA[1]
        x0b += bbox_at_coordB[0]
        y0b += bbox_at_coordB[1]

    img1_, n1, xy1 = preprocess(img1, 'img1',
                                maxpixel_out=opt_a['maxpixel_sift'],
                                predown=opt_a['predown'],
                                laplace=opt_a['laplace'],
                                dilsize=opt_a['dilsize'],
                                cutblack_topbottom=opt_a['cutblack_topbottom'],
                                cutblack_leftright=opt_a['cutblack_leftright'])
    img2_, n2, xy2 = preprocess(img2, 'img2',
                                maxpixel_out=opt_b['maxpixel_sift'],
                                predown=opt_b['predown'],
                                laplace=opt_b['laplace'],
                                dilsize=opt_b['dilsize'],
                                cutblack_topbottom=opt_b['cutblack_topbottom'],
                                cutblack_leftright=opt_b['cutblack_leftright'])
    na *= n1
    nb *= n2
    x0a += xy1[0]
    y0a += xy1[1]
    x0b += xy2[0]
    y0b += xy2[1]

    threshold_m1m2_ratio = opts['threshold_m1m2_ratio']
    H_, matchs = compare(img1_, img2_,                            # 已切割和缩小图像对的(B->A)透视矩阵
        opts['imgMatch'],
        maxpoints1=opt_a['maxpoint_sift'],
        maxpoints2=opt_b['maxpoint_sift'],
        threshold_m1m2_ratio=threshold_m1m2_ratio)
    H_orig = H_transpose(                                         # 原图像对的(B->A)透视矩阵
        H_,
        x0_d=x0a, y0_d=y0a, zoom_d=n1,
        x0_s=x0b, y0_s=y0b, zoom_s=n2)
    print('mearused perspective matrix:\n', H_orig)
    if len(matchs) < opts['minmatch']:
        print('too few match keypoint pairs')
        exit()
    # TODO: 更进一步，使用高分辨率的图像分块处理，进行更高精度的对齐

    x_a = x0a + img1_.shape[1]*na
    y_a = y0a + img1_.shape[0]*na
    x_b = x0b + img2_.shape[1]*nb
    y_b = y0b + img2_.shape[0]*nb
    rect_a = shapely.polygons([(x0a, y0a), (x_a, y0a), (x_a, y_a), (x0a, y_a)])
    rect_b = shapely.polygons([(x0b, y0b), (x_b, y0b), (x_b, y_b), (x0b, y_b)])
    b_in_a = shapely.transform(rect_b, functools.partial(shaply_proj, H_orig))

    if opts['addto_db']:
        assert H_orig.shape == (3,3)
        H_blob = H_orig.astype(np.float64).tobytes()
        # TODO: 检查数据库里是否已存在，避免UNIQUE约束错误
        if result_in_db is None:
            db.insert_match(iid1, iid2, H_blob, b_in_a.wkt)
        else:
            db.update_match(iid1, iid2, H_blob, b_in_a.wkt)
        db.commit()
        db.close()

    if opts['img3D'] is not None:
        create_rb3dview(img1[:32766, :32766], img2[:32766,:32766], H_orig, opts['img3D']) #32766是由于SHRT_MAX限制
