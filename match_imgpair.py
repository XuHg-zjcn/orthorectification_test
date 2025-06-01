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
import numpy as np
from preprocess_single import preprocess
from metadata import read_metadata
from imgmatch import H_transpose, compare, create_rb3dview
import database
import import_img
import shapely


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
        'laplace',
        'dilsize=',
        'maxpix_sift=',
        'maxpixel_sift=',
        'threshold_m1m2_ratio=',
        'cutblack_topbottom',
        'cutblack_leftright',
        'addto_db',
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
        'addto_db':False,
    }
    img_para_default = {
        'laplace':False,
        'dilsize':8,
        'maxpixel_sift':1e7,
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
    # TODO: 使用GDAL读取遥感影像
    opt_a = opts['imgA']
    opt_b = opts['imgB']
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
                                laplace=opt_a['laplace'],
                                dilsize=opt_a['dilsize'],
                                cutblack_topbottom=opt_a['cutblack_topbottom'],
                                cutblack_leftright=opt_a['cutblack_leftright'])
    img2_, n2, xy2 = preprocess(img2, 'img2',
                                maxpixel_out=opt_b['maxpixel_sift'],
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
    H_ = compare(img1_, img2_,                                    # 已切割和缩小图像对的(B->A)透视矩阵
                 opts['imgMatch'],
                 maxpoints=10000,
                 threshold_m1m2_ratio=threshold_m1m2_ratio)
    H_orig = H_transpose(                                         # 原图像对的(B->A)透视矩阵
        H_,
        x0_d=x0a, y0_d=y0a, zoom_d=n1,
        x0_s=x0b, y0_s=y0b, zoom_s=n2)
    print('mearused perspective matrix:\n', H_orig)
    # TODO: 更进一步，对重叠区域进行SIFT匹配，使用高分辨率的图像分块处理，进行更高精度的对齐

    def proj(x):
        ones = np.ones((x.shape[0], 1))
        x = np.concatenate((x, ones), axis=1)
        x = np.matmul(x, H_orig.transpose())  #(A . x^T)^T = x . A^T
        x /= x[:,2].reshape((-1,1))
        return x[:,:2]

    x_a = x0a + img1_.shape[1]*na
    y_a = y0a + img1_.shape[0]*na
    x_b = x0b + img2_.shape[1]*nb
    y_b = y0b + img2_.shape[0]*nb
    rect_a = shapely.polygons([(x0a, y0a), (x_a, y0a), (x_a, y_a), (x0a, y_a)])
    rect_b = shapely.polygons([(x0b, y0b), (x_b, y0b), (x_b, y_b), (x0b, y_b)])
    b_in_a = shapely.transform(rect_b, proj)

    if opts['addto_db']:
        db = database.Database('data/imagery.db')
        assert H_orig.shape == (3,3)
        H_blob = H_orig.astype(np.float64).tobytes()
        fid1, iid1 = import_img.import_img(db, opt_a['path'])
        fid2, iid2 = import_img.import_img(db, opt_b['path'])
        # TODO: 检查数据库里是否已存在，避免UNIQUE约束错误
        db.insert_match(iid1, iid2, H_blob, b_in_a.wkt)
        db.commit()
        db.close()

    if opts['img3D'] is not None:
        create_rb3dview(img1[:32766, :32766], img2[:32766,:32766], H_orig, opts['img3D']) #32766是由于SHRT_MAX限制
