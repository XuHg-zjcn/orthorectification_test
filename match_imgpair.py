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
import cv2
import numpy as np
from optparse import OptionParser
from preprocess_single import preprocess
from metadata import read_metadata
from imgmatch import H_transpose, compare, create_rb3dview
import database
import import_img
import shapely


# TODO: a,b 和 1,2 两种记号代表图片在代码中混用，需要进行统一
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-a', '--img1', dest='img1', help='first input image path')
    parser.add_option('-b', '--img2', dest='img2', help='second input image path')
    parser.add_option('-m', '--imgmatch', dest='imgmatch', help='output draw match image path')
    parser.add_option('-3', '--img3d', dest='img3d', help='output red-cyan 3D image path')
    parser.add_option('--laplace', dest='laplace', default=False, action='store_true', help='enable laplace preprocess')
    parser.add_option('--dilsize', dest='dilsize', default=8, help='kern size of dilate and downsample before sift')
    parser.add_option('--maxpix_sift', '--maxpixel_sift', dest='maxpixel_sift', default=1e7, help='maxpixel for sift')
    parser.add_option('--threshold_m1m2_ratio', dest='threshold_m1m2_ratio', default=0.8, help='threshold for match filter: SIFT vector distance ratio of first and second match')
    parser.add_option('--cutblack_topbottom', dest='cutblack_topbottom', action='store_true', default=False, help='cut black edge on top and bottom of image')
    parser.add_option('--cutblack_leftright', dest='cutblack_leftright', action='store_true', default=False, help='cut black edge on left and right of image')
    parser.add_option('--addto_db', dest='addto_db', action='store_true', default=False, help='add record to database')
    # TODO: 添加更多命令行参数
    options, args = parser.parse_args()
    # TODO: 使用GDAL读取遥感影像
    img1 = cv2.imread(options.img1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(options.img2, cv2.IMREAD_GRAYSCALE)

    paraA = read_metadata(options.img1)
    paraB = read_metadata(options.img2)
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

    maxpixel_sift = int(float(options.maxpixel_sift))
    laplace = options.laplace
    dilsize = int(options.dilsize)
    cutblack_topbottom = options.cutblack_topbottom
    cutblack_leftright = options.cutblack_leftright
    img1_, n1, xy1 = preprocess(img1, 'img1', maxpixel_out=maxpixel_sift,
                                laplace=laplace, dilsize=dilsize,
                                cutblack_topbottom=cutblack_topbottom, cutblack_leftright=cutblack_leftright)
    img2_, n2, xy2 = preprocess(img2, 'img2', maxpixel_out=maxpixel_sift,
                               laplace=laplace, dilsize=dilsize,
                               cutblack_topbottom=cutblack_topbottom, cutblack_leftright=cutblack_leftright)
    na *= n1
    nb *= n2
    x0a += xy1[0]
    y0a += xy1[1]
    x0b += xy2[0]
    y0b += xy2[1]

    threshold_m1m2_ratio = float(options.threshold_m1m2_ratio)
    H_ = compare(img1_, img2_,                                    # 已切割和缩小图像对的(B->A)透视矩阵
                 options.imgmatch,
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

    if options.addto_db:
        db = database.Database('data/imagery.db')
        assert H_orig.shape == (3,3)
        H_blob = H_orig.astype(np.float64).tobytes()
        fid1, iid1 = import_img.import_img(db, options.img1)
        fid2, iid2 = import_img.import_img(db, options.img2)
        # TODO: 检查数据库里是否已存在，避免UNIQUE约束错误
        db.insert_match(iid1, iid2, H_blob, b_in_a.wkt)
        db.commit()
        db.close()

    if options.img3d is not None:
        create_rb3dview(img1[:32766, :32766], img2[:32766,:32766], H_orig, options.img3d) #32766是由于SHRT_MAX限制
