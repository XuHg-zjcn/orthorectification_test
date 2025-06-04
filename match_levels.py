#!/usr/bin/env python3
#############################################################################
# 遥感影像跨等级匹配（低分辨率与高分辨率影像匹配）
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
import os
import functools
import bisect
import numpy as np
import cv2
import shapely
import database
import import_img
from common import shapely_perspective, findPerspective
from preprocess_single import preprocess
from imgmatch import compare


def plusminus(nmax, n0, plusmax=None, minumax=None):
    assert 0 <= n0 < nmax
    yield n0
    pm = 1
    while True:
        has_yield = False
        if n0+pm < nmax and (plusmax is None or pm<plusmax):
            has_yield = True
            yield n0+pm
        if n0-pm >= 0 and (minumax is None or pm<minumax):
            has_yield = True
            yield n0-pm
        if not has_yield:
            return
        pm += 1


def get_corners_coord(geom):
    lon_c = geom.centroid.x
    lat_c = geom.centroid.y

    coords = list(geom.exterior.coords)
    coord_nw = coords[0] # XXX : 不应该按顺序确定四角，应该用几何关系确定
    coord_ne = coords[1]
    coord_se = coords[2]
    coord_sw = coords[3]
    return coord_nw, coord_ne, coord_sw, coord_se


def compare_to(imgA_cut, pathB, name, ratio):
    print(f'compare with {pathB}')
    try:
        imgB = cv2.imread(pathB, cv2.IMREAD_GRAYSCALE)
        imgB_, nB, xyB = preprocess(imgB, 'imgB',
                            maxpixel_out=None,
                            predown=round(ratio/8),
                            laplace=True,
                            dilsize=8,
                            cutblack_topbottom=True,
                            cutblack_leftright=True)
        H_, matchs = compare(imgA_cut, imgB_,
                            f'data/match_levels_{name}.jpg',
                            maxpoints1=2000,
                            maxpoints2=2000,
                            threshold_m1m2_ratio=0.85)
    except:
        pass
    else:
        print(H_, len(matchs))
        if len(matchs) > 12:
            return H_, matchs
        else:
            return None


# 目前只支持KH-9低分辨率相机和高分辨率相机的影像匹配
# TODO: 添加其他传感器支持
if __name__ == '__main__':
    db = database.Database('data/imagery.db')
    # TODO: 添加命令行参数解析
    fid, iid = import_img.import_img(db, sys.argv[1])
    geom = shapely.from_wkt(db.get_frame_geom(fid))
    coord_nw, coord_ne, coord_sw, coord_se = get_corners_coord(geom)
    print(coord_nw, coord_ne, coord_sw, coord_se)

    imgA = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    imgA_, nA, xyA = preprocess(imgA, 'imgA',
                                maxpixel_out=None,
                                predown=2,
                                laplace=True,
                                dilsize=8,
                                cutblack_topbottom=True,
                                cutblack_leftright=True)

    print('imgA.shape', imgA.shape)
    print('imgA_.shape', imgA_.shape)
    A_height, A_width = imgA_.shape
    H = findPerspective(0, 0, A_width, A_height,
                        coord_ne, coord_se, coord_sw, coord_nw)
    Hinv = np.linalg.inv(H)

    for fid, name, poly, cpoint in db.get_intersect(fid):
        cpoint_inA = shapely_perspective(cpoint, Hinv)
        print(fid, name, cpoint, cpoint_inA)
        lst = db.get_splited_image(fid)
        cum_width = np.cumsum(list(map(lambda x:x[2], lst)))
        total_width = int(cum_width[-1])
        mean_height = sum(map(lambda x:x[3], lst))/len(lst)
        coord_nw, coord_ne, coord_sw, coord_se = get_corners_coord(poly)
        if 'F' in name:
            Hb = findPerspective(0, 0, total_width, mean_height,
                                 coord_nw, coord_ne, coord_se, coord_sw)
        elif 'A' in name:
            Hb = findPerspective(0, 0, total_width, mean_height,
                                 coord_se, coord_sw, coord_nw, coord_ne)
        else:
            print('unkown camera type')
            continue
        Hbinv = np.linalg.inv(Hb)
        cpoint_inB = shapely_perspective(cpoint, Hbinv)
        nth_img = bisect.bisect(cum_width, cpoint_inB.x)-1
        print('cpoint_inB', cpoint_inB, nth_img)
        H_B_to_A = np.matmul(Hinv, Hb)
        H_B_to_A /= H_B_to_A[-1, -1]
        x1 = cum_width[nth_img]
        x2 = cum_width[nth_img+1]
        ymax = lst[nth_img][3]
        B_in_A = shapely_perspective(shapely.box(x1, 0, x2, ymax), H_B_to_A)
        xmin, ymin, xmax, ymax = B_in_A.bounds
        xdelta = max(100, xmax - xmin)
        ydelta = max(100, ymax - ymin)
        xmin = int(max(xmin-xdelta*0.2, 0))
        xmax = int(min(xmax+xdelta*0.2, A_width))
        ymin = int(max(ymin-ydelta*0.2, 0))
        ymax = int(min(ymax+ydelta*0.2, A_height))
        alpha = H_B_to_A[2,0]*(x1+x2)/2 + H_B_to_A[2,1]*ymax/2 + H_B_to_A[2,2]
        ratio = 1/np.sqrt(np.sum((H_B_to_A[:2,:2]/alpha)**2)/2)
        print(xmin, xmax, ymin, ymax, ratio)

        imgA_cut = imgA_[ymin:ymax, xmin:xmax]
        cv2.imwrite(f'data/cut_{name}.jpg', imgA_cut)

        for i in plusminus(len(lst), nth_img, 3, 3):
            pathB = lst[i][1].split('\n')[0]
            retval = compare_to(imgA_cut, pathB, os.path.basename(pathB), ratio)
            if retval is not None:
                # TODO: 对其他分幅进行匹配
                break
            print('--------------------------')
        # TODO: 匹配结果添加到数据库
        print('==========================')