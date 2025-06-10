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
import argparse
import numpy as np
import cv2
from osgeo import gdal
import shapely
import database
import import_img
from common import shapely_perspective, findPerspective, try_func, CropZoom2D
import preprocess
from imgmatch import H_transpose
from imgmatch2 import ImgMatch
from imgview import ImgView


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


def compare_to(
        imgA_, cut_A,
        pathB, name, ratio,
        maxpointA=2000, maxpointB=2000,
        threshold_m1m2_ratio=0.85):
    print(f'compare with {pathB}')
    dsB = gdal.Open(pathB, gdal.GA_ReadOnly)
    ivB = ImgView(dsB.GetRasterBand(1))
    im = ImgMatch(imgA_, ivB)
    im.append_preprocessA(functools.partial(preprocess.cut, cz2d=cut_A))
    im.append_preprocessB(functools.partial(preprocess.auto_zoom, predown=round(ratio/8)))
    im.append_preprocessB(functools.partial(preprocess.laplacian_and_dilate, nz=8))
    im.append_preprocessB(functools.partial(preprocess.cutblack_topbottom, name='imgB'))
    im.append_preprocessB(functools.partial(preprocess.cutblack_leftright, name='imgB'))
    im.setParam_compare(
        outpath_match=f'data/match_levels_{name}.jpg',
        maxpoints1=maxpointA,
        maxpoints2=maxpointB,
        threshold_m1m2_ratio=threshold_m1m2_ratio)
    H_B_to_Ap, n_match = im.match()
    return H_B_to_Ap, n_match

def calc_H_Frame_to_geo(lst, name, poly):
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
        return None
    return Hb

def choiceImagePartInFrame(cpoint_inF, lst):
    cum_width = np.cumsum(list(map(lambda x:x[2], lst)))
    total_width = int(cum_width[-1])
    mean_height = sum(map(lambda x:x[3], lst))/len(lst)
    nth_img = bisect.bisect(cum_width, cpoint_inF.x)-1
    print('cpoint in Frame', cpoint_inF, nth_img)
    x1 = cum_width[nth_img]
    x2 = cum_width[nth_img+1]
    ymax = lst[nth_img][3]
    poly_B_in_F = shapely.box(x1, 0, x2, ymax)
    return nth_img, poly_B_in_F

def calcBox_BinA(H_F_to_A, A_shape, poly_B_in_F, extRangeA=0.2):
    B_in_A = shapely_perspective(poly_B_in_F, H_F_to_A)
    xmin, ymin, xmax, ymax = B_in_A.bounds
    xdelta = max(100, xmax - xmin)*extRangeA
    ydelta = max(100, ymax - ymin)*extRangeA
    A_height, A_width = A_shape
    xmin = int(max(xmin-xdelta, 0))
    xmax = int(min(xmax+xdelta, A_width))
    ymin = int(max(ymin-ydelta, 0))
    ymax = int(min(ymax+ydelta, A_height))
    x1, y1, x2, y2 = poly_B_in_F.bounds
    alpha = H_F_to_A[2,0]*(x1+x2)/2 + H_F_to_A[2,1]*(y1+y2)/2 + H_F_to_A[2,2]
    ratio = 1/np.sqrt(np.sum((H_F_to_A[:2,:2]/alpha)**2)/2)
    return CropZoom2D(x0=xmin, y0=ymin, x1=xmax, y1=ymax, nz=1), ratio

def calc_bbox_withH(D_shape, S_shape, H):
    # 不应该在此阶段估计bbox，图像预处理后会切除边框，在预处理后再进行估计更准确
    # 需在`ImgMatch`类实现`match_with_estH`方法
    S_height, S_width = S_shape
    D_height, D_width = D_shape
    box_S = shapely.box(0, 0, S_width, S_height)
    S_in_D = shapely_perspective(box_S, H)
    box_D = shapely.box(0, 0, D_width, D_height)
    if not shapely.intersects(S_in_D, box_D):
        return None
    xmin, ymin, xmax, ymax = S_in_D.bounds
    xmin = int(max(xmin, 0))
    xmax = int(min(xmax, D_width))
    ymin = int(max(ymin, 0))
    ymax = int(min(ymax, D_height))
    alpha = H[2,0]*D_width/2 + H[2,1]*D_height/2 + H[2,2]
    ratio = 1/np.sqrt(np.sum((H[:2,:2]/alpha)**2)/2)
    return CropZoom2D(x0=xmin, y0=ymin, x1=xmax, y1=ymax, nz=1), ratio

def match_other_oneway(db, imgA_, H_B_to_Ap, iB, lst, irange, **kwargs):
    H_C_to_Ap = H_B_to_Ap
    iC = iB
    for iD in irange:
        iidC = lst[iC][0]
        iidD = lst[iD][0]
        pathD = lst[iD][1].split('\n')[0]
        nameD = os.path.basename(pathD)
        D_shape = (lst[iD][3], lst[iD][2])
        H_D_to_C = db.get_match_tranform_bidire(iidC, iidD)
        if H_D_to_C is None:
            break
        H_D_to_Ap_est = np.matmul(H_C_to_Ap, H_D_to_C)
        H_D_to_Ap_est /= H_D_to_Ap_est[-1, -1]
        res = calc_bbox_withH(imgA_.shape, D_shape, H_D_to_Ap_est)
        if res is None:
            break
        cut_A, ratio = res
        H_D_to_Ap, n_match = compare_to(imgA_, cut_A, pathD, nameD+'e', ratio, **kwargs)
        print(nameD, n_match)
        if H_D_to_Ap is None:
            break
        else:
            yield iidD, H_D_to_Ap, n_match
        iC = iD
        H_C_to_Ap = H_D_to_Ap

def match_other(db, imgA_, H_B_to_Ap, iB, lst, **kwargs):
    # TODO: 按照现有的变换矩阵（粗略）重新匹配一遍并yield
    print('match other forward')
    for res in match_other_oneway(db, imgA_, H_B_to_Ap, iB, lst, range(iB+1, len(lst)), **kwargs):
        yield res
    print('match other backward')
    for res in match_other_oneway(db, imgA_, H_B_to_Ap, iB, lst, range(iB-1, -1, -1), **kwargs):
        yield res


# 目前只支持KH-9低分辨率相机和高分辨率相机的影像匹配
# TODO: 添加其他传感器支持
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predownA', default=2, type=int)
    parser.add_argument('--minmatch', default=12, type=int)
    parser.add_argument('--extRangeA', default=0.2, type=float)
    parser.add_argument('--plusminus', default=3, type=int)
    parser.add_argument('--maxpointA', default=2000, type=int)
    parser.add_argument('--maxpointB', default=2000, type=int)
    parser.add_argument('--threshold_m1m2_ratio', default=0.85, type=float)
    parser.add_argument('imgA', nargs=1)
    args = parser.parse_args()
    db = database.Database('data/imagery.db')
    pathA = args.imgA[0]
    fidA, iidA = import_img.import_img(db, pathA)
    geom = shapely.from_wkt(db.get_frame_geom(fidA))
    coord_nw, coord_ne, coord_sw, coord_se = get_corners_coord(geom)
    print(coord_nw, coord_ne, coord_sw, coord_se)

    dsA = gdal.Open(pathA, gdal.GA_ReadOnly)
    ivA = ImgView(dsA.GetRasterBand(1))
    imgA_, tA = preprocess.auto_zoom(ivA, predown=args.predownA)
    imgA_, t_ = preprocess.cutblack_topbottom(imgA_, name='imgA')
    tA = tA.fog(t_)
    imgA_, t_ = preprocess.cutblack_leftright(imgA_, name='imgA')
    tA = tA.fog(t_)
    imgA_, t_ = preprocess.laplacian_and_dilate(imgA_, nz=8)
    tA = tA.fog(t_)

    print('ivA.shape', ivA.shape)
    print('imgA_.shape', imgA_.shape)
    A_height, A_width = imgA_.shape
    H_A_to_geo = findPerspective(0, 0, A_width, A_height,
                                coord_ne, coord_se, coord_sw, coord_nw)
    H_geo_to_A = np.linalg.inv(H_A_to_geo)

    for fid, name, poly, cpoint in db.get_intersect(fidA):
        lst = db.get_splited_image(fid)
        H_F_to_geo = calc_H_Frame_to_geo(lst, name, poly)
        if H_F_to_geo is None:
            continue
        H_geo_to_F = np.linalg.inv(H_F_to_geo)
        H_F_to_A = np.matmul(H_geo_to_A, H_F_to_geo)
        cpoint_inF = shapely_perspective(cpoint, H_geo_to_F)
        nth_img, poly_B_in_F = choiceImagePartInFrame(cpoint_inF, lst)
        cut_A, ratio = calcBox_BinA(H_F_to_A, imgA_.shape, poly_B_in_F, args.extRangeA)
        print(cut_A, ratio)

        H_B_to_Ap = None
        iB = None  # iB是F中选中的分幅序号
        for i in plusminus(len(lst), nth_img, args.plusminus, args.plusminus):
            pathB = lst[i][1].split('\n')[0]
            retval = try_func(compare_to, imgA_, cut_A, pathB, os.path.basename(pathB), ratio,
                             maxpointA=args.maxpointA,
                             maxpointB=args.maxpointB,
                             threshold_m1m2_ratio=args.threshold_m1m2_ratio)
            if retval is None:
                continue
            H_B_to_Ap, n_match = retval
            if n_match < args.minmatch:
                continue
            iB = i
            break
            print('--------------------------')
        if iB is None or H_B_to_Ap is None:
            continue
        for iidD, H_D_to_Ap, n_match in match_other(db, imgA_, H_B_to_Ap, iB, lst):
            if n_match < args.minmatch:
                continue
            H_D_to_A = H_transpose(H_D_to_Ap, x0_d=tA.x0, y0_d=tA.y0, zoom_d=tA.nz)
            db.insert_match(iidA, iidD, H_D_to_A, None)
            db.commit()
        print('==========================')
    db.close()