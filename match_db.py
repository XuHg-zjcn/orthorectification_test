#!/usr/bin/env python3
#############################################################################
# 基于数据库的遥感影像匹配
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
import database
from osgeo import gdal
from imgview import ImgView
from imgmatch2 import ImgMatch
import preprocess
import common


def match_img(pathA, pathB, iidA, iidB, estT_B_to_A):
    dsA = gdal.Open(pathA, gdal.GA_ReadOnly)
    dsB = gdal.Open(pathB, gdal.GA_ReadOnly)
    ivA = ImgView(dsA.GetRasterBand(1))
    ivB = ImgView(dsB.GetRasterBand(1))
    im = ImgMatch(ivA, ivB)
    im.append_pobj(preprocess.AutoCutEstTf('A', estT_B_to_A, extCoef=0, extMin=100))
    im.append_pobj(preprocess.AutoCutEstTf('B', estT_B_to_A, extCoef=0, extMin=100))
    im.append_pobj(preprocess.AutoZoomEstTf('A', estT_B_to_A, nX=8))
    im.append_pobj(preprocess.AutoZoom('A', maxpixel=1e6*8**2))
    im.append_pobj(preprocess.AutoZoomEstTf('B', estT_B_to_A, nX=8))
    im.append_pobj(preprocess.AutoZoom('B', maxpixel=1e6*8**2))
    if im.imgA.shape[0]*im.imgA.shape[1] < 1e6:
        return
    if im.imgB.shape[0]*im.imgB.shape[1] < 1e6:
        return
    im.append_pobj(preprocess.CutBlackTopBottom('A'))
    im.append_pobj(preprocess.CutBlackLeftRight('A'))
    im.append_pobj(preprocess.EdgeDetection('A'))
    im.append_pobj(preprocess.DilateAndDownsamp('A', nz=8))
    im.append_pobj(preprocess.CutBlackTopBottom('B'))
    im.append_pobj(preprocess.CutBlackLeftRight('B'))
    im.append_pobj(preprocess.EdgeDetection('B'))
    im.append_pobj(preprocess.DilateAndDownsamp('B', nz=8))
    im.append_pobj(preprocess.AutoCutEstTf('A', estT_B_to_A, extCoef=0, extMin=10))
    im.append_pobj(preprocess.AutoCutEstTf('B', estT_B_to_A, extCoef=0, extMin=10))
    im.setParam_compare(
        outpath_match=f'data/match_db_{iidA}_{iidB}.jpg',
        maxpoints1=5000,
        maxpoints2=5000,
        threshold_m1m2_ratio=0.85)
    pt, n_match = im.match()
    return im


if __name__ == '__main__':
    # TODO: 添加命令行参数解析
    db = database.Database('data/imagery.db')
    imgpairs = db.get_imagepairs_need_match()
    n_total = len(imgpairs)
    n_sucess = 0
    print(f'found {n_total} imagepairs need matching in database')
    for iidA, iidB in imgpairs:
        print('===================')
        for iidC in db.get_matchways(iidA, iidB):
            print(iidA, iidB, iidC)
            t_A_to_C = db.get_match_tranform_bidire(iidC, iidA)
            t_B_to_C = db.get_match_tranform_bidire(iidC, iidB)
            t_B_to_A = t_A_to_C.inv().fog(t_B_to_C)
            pathA = db.get_img_path_byid(iidA)
            pathB = db.get_img_path_byid(iidB)
            print(pathA)
            print(pathB)
            print(t_B_to_A)
            im = common.try_func(match_img, pathA, pathB, iidA, iidB, t_B_to_A)
            if im is None or im.H is None:
                print('match failed')
                continue
            if im.n_match < 12:
                print(f'{im.n_match} points match, too few')
                continue
            print(f'{im.n_match} points match')
            print(im.H)
            n_sucess += 1
            break
    print('===================')
    print(f'matched {n_total} imgpairs, {n_sucess} sucess, {n_total-n_sucess} failed')
