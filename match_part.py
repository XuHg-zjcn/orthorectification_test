#!/usr/bin/env python3
#############################################################################
# 分幅扫描遥感影像匹配
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
import numpy as np
from osgeo import gdal
import database
import import_img
from imgview import ImgView
import preprocess
from imgmatch2 import ImgMatch
from common import CropZoom2D


if __name__ == '__main__':
    db = database.Database('data/imagery.db')
    # TODO: 添加命令行参数解析
    d_img_in_frame = {}
    for path in sys.argv[1:]:
        res = import_img.import_img(db, path)
        if res is None:
            continue
        fid, iid = res
        t = (iid, path)
        if fid not in d_img_in_frame:
            d_img_in_frame[fid] = [t]
        else:
            d_img_in_frame[fid].append(t)
    print(f'{len(d_img_in_frame)} frames')
    print(f'{sum(map(lambda x:len(x), d_img_in_frame.values()))} images')
    print('=======start compare=======')
    for fid, iid_lst in d_img_in_frame.items():
        for (iidA, pathA), (iidB, pathB) in zip(iid_lst[:-1], iid_lst[1:]):
            print('compare')
            print(f'A: {pathA}')
            print(f'B: {pathB}')
            result_in_db = db.get_match(iidA, iidB)
            if result_in_db is not None:
                print('already in db')
                continue
            dsA = gdal.Open(pathA, gdal.GA_ReadOnly)
            ivA = ImgView(dsA.GetRasterBand(1))
            dsB = gdal.Open(pathB, gdal.GA_ReadOnly)
            ivB = ImgView(dsB.GetRasterBand(1))
            im = ImgMatch(ivA, ivB)
            czA = CropZoom2D.with_shape(ivA.shape)[::2, -3000::2]
            czB = CropZoom2D.with_shape(ivB.shape)[::2, :3000:2]
            im.append_pobj(preprocess.CutImg('A', czA))
            im.append_pobj(preprocess.AutoZoom('A', predown=1))
            im.append_pobj(preprocess.CutImg('B', czB))
            im.append_pobj(preprocess.AutoZoom('B',predown=1))
            im.setParam_compare(outpath_match=f'data/match_{iidA}_{iidB}.jpg',
                                maxpoints1=5000, maxpoints2=5000,
                                threshold_m1m2_ratio=0.8)
            H_orig, n_match = im.match()
            if H_orig is None:
                print('match failed')
                continue
            print(H_orig)
            poly_B_in_A = im.get_poly_B_in_A()
            db.insert_match(iidA, iidB, H_orig, poly_B_in_A.wkt)
            db.commit()
            print('--------------------------')
        print('==========================')
    db.close()
