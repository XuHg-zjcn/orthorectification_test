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
import argparse
import database
import numpy as np
from osgeo import gdal
from imgview import ImgView
from imgmatch2 import ImgMatch
import transform
import preprocess
from preprocess_seq import PreprocessWithEst
import common

# 全局变量
filter_iid_in_pair = None
pseq = None

def filter_imgpair(pairs):
    if filter_iid_in_pair is None:
        return pairs
    return list(filter(lambda x:(x[0] in filter_iid_in_pair or x[1] in filter_iid_in_pair), pairs))

def get_suggest_transforms(lst):
    # TODO: 返回数据来源'avg_filted', 'mid', 'avg', 路径节点序列等
    def generate_out(x, tfs):
        x = np.concatenate((x, tfs), axis=0)
        x = np.unique(x, axis=0)
        ones = np.ones(x.shape[0]).reshape((-1, 1))
        x = np.concatenate((x, ones), axis=1)
        x = x.reshape((-1, 3, 3))
        return x
    n = len(lst)
    if n <= 2:
        return list(map(lambda x:x.transform, lst))
    tfs = np.zeros((n, 8))
    weights = np.zeros((n,))
    for i in range(n):
        _, weight, transform = lst[i]
        tfs[i] = (transform/transform[-1,-1]).flatten()[:-1]
        weights[i] = weight
    avg = np.average(tfs, axis=0, weights=weights)
    mid = np.median(tfs, axis=0)
    mad = np.median(np.abs(tfs-mid), axis=0)
    z_score_m = np.sqrt(np.mean(((tfs-mid)/mad)**2, axis=1))

    tfs_filted = tfs[z_score_m<3]
    weights_filted = tfs[z_score_m<3]
    if len(tfs_filted) == 0:
        return generate_out([mid, avg], tfs)
    else:
        avg_filted = np.average(tfs_filted, axis=0, weights=weights_filted)
        return generate_out([avg_filted, mid, avg], tfs)


def match_img(pathA, pathB, iidA, iidB, estT_B_to_A):
    dsA = gdal.Open(pathA, gdal.GA_ReadOnly)
    dsB = gdal.Open(pathB, gdal.GA_ReadOnly)
    ivA = ImgView(dsA.GetRasterBand(1))
    ivB = ImgView(dsB.GetRasterBand(1))
    im = ImgMatch(ivA, ivB)
    im.set_estT(estT_B_to_A)
    im.setPreprocessSeq(pseq)
    im.setParam_compare(
        outpath_match=f'data/match_db_{iidA}_{iidB}.jpg',
        maxpoints1=5000,
        maxpoints2=5000,
        threshold_m1m2_ratio=0.85)
    pt, n_match = im.match()
    return im


def match_imgpair_autoest(db, iidA, iidB):
    result = db.get_matchways_more_tf_singlepair(iidB, iidA, maxlength=3)
    tf_suggest = get_suggest_transforms(result)
    for transform_np in tf_suggest[:3]:
        t_B_to_A = transform.PerspectiveTransform(transform_np)
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
        B_in_A = im.get_poly_B_in_A()
        A_in_B = im.get_poly_A_in_B()
        db.insert_replace_match(iidA, iidB, im.H, B_in_A.wkt, im.n_match)
        db.insert_replace_match(iidB, iidA, im.H.inv(), A_in_B.wkt, im.n_match)
        db.commit()
        return im


def intersect_matchs_in_db(db):
    imgpairs = db.get_imagepairs_need_match()
    imgpairs = filter_imgpair(imgpairs)
    n_total = len(imgpairs)
    n_sucess = 0
    print(f'found {n_total} imagepairs need matching in database')
    for iidA, iidB in imgpairs:
        print('===================')
        print(iidA, iidB)
        im = match_imgpair_autoest(db, iidA, iidB)
        if im is not None:
            n_sucess += 1
            continue
    print('===================')
    print(f'matched {n_total} imgpairs, {n_sucess} sucess, {n_total-n_sucess} failed')


def update_worst_match(db, maxpoints):
    imgpairs = db.get_worst_match(maxpoints)  # 不应该在这里添加获取n_points，可能程序运行时改变
    imgpairs = filter_imgpair(imgpairs)
    n_total = len(imgpairs)
    print(f'need re-matching {n_total} imgpairs')
    if n_total == 0:
        return
    n_update = 0
    for iidA, iidB in imgpairs:
        print('===================')
        tf_old, _, n_points, last_update = db.get_match(iidA, iidB)
        print(f'{iidA},{iidB}, orignal n_points={n_points}, last_update={last_update}')
        print(tf_old)
        pathA = db.get_img_path_byid(iidA)
        pathB = db.get_img_path_byid(iidB)
        result = db.get_matchways_more_tf_singlepair(iidB, iidA, maxlength=2)
        tf_suggest = get_suggest_transforms(result)
        tf_ests = [tf_old]
        tf_ests.extend(tf_suggest[:3])
        updated = False
        for tf in tf_ests:
            tf = transform.PerspectiveTransform(tf)
            im = common.try_func(match_img, pathA, pathB, iidA, iidB, tf)
            if im is None or im.n_match < 12:
                print('failed')
                continue
            print(f'matched {im.n_match}')
            if im.n_match >= n_points*1.1:
                print('replace old match')
                print(im.H)
                B_in_A = im.get_poly_B_in_A()
                A_in_B = im.get_poly_A_in_B()
                db.insert_replace_match(iidA, iidB, im.H, B_in_A.wkt, im.n_match)
                db.insert_replace_match(iidB, iidA, im.H.inv(), A_in_B.wkt, im.n_match)
                n_points = im.n_match
                db.commit()
                updated = True
        if updated:
            n_update += 1
    print('===================')
    print(f'matched {n_total} imgpairs, {n_update}({n_update/n_total:.2%}) updated')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--intersect_matchs_in_db', action='store_true')
    parser.add_argument('--update_worst_match', action='store_true')
    parser.add_argument('--worst_maxpoint', default=100, type=int)
    parser.add_argument('--delete_error_match', action='store_true')
    parser.add_argument('-r', '--recursive', action='store_true')
    parser.add_argument('--intersect_pyproj', action='store_true')
    parser.add_argument('--filter_iid_in_pair', default=None, type=str)
    # preprocess setting
    parser.add_argument('--extCoef', default=0.1, type=float)
    parser.add_argument('--extMin', default=10, type=int)
    parser.add_argument('--nX', default=8, type=int)
    parser.add_argument('--maxpixel', default=1e6, type=float)
    parser.add_argument('-wL', '--w_Laplace', default=1.0, type=float)
    parser.add_argument('-wR', '--w_Roberts', default=1.414, type=float)
    parser.add_argument('-wS', '--w_Sobel', default=0.53, type=float)
    args = parser.parse_args()
    db = database.Database('data/imagery.db')
    pseq = PreprocessWithEst.params_from_dict(vars(args), raise_unknown=False)
    if args.filter_iid_in_pair is not None:
        filter_iid_in_pair = set(map(lambda x:int(x), args.filter_iid_in_pair.split(',')))
    else:
        filter_iid_in_pair = None
    if args.update_worst_match:
        update_worst_match(db, args.worst_maxpoint)
    if args.intersect_matchs_in_db:
        intersect_matchs_in_db(db)
    db.close()
