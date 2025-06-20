#!/usr/bin/env python3
#############################################################################
# 遥感影像重叠区域匹配
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
import math
import functools
import cv2
from osgeo import gdal
import numpy as np
import shapely
from common import shapely_perspective


# 透视矩阵在图像裁剪，缩放下的变换
# 此函数已弃用，请用Transform的复合运算代替
def H_transpose(H, x0_s=0, y0_s=0, x0_d=0, y0_d=0, zoom_s=1, zoom_d=1):
    Td = np.array([[zoom_d, 0,      x0_d],
                   [0,      zoom_d, y0_d],
                   [0,      0,      1   ]])

    Ts = np.array([[zoom_s, 0,      x0_s],
                   [0,      zoom_s, y0_s],
                   [0,      0,      1   ]])
    H2 = np.matmul(Td, np.matmul(H, np.linalg.inv(Ts)))
    H2 /= H2[-1, -1]
    return H2

def match_KNN_RANSAC(kpsA, kpsB, descA, descB, threshold_m1m2_ratio=0.8):
    if min(len(kpsA), len(kpsB)) < 4:
        return None, []
    flann = cv2.FlannBasedMatcher()
    try:
        matches = flann.knnMatch(descA, descB, k=2)
    except:
        return None, []
    matches_filterd = []
    match_points1 = []
    match_points2 = []
    for match1, match2 in matches:
        if match1.distance > threshold_m1m2_ratio*match2.distance:
            continue
        match_points1.append(kpsA[match1.queryIdx].pt)
        match_points2.append(kpsB[match1.trainIdx].pt)
        matches_filterd.append(match1)
    match_points1 = np.array(match_points1, dtype=np.float32)
    match_points2 = np.array(match_points2, dtype=np.float32)
    try:
        H, mask = cv2.findHomography(match_points2, match_points1, method=cv2.RANSAC, maxIters=10000, confidence=0.99)
    except:
        return None, []
    good_matches = [match for match, flag in zip(matches_filterd, mask) if flag]
    n_good = len(good_matches)
    return H, good_matches

def filter_sift_feat_by_polygon(poly, kps, desc):
    kp_filted = []
    desc_filted = []
    for i, kp in enumerate(kps):
        if poly.covers(shapely.points(kp.pt)):
            kp_filted.append(kp)
            desc_filted.append(desc[i])
    desc_filted = np.array(desc_filted)
    return kp_filted, desc_filted


# 对比两张图片的对应点
def compare(imgA, imgB,
            outpath_match=None,
            maxpoints1=2000, maxpoints2=2000,
            threshold_m1m2_ratio=0.8):
    # 部分代码由deepseek给出
    sift = cv2.SIFT_create()

    print(f'imgA.shape = {imgA.shape}')
    sift.setNFeatures(maxpoints1)
    kpsA, descA = sift.detectAndCompute(imgA, None)
    print(f'found {len(kpsA)} keypoints in imgA')

    print(f'imgB.shape = {imgB.shape}')
    sift.setNFeatures(maxpoints2)
    kpsB, descB = sift.detectAndCompute(imgB, None)
    print(f'found {len(kpsB)} keypoints in imgB')

    H1, good1 = match_KNN_RANSAC(kpsA, kpsB, descA, descB, threshold_m1m2_ratio)
    if H1 is None:
        return None, []
    print(f'first matched {len(good1)} keypoints')

    # 生成两张图片中对应区域的多边形
    height_A, width_A = imgA.shape
    height_B, width_B = imgB.shape
    rect_A = shapely.polygons([(0, 0), (width_A, 0), (width_A, height_A), (0, height_A)])
    rect_B = shapely.polygons([(0, 0), (width_B, 0), (width_B, height_B), (0, height_B)])
    poly_B_in_A = shapely_perspective(rect_B, H1)
    poly_A_in_B = shapely_perspective(rect_A, np.linalg.inv(H1))

    # 用多边形过滤特征点
    kpsA_filted, descA_filted = filter_sift_feat_by_polygon(poly_B_in_A, kpsA, descA)
    print(f'imgA {len(kpsA_filted)}/{len(kpsA)} keypoint filterd')
    kpsB_filted, descB_filted = filter_sift_feat_by_polygon(poly_A_in_B, kpsB, descB)
    print(f'imgB {len(kpsB_filted)}/{len(kpsB)} keypoint filterd')

    # 再次匹配特征向量
    H2, good2 = match_KNN_RANSAC(kpsA_filted, kpsB_filted, descA_filted, descB_filted, threshold_m1m2_ratio)
    if H2 is None:
        return None, []
    print(f'second matched {len(good2)} keypoints')

    if outpath_match is not None:
        img3 = cv2.drawMatches(imgA, kpsA_filted, imgB, kpsB_filted, good2, None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255))
        cv2.imwrite(outpath_match, img3)
    return H2, good2

# 创建红蓝3D图片
def create_rb3dview(imgA, imgB, H, outpath):
    # TODO: 将图片切成多个部分后再处理
    # FIXME: 由于目前使用GDAL+自定义ImgView类读取文件，此处会产生错误，待修复
    imgBw = cv2.warpPerspective(imgB, H, (imgA.shape[1], imgA.shape[0]))
    merge = cv2.merge([imgBw, imgBw, imgA]) # BGR顺序
    cv2.imwrite(outpath, merge)
    # 输出图片可能需要旋转来适合3D眼镜
