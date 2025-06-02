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
from common import shaply_proj


# 透视矩阵在图像裁剪，缩放下的变换
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

def match_KNN_RANSAC(kps1, kps2, desc1, desc2, threshold_m1m2_ratio=0.8):
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(desc1, desc2, k=2)
    matches_filterd = []
    match_points1 = []
    match_points2 = []
    for match1, match2 in matches:
        if match1.distance > threshold_m1m2_ratio*match2.distance:
            continue
        match_points1.append(kps1[match1.queryIdx].pt)
        match_points2.append(kps2[match1.trainIdx].pt)
        matches_filterd.append(match1)
    match_points1 = np.array(match_points1, dtype=np.float32)
    match_points2 = np.array(match_points2, dtype=np.float32)
    H, mask = cv2.findHomography(match_points2, match_points1, method=cv2.RANSAC, maxIters=10000, confidence=0.99)
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
def compare(img1, img2,
            outpath_match=None,
            maxpoints1=2000, maxpoints2=2000,
            threshold_m1m2_ratio=0.8):
    # 部分代码由deepseek给出
    sift = cv2.SIFT_create()

    print(f'img1.shape = {img1.shape}')
    sift.setNFeatures(maxpoints1)
    kps1, desc1 = sift.detectAndCompute(img1, None)
    print(f'found {len(kps1)} keypoints in img1')

    print(f'img2.shape = {img2.shape}')
    sift.setNFeatures(maxpoints2)
    kps2, desc2 = sift.detectAndCompute(img2, None)
    print(f'found {len(kps2)} keypoints in img2')

    H1, good1 = match_KNN_RANSAC(kps1, kps2, desc1, desc2, threshold_m1m2_ratio)
    print(f'first matched {len(good1)} keypoints')

    # 生成两张图片中对应区域的多边形
    height_a, width_a = img1.shape
    height_b, width_b = img2.shape
    rect_a = shapely.polygons([(0, 0), (width_a, 0), (width_a, height_a), (0, height_a)])
    rect_b = shapely.polygons([(0, 0), (width_b, 0), (width_b, height_b), (0, height_b)])
    poly_b_in_a = shapely.transform(rect_b, functools.partial(shaply_proj, H1))
    poly_a_in_b = shapely.transform(rect_a, functools.partial(shaply_proj, np.linalg.inv(H1)))

    # 用多边形过滤特征点
    kps1_filted, desc1_filted = filter_sift_feat_by_polygon(poly_b_in_a, kps1, desc1)
    print(f'img1 {len(kps1_filted)}/{len(kps1)} keypoint filterd')
    kps2_filted, desc2_filted = filter_sift_feat_by_polygon(poly_a_in_b, kps2, desc2)
    print(f'img2 {len(kps2_filted)}/{len(kps2)} keypoint filterd')

    # 再次匹配特征向量
    H2, good2 = match_KNN_RANSAC(kps1_filted, kps2_filted, desc1_filted, desc2_filted, threshold_m1m2_ratio)
    print(f'second matched {len(good2)} keypoints')

    if outpath_match is not None:
        img3 = cv2.drawMatches(img1, kps1_filted, img2, kps2_filted, good2, None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255))
        cv2.imwrite(outpath_match, img3)
    return H2

# 创建红蓝3D图片
def create_rb3dview(img1, img2, H, outpath):
    # TODO: 将图片切成多个部分后再处理
    img2w = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
    merge = cv2.merge([img2w, img2w, img1]) # BGR顺序
    cv2.imwrite(outpath, merge)
    # 输出图片可能需要旋转来适合3D眼镜
