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
import cv2
from osgeo import gdal
import numpy as np


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

# 对比两张图片的对应点
def compare(img1, img2, outpath_match=None, maxpoints=2000, threshold_m1m2_ratio=0.8):
    # 部分代码由deepseek给出
    sift = cv2.SIFT_create(maxpoints)
    print(f'img1.shape = {img1.shape}')
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    print(f'found {len(keypoints1)} keypoints in img1')
    print(f'img2.shape = {img2.shape}')
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    print(f'found {len(keypoints2)} keypoints in img2')
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    matches_filterd = []
    match_points1 = []
    match_points2 = []
    for match1, match2 in matches:
        if match1.distance > threshold_m1m2_ratio*match2.distance:
            continue
        match_points1.append(keypoints1[match1.queryIdx].pt)
        match_points2.append(keypoints2[match1.trainIdx].pt)
        matches_filterd.append(match1)
    match_points1 = np.array(match_points1, dtype=np.float32)
    match_points2 = np.array(match_points2, dtype=np.float32)
    H, mask = cv2.findHomography(match_points2, match_points1, method=cv2.RANSAC, maxIters=10000, confidence=0.99)
    good_matches = [match for match, flag in zip(matches_filterd, mask) if flag]
    print(f'matched {len(good_matches)} keypoints')

    if outpath_match is not None:
        img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255))
        cv2.imwrite(outpath_match, img3)
    return H

# 创建红蓝3D图片
def create_rb3dview(img1, img2, H, outpath):
    # TODO: 将图片切成多个部分后再处理
    img2w = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
    merge = cv2.merge([img2w, img2w, img1]) # BGR顺序
    cv2.imwrite(outpath, merge)
    # 输出图片可能需要旋转来适合3D眼镜
