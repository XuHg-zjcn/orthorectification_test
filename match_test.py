#!/usr/bin/env python3
#############################################################################
# 遥感影像立体图像对齐测试程序
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
import cv2
import numpy as np
from optparse import OptionParser


# 变化投影矩阵
def H_zoom(H, zoom):
    H2 = H.copy()
    H2[2,0:2] /= zoom
    H2[0:2,2] *= zoom
    return H2

# 对比两张图片的对应点
def compare(img1, img2, outpath_match=None):
    # 部分代码由deepseek给出
    sift = cv2.SIFT_create(20000)
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    print(f'found {len(keypoints1)} keypoints in img1')
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    print(f'found {len(keypoints2)} keypoints in img2')
    flann = cv2.FlannBasedMatcher()
    matches = flann.match(descriptors1, descriptors2)
    match_points1 = []
    match_points2 = []
    for match in matches:
        match_points1.append(keypoints1[match.queryIdx].pt)
        match_points2.append(keypoints2[match.trainIdx].pt)
    match_points1 = np.array(match_points1, dtype=np.float32)
    match_points2 = np.array(match_points2, dtype=np.float32)
    H, mask = cv2.findHomography(match_points2, match_points1, method=cv2.RANSAC, maxIters=2000, confidence=0.99)
    good_matches = [match for match, flag in zip(matches, mask) if flag]
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


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-a', '--img1', dest='img1', help='first input image path')
    parser.add_option('-b', '--img2', dest='img2', help='second input image path')
    parser.add_option('-m', '--imgmatch', dest='imgmatch', help='output draw match image path')
    parser.add_option('-3', '--img3d', dest='img3d', help='output red-cyan 3D image path')
    options, args = parser.parse_args()
    img1 = cv2.imread(options.img1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(options.img2, cv2.IMREAD_GRAYSCALE)
    img1_ = cv2.resize(img1, None, None, 1.0/4, 1.0/4, cv2.INTER_AREA)
    img2_ = cv2.resize(img2, None, None, 1.0/4, 1.0/4, cv2.INTER_AREA)
    H_4 = compare(img1_, img2_, options.imgmatch)
    H = H_zoom(H_4, 4)
    print(H)
    if options.img3d is not None:
        create_rb3dview(img1[:32766, :32766], img2[:32766,:32766], H, options.img3d) #32766是由于SHRT_MAX限制
