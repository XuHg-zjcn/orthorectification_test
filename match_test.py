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
import math
import cv2
from osgeo import gdal
import numpy as np
from optparse import OptionParser


# 经测试能读取SPOT1-5卫星影像的四角坐标
def read_metadata(path):
    img = gdal.Open(path)
    gcps = img.GetGCPs()
    img.Close()
    if len(gcps) >= 4:
        gcps = np.array(list(map(lambda x:(x.GCPPixel, x.GCPLine, x.GCPX, x.GCPY), gcps)))
        H, mask = cv2.findHomography(gcps[:,:2], gcps[:,2:])  # 数据坐标到地理坐标的投影矩阵
        return gcps, H
    return None

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

# 计算图像B透视投影到图像A中与图像A的交集在图像A的坐标系下的包围框
def perspective_boundingbox(H, widthA, heightA, widthB, heightB):
    conrerB = np.array([[0, 0, 1],
                        [0, heightB, 1],
                        [widthB, heightB, 1],
                        [widthB, 0, 1]])
    conrerB_at_coordA = np.matmul(H, conrerB.transpose())
    conrerB_at_coordA /= conrerB_at_coordA[-1]
    xmin_a = 0
    xMax_a = widthA
    ymin_a = 0
    yMax_a = heightA
    xmin_b = np.min(conrerB_at_coordA[0])
    xMax_b = np.max(conrerB_at_coordA[0])
    ymin_b = np.min(conrerB_at_coordA[1])
    yMax_b = np.max(conrerB_at_coordA[1])
    xmin = max(xmin_a, xmin_b)
    xMax = min(xMax_a, xMax_b)
    ymin = max(ymin_a, ymin_b)
    yMax = min(yMax_a, yMax_b)
    return int(xmin), int(ymin), int(xMax), int(yMax)

# 以合适的整数倍缩小图像
def auto_zoom(img, maxpixel=1e6):
    if img.shape[0]*img.shape[1] > maxpixel:
        n = math.ceil(math.sqrt(img.shape[0]*img.shape[1]/maxpixel))
        img_ = cv2.resize(img, None, None, 1.0/n, 1.0/n, cv2.INTER_AREA)
    else:
        n = 1
        img_ = img
    return img_, n

def preprocess(img, nz):
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE if nz >= 5 else cv2.MORPH_RECT, (nz, nz))
    img = cv2.Laplacian(img, cv2.CV_8U)
    img = cv2.dilate(img, kern)
    img = cv2.resize(img, None, None, 1.0/nz, 1.0/nz, cv2.INTER_AREA)
    return img

# 对比两张图片的对应点
def compare(img1, img2, outpath_match=None, maxpoints=2000, threshold_m1m2_ratio=0.8):
    # 部分代码由deepseek给出
    sift = cv2.SIFT_create(maxpoints)
    print(f'img1.shape = {img1.shape}')
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    print(f'found {len(keypoints1)} keypoints in img1')
    print(f'img2.shape = {img1.shape}')
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
    # TODO: 添加更多命令行参数
    options, args = parser.parse_args()
    # TODO: 使用GDAL读取遥感影像
    img1 = cv2.imread(options.img1, cv2.IMREAD_GRAYSCALE)#[1400:-2000]  # TODO: 自动检测胶片边缘
    img2 = cv2.imread(options.img2, cv2.IMREAD_GRAYSCALE)#[1400:-2000]

    paraA = read_metadata(options.img1)
    paraB = read_metadata(options.img2)
    hasMetadata = False
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

    maxpixel_sift = int(float(options.maxpixel_sift))
    if options.laplace:
        dilsize = int(options.dilsize)
        maxpix1 = maxpixel_sift*dilsize**2
    else:
        maxpix1 = maxpixel_sift
    img1_, n1 = auto_zoom(img1, maxpixel=maxpix1)
    img2_, n2 = auto_zoom(img2, maxpixel=maxpix1)
    print(f'downsamp for input img1 in {n1}, img2 in {n2}')
    if options.laplace:
        img1_ = preprocess(img1_, dilsize)
        img2_ = preprocess(img2_, dilsize)
        n1 *= dilsize
        n2 *= dilsize
        print(f'Laplacian, dilate and downsamp in {dilsize}')
    threshold_m1m2_ratio = float(options.threshold_m1m2_ratio)
    H_ = compare(img1_, img2_,                                    # 已切割和缩小图像对的(B->A)透视矩阵
                 options.imgmatch,
                 maxpoints=10000,
                 threshold_m1m2_ratio=threshold_m1m2_ratio)
    H_cut = H_transpose(H_, zoom_d=n1, zoom_s=n2)                 # 已切割图像对的(B->A)透视矩阵
    if hasMetadata and bbox_at_coordA is not None and bbox_at_coordB is not None:
        H_orig = H_transpose(                                     # 原图像对的(B->A)透视矩阵
            H_,
            x0_d=bbox_at_coordA[0], y0_d=bbox_at_coordA[1], zoom_d=n1,
            x0_s=bbox_at_coordB[0], y0_s=bbox_at_coordB[1], zoom_s=n2)
        print('mearused perspective matrix:\n', H_orig)
    else:
        print('mearused perspective matrix:\n', H_cut)

    if options.img3d is not None:
        create_rb3dview(img1[:32766, :32766], img2[:32766,:32766], H_cut, options.img3d) #32766是由于SHRT_MAX限制
