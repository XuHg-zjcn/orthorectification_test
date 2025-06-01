from osgeo import gdal
import cv2
import numpy as np


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
