#!/usr/bin/env python3.10
#############################################################################
# GDAL光栅图像自定义接口封装
# Copyright (C) 2024  Xu Ruijun
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
from scipy import interpolate
from osgeo import gdal, osr

class MyRaster:
    def __init__(self, ds):
        self.ds = ds
        self.band = ds.GetRasterBand(1)
        osrprj = osr.SpatialReference()
        osrprj.ImportFromWkt(ds.GetProjectionRef())
        osrgeo = osr.SpatialReference()
        osrgeo.SetWellKnownGeogCS("WGS84")
        self.ct_geo2prj = osr.CoordinateTransformation(osrgeo, osrprj)
        self.ct_prj2geo = osr.CoordinateTransformation(osrprj, osrgeo)
        self.gt = ds.GetGeoTransform()
        assert self.gt[2] == 0
        assert self.gt[4] == 0

    def geo2prj(self, lat, lon):
        return self.ct_geo2prj.TransformPoint(lat, lon)[:2]

    def prj2geo(self, x, y):
        return self.ct_prj2geo.TransformPoint(x, y)[:2]

    def rc2prj(self, col, row):
        x = self.gt[0] + self.gt[1]*col
        y = self.gt[3] + self.gt[5]*row
        return x, y

    def prj2rc(self, x, y):
        col = (x - self.gt[0])/self.gt[1]
        row = (y - self.gt[3])/self.gt[5]
        return col, row

    def geo2rc(self, lat, lon, crop=True):
        col, row = self.prj2rc(*self.geo2prj(lat, lon))
        if crop:
            col, row = self.crop_rc(col, row)
        return col, row

    def rc2geo(self, col, row):
        return self.prj2geo(*self.rc2prj(col, row))

    def crop_rc(self, col, row):
        if row < 0:
            row = 0
        elif row >= self.band.YSize-1:
            row = self.band.YSize-1
        if col < 0:
            col = 0
        elif col >= self.band.XSize-1:
            col = self.band.XSize-1
        return col, row

    def geo2rcbbox(self, lat0, lon0, lat1, lon1):
        c1, r1 = self.geo2rc(lat0, lon0)
        c2, r2 = self.geo2rc(lat1, lon1)
        c0 = int(min(c1, c2))
        r0 = int(min(r1, r2))
        w = int(abs(c1-c2))
        h = int(abs(r1-r2))
        return c0, r0, w, h

    def get_geopoint_cubic(self, lat, lon):
        col, row = self.geo2rc(lat, lon)
        area4x4 = self.band.ReadAsArray(int(col)-1, int(row)-1, 4, 4)
        assert area4x4.shape == (4, 4)
        xg, yg = np.meshgrid([-1,0,1,2], [-1,0,1,2])
        return interpolate.interp2d(xg, yg, area4x4, kind='cubic')(col%1, row%1)[0]

if __name__ == "__main__":
    ds = gdal.Open(sys.argv[1])
    raster = MyRaster(ds)
    print(raster.geo2rc(30, 120.5))
    print(raster.rc2geo(1000, 1000))
    band = ds.GetRasterBand(1)
    #print(band.ReadAsArray(band.YSize-100,band.YSize-100,10,10))
