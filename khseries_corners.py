#!/usr/bin/env python3
#############################################################################
# 解析USGS提供的KH系列卫星影像元数据
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
from osgeo import gdal

def read_corner(filename):
    s = {'Center Latitude dec',
        'Center Longitude dec',
        'NW Corner Lat dec',
        'NW Corner Long dec',
        'NE Corner Lat dec',
        'NE Corner Long dec',
        'SE Corner Lat dec',
        'SE Corner Long dec',
        'SW Corner Lat dec',
        'SW Corner Long dec'}
    d = {}
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            line = line.rstrip('\n')
            spt = line.split('\t')
            if len(spt) != 2:
                continue
            name, value = spt
            name = name.rstrip(' ')
            if name in s:
                d[name] = float(value)
    return d

if __name__ == '__main__':
    preview_path = sys.argv[1]
    metadata_path = sys.argv[2]
    out_path = sys.argv[3]
    conrner = read_corner(metadata_path)
    preview = gdal.Open(preview_path)
    width = preview.RasterXSize
    height = preview.RasterYSize
    GCPs = []
    GCPs.append(gdal.GCP(conrner['NW Corner Long dec'], conrner['NW Corner Lat dec'], 0, 0,     0     ))
    GCPs.append(gdal.GCP(conrner['NE Corner Long dec'], conrner['NE Corner Lat dec'], 0, width, 0     ))
    GCPs.append(gdal.GCP(conrner['SE Corner Long dec'], conrner['SE Corner Lat dec'], 0, width, height))
    GCPs.append(gdal.GCP(conrner['SW Corner Long dec'], conrner['SW Corner Lat dec'], 0, 0,     height))
    #GCPs.append(gdal.GCP(conrner['Center Longitude dec'], conrner['Center Latitude dec'], 0, width/2,     height/2))
    options = gdal.TranslateOptions(GCPs=GCPs, outputSRS='EPSG: 4326')
    gdal.Translate(out_path, preview, options=options)
    # 运行gdalwarp进行转换
