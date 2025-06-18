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
from datetime import datetime
import shapely

def read_metadata(filename):
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
            # 尝试以整数形式解析
            try:
                value_i = int(value)
            except Exception as e:
                pass
            else:
                d[name] = value_i
                continue
            # 尝试以浮点数形式解析
            try:
                value_f = float(value)
            except Exception as e:
                pass
            else:
                d[name] = value_f
                continue
            # 尝试以日期形式解析
            try:
                value_d = datetime.strptime(value, '%Y/%m/%d')
            except Exception as e:
                pass
            else:
                d[name] = value_d
                continue
            # 直接以字符串储存
            d[name] = value
    return d

def get_shapely_polygon(d):
    name_lst = [('NW Corner Lat dec', 'NW Corner Long dec'),
                ('NE Corner Lat dec', 'NE Corner Long dec'),
                ('SE Corner Lat dec', 'SE Corner Long dec'),
                ('SW Corner Lat dec', 'SW Corner Long dec')]
    if 'NW Cormer Lat dec' in d and 'NW Corner Lat dec' not in d:
        # Declass 1元数据有拼写错误
        d['NW Corner Lat dec'] = d['NW Cormer Lat dec']
    points = list(map(lambda x:(d[x[1]],d[x[0]]), name_lst))
    return shapely.polygons(points)

if __name__ == '__main__':
    preview_path = sys.argv[1]
    metadata_path = sys.argv[2]
    out_path = sys.argv[3]
    conrner = read_metadata(metadata_path)
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
