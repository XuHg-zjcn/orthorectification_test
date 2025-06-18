#!/usr/bin/env python3
#############################################################################
# 导入影像
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
import os
import re
import shapely
from osgeo import gdal
import database
import khseries_corners
import metadata

pattern_kh = r'^D\d?[A-Z]{0,2}\d{4}-\d+[A-Z]+\d+'
pattern_spot = r'\d{3}-\d{3}_S\d_\d{3}-\d{3}-\d_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_\w+-\d_\w_\w{2}_\w{2}'


def process_kh_fid(db, path, rpath, name):
    print('is KH-series imagery', name)
    fid = db.get_frame_by_name(name)
    if fid is not None:
        print(f'frame found in db, id={fid}')
    else:
        arquire_date = None
        geom = None
        filename_metadata = name+'.txt'
        dirs_metadata = list(filter(
            lambda x:os.path.isfile(os.path.join(x, filename_metadata)),
            [os.path.dirname(path), os.path.dirname(rpath)]))
        for dir_metadata in dirs_metadata:
            path_metadata = os.path.join(dir_metadata, filename_metadata)
            metadata = khseries_corners.read_metadata(path_metadata)
            if 'Entity ID' not in metadata or metadata['Entity ID'] != name:
                continue
            if arquire_date is None:
                arquire_date = metadata.get('Acquisition Date')
            if geom is None:
                geom = khseries_corners.get_shapely_polygon(metadata).wkt
        print('arquire_date:', arquire_date)
        fid = db.insert_frame(name, None, arquire_date, geom)
        print(f'frame insert to db, id={fid}')
    return fid

# TODO: 应该设置Frame类，每种传感器继承子类
def process_spot_fid(db, path, rpath, name):
    Sx = name[name.find('S')+1]
    print(f'is SPOT{Sx} imagery')
    fid = db.get_frame_by_name(name)
    if fid is not None:
        print(f'frame found in db, id={fid}')
    else:
        arquire_date = None
        retval = metadata.read_metadata(rpath)
        if retval is None:
            geom = None
        else:
            gcps, H = retval
            geom = shapely.Polygon(tuple(map(lambda x:(x[2],x[3]), gcps))).wkt
        fid = db.insert_frame(name, None, arquire_date, geom)
    return fid

def import_img(db, path):
    rpath = os.path.realpath(path)
    filename = os.path.basename(rpath)
    fid = None
    m = re.match(pattern_kh, filename)
    if m is not None:
        name = m.group(0)
        fid = process_kh_fid(db, path, rpath, name)
    m = re.search(pattern_spot, rpath)
    if m is not None:
        name = m.group(0)
        fid = process_spot_fid(db, path, rpath, name)
    if fid is None:
        return None
    size = os.stat(rpath).st_size
    ds = gdal.Open(rpath)
    iid = db.get_img(rpath, size, fid)
    if iid is None:
        iid = db.insert_img(fid, [rpath], size, ds.RasterXSize, ds.RasterYSize)
        print(f'image insert to db, id={iid}')
    else:
        print(f'image already in db, id={iid}')
    print()
    ds.Close()
    return fid, iid

if __name__ == '__main__':
    db = database.Database('data/imagery.db')
    paths = sys.argv[1:]
    for path in paths:
        import_img(db, path)
    db.commit()
    db.close()