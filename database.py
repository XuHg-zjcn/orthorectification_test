#!/usr/bin/env python3
#############################################################################
# 数据库接口
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
import sqlite3
import os
import shapely


class Database:
    def __init__(self, path='imagery.db'):
        conn = sqlite3.connect(path)
        conn.enable_load_extension(True)
        conn.execute('SELECT load_extension("mod_spatialite")')
        cursor = conn.cursor()
        cursor.execute('SELECT InitSpatialMetaData(1)')
        self.conn = conn
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        #ref: https://www.osgeo.cn/pygis/spatialite-create.html
        cursor.execute(  # 快门事件
            'CREATE TABLE IF NOT EXISTS frames('
            'id INTEGER PRIMARY KEY AUTOINCREMENT,'
            'name TEXT,'
            'platform TEXT,'
            'arquire_date DATE)'
        )
        cursor.execute("SELECT AddGeometryColumn('frames', 'geom', 4326, 'POLYGON', 2)")
        cursor.execute(  # 图像文件
            'CREATE TABLE IF NOT EXISTS images('
            'id INTEGER PRIMARY KEY AUTOINCREMENT,'
            'frameid INT,'  # 对应的快门事件，目前只支持一个
            'paths TEXT,'   # 文件的完整路径，用'\n'(换行符)分隔
            'size INT,'     # 相同大小和文件名视为同一图像文件
            'width INT,'
            'height INT,'
            'geom POLYGON,'
            'FOREIGN KEY (frameid) REFERENCES frames(id))'
        )
        cursor.execute(  # 图像匹配
            'CREATE TABLE IF NOT EXISTS matchs('
            'id INTEGER PRIMARY KEY AUTOINCREMENT,'
            'a_imgid INT,'
            'b_imgid INT,'
            'transfrom BLOB,'  # 投影矩阵，9个double值
            'area_b_in_a POLYGON,'
            'isChecked BOOL,'  # 是否经过人工检查
            'FOREIGN KEY (a_imgid) REFERENCES images(id),'
            'FOREIGN KEY (b_imgid) REFERENCES images(id),'
            'UNIQUE (a_imgid, b_imgid))'
        )
        cursor.execute(  # 特征
            'CREATE TABLE IF NOT EXISTS features('
            'id INTEGER PRIMARY KEY AUTOINCREMENT,'
            'imgid INT,'
            'type VARCHAR(256),'
            'preprocess TEXT,'
            'data BLOB,'
            'time INT,'
            'FOREIGN KEY (imgid) REFERENCES images(id))'
        )

    def insert_frame(self, name, platform, arquire_date, geom):
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO frames (name, platform, arquire_date, geom) VALUES(?,?,?,GeomFromText(?, 4326));',
            (name, platform, arquire_date, geom)
        )
        cursor.execute('SELECT last_insert_rowid() FROM frames LIMIT 1;')
        return next(cursor)[0]

    def get_frame_by_name(self, name):
        cursor = self.conn.cursor()
        cursor.execute('SELECT id FROM frames WHERE name=? LIMIT 1;', (name,))
        for idx in cursor:
            return idx[0]
        return None

    def get_frame_geom(self, id_):
        cursor = self.conn.cursor()
        cursor.execute('SELECT ST_AsText(geom) FROM frames WHERE id=? LIMIT 1;', (id_,))
        for geom in cursor:
            return geom[0]
        return None

    def insert_img(self, frameid, paths, size, width, height):
        paths_str = ''.join(map(lambda x:x+'\n', paths))
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO images (frameid, paths, size, width, height) VALUES(?,?,?,?,?);',
            (frameid, paths_str, size, width, height)
        )
        cursor.execute('SELECT last_insert_rowid() FROM frames LIMIT 1;')
        return next(cursor)[0]

    # 获取分幅扫描的KH影像文件
    def get_splited_image(self, fid):
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT id, paths, width, height '
            'FROM images '
            r"WHERE frameid=? AND paths LIKE '%\__.tif%' ESCAPE '\';",
            (fid,)
        )
        return list(cursor)

    def get_img(self, path, size):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id FROM images WHERE size=? AND paths LIKE ? LIMIT 1;",
            (size, '%\n'+path+'\n%'))
        for idx in cursor:
            return idx
        filename = os.path.basename(path)
        cursor.execute(
            "SELECT id, paths FROM images WHERE size=? AND paths LIKE ? LIMIT 1;",
            (size, '%/'+filename+'\n%'))
        for idx, paths in cursor:
            path_lst =  paths.split('\n')
            if any(map(lambda x:filename==os.path.basename(x), path_lst)):
                return idx

    def insert_match(self, iid1, iid2, transform, area_b_in_a):
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO matchs (a_imgid, b_imgid, transfrom, area_b_in_a) VALUES(?,?,?,ST_GeomFromText(?,0));',
            (iid1, iid2, transform, area_b_in_a)
        )

    def update_match(self, iid1, iid2, transform, area_b_in_a):
        cursor = self.conn.cursor()
        cursor.execute(
            'UPDATE matchs SET transfrom=?, area_b_in_a=ST_GeomFromText(?,0) WHERE a_imgid=? AND b_imgid=?;',
            (transform, area_b_in_a, iid1, iid2)
        )

    def get_match(self, iid1, iid2):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT transfrom, ST_AsText(area_b_in_a) FROM matchs WHERE a_imgid=? AND b_imgid=?;",
            (iid1, iid2))
        for i in cursor:
            return i

    def get_intersect(self, fid):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, name, ST_ASText(geom), "
            "ST_ASText(ST_Centroid(ST_Intersection(geom,"
            "(SELECT geom FROM frames WHERE id=?))))"
            "FROM frames;",
            (fid,)
        )
        lst = []
        for id_, name, wkt_poly, wkt_cpoint in cursor:
            poly = shapely.from_wkt(wkt_poly)
            cpoint = shapely.from_wkt(wkt_cpoint)
            lst.append((id_, name, poly, cpoint))
        return lst

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()