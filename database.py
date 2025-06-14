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
from collections import defaultdict, namedtuple
import os
import numpy as np
import shapely
from transform import PerspectiveTransform


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
            'transform BLOB,'  # 投影矩阵，9个double值
            'area_b_in_a POLYGON,'
            'n_points INTEGER,'     # 匹配的点数
            'last_update DATETIME,' # 最后更新时间
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

    def get_img(self, path, size, fid=None):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id FROM images WHERE size=? AND paths LIKE ? LIMIT 1;",
            (size, '%\n'+path+'\n%'))
        for idx in cursor:
            return idx
        filename = os.path.basename(path)
        if fid is not None:
            cursor.execute(
                "SELECT id, paths FROM images WHERE size=? AND frameid=? AND paths LIKE ? LIMIT 1;",
                (size, fid, '%/'+filename+'\n%'))
        else:
            cursor.execute(
                "SELECT id, paths FROM images WHERE size=? AND paths LIKE ? LIMIT 1;",
                (size, '%/'+filename+'\n%'))
        for idx, paths in cursor:
            path_lst =  paths.split('\n')
            if any(map(lambda x:filename==os.path.basename(x), path_lst)):
                return idx

    def get_img_path_byid(self, iid):
        cursor = self.conn.cursor()
        cursor.execute("SELECT paths, size FROM images WHERE id=?;", (iid,))
        lst = list(cursor)
        if len(lst) == 0:
            return None
        paths = lst[0][0].split('\n')
        size = lst[0][1]
        for path in paths:
            if os.path.isfile(path) and os.stat(path).st_size == size:
                return path

    def insert_match(self, iid1, iid2, transform, area_b_in_a, n_points):
        assert transform.shape == (3,3)
        transform_blob = transform.astype(np.float64).tobytes()
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO matchs '
            '(a_imgid, b_imgid, transform, area_b_in_a, n_points, last_update) '
            'VALUES(?,?,?,ST_GeomFromText(?,0), ?, CURRENT_TIMESTAMP);',
            (iid1, iid2, transform_blob, area_b_in_a, n_points)
        )

    def update_match(self, iid1, iid2, transform, area_b_in_a, n_points):
        assert transform.shape == (3,3)
        transform_blob = transform.astype(np.float64).tobytes()
        cursor = self.conn.cursor()
        cursor.execute(
            'UPDATE matchs '
            'SET transform=?, '
            'area_b_in_a=ST_GeomFromText(?,0), '
            'n_points=?, '
            'last_update=CURRENT_TIMESTAMP '
            'WHERE a_imgid=? AND b_imgid=?;',
            (transform_blob, area_b_in_a, iid1, iid2, n_points)
        )

    def insert_replace_match(self, iid1, iid2, transform, area_b_in_a, n_points):
        assert transform.shape == (3,3)
        transform_blob = transform.astype(np.float64).tobytes()
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO matchs '
            '(a_imgid, b_imgid, transform, area_b_in_a, n_points, last_update) '
            'VALUES(?,?,?,ST_GeomFromText(?,0), ?, CURRENT_TIMESTAMP);',
            (iid1, iid2, transform_blob, area_b_in_a, n_points)
        )

    def get_match(self, iid1, iid2):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT transform, ST_AsText(area_b_in_a), n_points, last_update "
            "FROM matchs "
            "WHERE a_imgid=? AND b_imgid=?;",
            (iid1, iid2))
        for tf_blob, area_b_in_a, n_points, last_update in cursor:
            tf = np.frombuffer(tf_blob, np.float64).reshape((3, 3))
            tf = PerspectiveTransform(tf)
            return tf, area_b_in_a, n_points, last_update

    def get_match_tranform_bidire(self, iid1, iid2):
        res1 = self.get_match(iid1, iid2)
        if res1 is not None:
            return res1[0]
        res2 = self.get_match(iid2, iid1)
        if res2 is not None:
            tf, _ = res2
            return tf.inv()

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

    def get_imagepairs_need_match(self, iid=None):
        cursor = self.conn.cursor()
        # 此处SQL代码按照DeepSeek的提示进行修改
        query = (
            'SELECT DISTINCT match1.b_imgid, match2.b_imgid '
            'FROM matchs AS match1 '
            'JOIN matchs AS match2 '
            'ON match1.a_imgid = match2.a_imgid '
            'AND match1.b_imgid < match2.b_imgid '
            'WHERE '
            +('match1.a_imgid = ? AND ' if iid is not None else '')+
            'ST_Intersects(match1.area_b_in_a, match2.area_b_in_a) '
            'AND NOT EXISTS ( '
            '   SELECT 1 '
            '   FROM matchs '
            '   WHERE (a_imgid=match1.b_imgid AND b_imgid=match2.b_imgid) '
            '   OR (b_imgid=match1.b_imgid AND a_imgid=match2.b_imgid)) '
            'ORDER BY ST_Area(ST_Intersection(match1.area_b_in_a, match2.area_b_in_a)) DESC;'
        )
        if iid is not None:
            cursor.execute(query, (iid,))
        else:
            cursor.execute(query)
        return list(cursor)

    def get_matchways_len2(self, iid1, iid2):
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT match1.a_imgid '
            'FROM matchs AS match1 '
            'JOIN matchs AS match2 '
            'ON match1.a_imgid = match2.a_imgid '
            'WHERE match1.b_imgid=? AND match2.b_imgid=? '
            'ORDER BY match1.n_points * match2.n_points DESC;',
            (iid1, iid2)
        )
        return list(map(lambda x:x[0],cursor))

    def get_matchways_more(self, start, ends, maxlength=3):
        ResultItem = namedtuple('ResultItem', ['path', 'weight'])
        # 此处SQL代码由DeepSeek给出并做修改
        cursor = self.conn.cursor()
        sql = '''
        WITH RECURSIVE cte AS (
        SELECT b_imgid AS current_node,
            ',' || a_imgid || ',' || b_imgid || ',' AS path,
            1.0/n_points AS total_weight,
            1 AS length
        FROM matchs
        WHERE a_imgid = ? -- 起点
        UNION ALL
        SELECT m.b_imgid AS current_node,
            cte.path || m.b_imgid || ',' AS path,
            cte.total_weight + 1.0/m.n_points AS total_weight,
            cte.length + 1 AS length
        FROM cte, matchs m
        WHERE m.a_imgid = cte.current_node
        AND cte.path NOT LIKE '%,' || m.b_imgid || ',%'
        AND cte.length < ? -- 路径长度限制
        )
        SELECT path, 1.0/total_weight
        FROM cte
        WHERE ? LIKE '%,' || current_node || ',%' -- 多个终点
        ORDER BY 1.0/total_weight DESC;
        '''
        ends_str = ','+(','.join(map(lambda x:str(x), ends)))+','
        cursor.execute(sql, (start, maxlength, ends_str))
        paths_lst = []
        for path_str, weight in cursor:
            path_intlst = list(map(lambda x:int(x), path_str.split(',')[1:-1]))
            paths_lst.append(ResultItem(path_intlst, weight))
        return paths_lst

    def get_matchways_more_tf(self, start_end_pairs, maxlength=3):
        ResultItem = namedtuple('ResultItem', ['path', 'weight', 'transform'])
        d = defaultdict(set) # 以start为键，值为end组成的集合
        for start, end in start_end_pairs:
            d[start].add(end)
        result = defaultdict(list)  # 以(start,end)为键，值为ResultItem对象
        for start, ends in d.items():
            paths_lst = self.get_matchways_more(start, ends, maxlength)
            for path, weight in paths_lst:
                node_pre = None
                t = PerspectiveTransform.keep()
                for node in path:
                    if node_pre is None:
                        node_pre = node
                        continue
                    t_ = self.get_match_tranform_bidire(node, node_pre)
                    t = t_.fog(t)
                    node_pre = node
                pair = (path[0], path[-1])
                result[pair].append(ResultItem(path, weight, t))
        return result

    def get_matchways_more_tf_singlepair(self, start, end, maxlength=3):
        pair = (start, end)
        result = self.get_matchways_more_tf([pair], maxlength)
        return result[pair]

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()