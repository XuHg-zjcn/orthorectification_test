#!/usr/bin/env python3
#############################################################################
# 从SPOT5影像元数据生成RPC文件
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
import xml.etree.ElementTree as ET
import numpy as np
import shapely
import common
from datetime import datetime


class SPOTMetadata:
    pattern = r'\d{3}-\d{3}_S\d_\d{3}-\d{3}-\d_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_\w+-\d_\w_\w{2}_\w{2}'
    def __init__(self, path):
        self.path = path
        self.tree = ET.parse(path)

    @classmethod
    def open_by_other_path(cls, path):
        basename_upper = os.path.basename(path)
        if basename_upper == 'METADATA.DIM':
            pass
        elif basename_upper[-4:] == '.TIF':
            path = os.path.join(os.path.dirname(path), 'METADATA.DIM')
        elif basename_upper == 'SCENE01':
            path = os.path.join(path, 'METADATA.DIM')
        elif re.match(basename_upper, cls.pattern) is not None:
            path = os.path.join(path, 'SCENE01/METADATA.DIM')
        else:
            return None
        return cls(path)

    # ref: *The SPOT Scene Standard Digital Product Format* p.57
    # on https://regards.cnes.fr/user/swh/modules/54  Identifier:S4-ST-73-01-SI
    def read_spot_rpc_simplified(self):
        def get_floats(node):
            return tuple(map(lambda x:float(x.text), node))
        def get_lcpc(model):
            lc = get_floats(model.find('lc_List'))
            pc = get_floats(model.find('pc_List'))
            return lc, pc
        root = self.tree.getroot()
        center = get_floats(root.find('Dataset_Frame/Scene_Center'))[:2]
        imgsize = get_floats(root.find('Raster_Dimensions'))[:2]
        dire = get_lcpc(root.find('Geoposition/Simplified_Location_Model/Direct_Location_Model'))
        reve = get_lcpc(root.find('Geoposition/Simplified_Location_Model/Reverse_Location_Model'))
        return center, imgsize, dire, reve

    def read_spot1_4_rpc(self):
        def get_floats(node):
            return tuple(map(lambda x:float(x.text), node))
        root = self.tree.getroot()
        obm = root.find('Data_Strip/Models/OneB_Model')
        if obm is None:  # 我下载的SPOT5元数据里没有OneB_Model
            return
        Norm = get_floats(obm.find('Coordinate_Normalization'))
        A = get_floats(obm.find('A'))
        B = get_floats(obm.find('B'))
        I = get_floats(obm.find('Direct_OneB_Model/I'))
        J = get_floats(obm.find('Direct_OneB_Model/J'))
        L = get_floats(obm.find('Reverse_OneB_Model/L'))
        P = get_floats(obm.find('Reverse_OneB_Model/P'))
        # TODO: 使用精度更高的模型

    def read_calib(self):
        root = self.tree.getroot()
        band_params = list(root.find('Data_Strip/Sensor_Calibration/Calibration'))
        band_lst = []
        for band in band_params:
            cell_lst = []
            for cell in band.find('Gain_Section/Pixel_Parameters/Cells'):
                G = float(cell.find('G').text)
                DC = float(cell.find('DARK_CURRENT').text)
                cell_lst.append((G, DC))
            band_lst.append(cell_lst)
        return band_lst

    def get_tf_to_geo(self):
        root = self.tree.getroot()
        v_lst = list(root.find('Dataset_Frame'))
        corn_pix = []
        corn_geo = []
        for v in v_lst:
            if v.tag != 'Vertex':
                continue
            lon = float(v.find('FRAME_LON').text)
            lat = float(v.find('FRAME_LAT').text)
            row = int(v.find('FRAME_ROW').text)-1  # SPOT元数据中以1为起点
            col = int(v.find('FRAME_COL').text)-1
            corn_pix.append((col, row))
            corn_geo.append((lon, lat))
        poly_geo = shapely.polygons(corn_geo)
        pt = common.getPerspective_4coords(corn_pix, corn_geo)
        return poly_geo, pt

    def get_arquire_date(self):
        root = self.tree.getroot()
        dataset_name = root.find('Dataset_Id/DATASET_NAME').text
        m = re.match(r'SCENE \d \d{3}-\d{3}[//]?\d? (\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) \d \w', dataset_name)
        return datetime.strptime(m.group(1), '%y/%m/%d %H:%M:%S')


# ref: https://gdal.org/en/stable/drivers/raster/gtiff.html#georeferencing
# ref: http://geotiff.maptools.org/rpc_prop.html
def write_rpc(fn, center, imgsize, model_i, model_j):
    with open(fn, 'w') as f:
        lon_off = center[0]
        lat_off = center[1]
        f.write('ERR_BIAS: 0.0\n'
                'ERR_RAND: 0.0\n'
                'LINE_OFF: 0.0\n'
                'SAMP_OFF: 0.0\n'
                f'LAT_OFF: {lat_off}\n'
                f'LONG_OFF: {lon_off}\n'
                'HEIGHT_OFF: 1\n'
                'LINE_SCALE: 1\n'
                'SAMP_SCALE: 1\n'
                'LAT_SCALE: 1\n'
                'LONG_SCALE: 1\n'
                'HEIGHT_SCALE: 1\n')
        # TODO: 对多项式变换进行抽象
        # TODO: 归一化处理
        line_num_coeff = np.zeros(20)
        line_num_coeff[0] = model_i[0] + \
            model_i[1]*lat_off + model_i[2]*lon_off + \
            model_i[3]*lat_off*lon_off + \
            model_i[4]*lat_off**2 + model_i[5]*lon_off**2
        line_num_coeff[2] = model_i[1] + model_i[3]*lon_off + 2*model_i[4]*lat_off
        line_num_coeff[1] = model_i[2] + model_i[3]*lat_off + 2*model_i[5]*lon_off
        line_num_coeff[4] = model_i[3]
        line_num_coeff[8] = model_i[4]
        line_num_coeff[7] = model_i[5]
        line_den_coeff = np.zeros(20)
        line_den_coeff[0] = 1
        samp_num_coeff = np.zeros(20)
        samp_num_coeff[0] = model_j[0] + \
            model_j[1]*lat_off + model_j[2]*lon_off + \
            model_j[3]*lat_off*lon_off + \
            model_j[4]*lat_off**2 + model_j[5]*lon_off**2
        samp_num_coeff[2] = model_j[1] + model_j[3]*lon_off +2*model_j[4]*lat_off
        samp_num_coeff[1] = model_j[2] + model_j[3]*lat_off +2*model_j[5]*lon_off
        samp_num_coeff[4] = model_j[3]
        samp_num_coeff[8] = model_j[4]
        samp_num_coeff[7] = model_j[5]
        samp_den_coeff = np.zeros(20)
        samp_den_coeff[0] = 1
        for i in range(20):
            f.write(f'LINE_NUM_COEFF_{i+1}: {line_num_coeff[i]}\n')
        for i in range(20):
            f.write(f'LINE_DEN_COEFF_{i+1}: {line_den_coeff[i]}\n')
        for i in range(20):
            f.write(f'SAMP_NUM_COEFF_{i+1}: {samp_num_coeff[i]}\n')
        for i in range(20):
            f.write(f'SAMP_DEN_COEFF_{i+1}: {samp_den_coeff[i]}\n')

if __name__ == '__main__':
    path_in = sys.argv[1]
    path_out = sys.argv[2]
    metadata = SPOTMetadata(path_in)
    center, imgsize, _, (model_i, model_j) = metadata.read_spot_rpc_simplified()
    write_rpc(sys.argv[2], center, imgsize, model_i, model_j)
    # RPC文件应该是 影像文件名+_RPC.TXT TODO: 自动决定输出文件名
    # 生成RPC后可用gdalwarp进行变换，如:
    # gdalwarp -r cubic -rpc -t_srs EPSG:32651 IMAGERY.TIF IMAGERY_warp.TIF
    # 但是SPOT1-4可能因为卫星自定位精度低效果不是很好，SPOT5又能用otb处理
