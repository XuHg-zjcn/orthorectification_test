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
import xml.etree.ElementTree as ET
import numpy as np


def get_floats(node):
    return tuple(map(lambda x:float(x.text), node))

def get_lcpc(model):
    lc = get_floats(model.find('lc_List'))
    pc = get_floats(model.find('pc_List'))
    return lc, pc

# ref: *The SPOT Scene Standard Digital Product Format* p.57
# on https://regards.cnes.fr/user/swh/modules/54  Identifier:S4-ST-73-01-SI
def read_spot_rpc_simplified(metadata_path):
    tree = ET.parse(metadata_path)
    root = tree.getroot()
    center = get_floats(root.find('Dataset_Frame/Scene_Center'))[:2]
    imgsize = get_floats(root.find('Raster_Dimensions'))[:2]
    dire = get_lcpc(root.find('Geoposition/Simplified_Location_Model/Direct_Location_Model'))
    reve = get_lcpc(root.find('Geoposition/Simplified_Location_Model/Reverse_Location_Model'))
    return center, imgsize, dire, reve

def read_spot1_4_rpc(metadata_path):
    def get_floats(node):
        return tuple(map(lambda x:float(x.text), node))
    tree = ET.parse(metadata_path)
    root = tree.getroot()
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
    center, imgsize, _, (model_i, model_j) = read_spot_rpc_simplified(path_in)
    write_rpc(sys.argv[2], center, imgsize, model_i, model_j)
    # RPC文件应该是 影像文件名+_RPC.TXT TODO: 自动决定输出文件名
    # 生成RPC后可用gdalwarp进行变换，如:
    # gdalwarp -r cubic -rpc -t_srs EPSG:32651 IMAGERY.TIF IMAGERY_warp.TIF
    # 但是SPOT1-4可能因为卫星自定位精度低效果不是很好，SPOT5又能用otb处理
