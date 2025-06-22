#!/usr/bin/env python3
#############################################################################
# 查看统计信息和绘制匹配关系的图
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
import networkx as nx
import matplotlib.pyplot as plt
import database


def print_info(G):
    n_allnode = len(G.nodes)
    components = list(nx.connected_components(G))
    components_ = filter(lambda x:len(x)>=2, components)
    components_ = sorted(components_, key=len, reverse=True)
    isolates = set(nx.isolates(G))
    print(f'{n_allnode} nodes in the graph')
    print(f'found {len(components_)} components with node>=2')
    for i, component in enumerate(components_):
        n_node = len(component)
        print(f'componet {i+1} : {n_node} nodes, {n_node/n_allnode:.1%} node of graph')
        print(component)
        print()
    print(f'{len(isolates)} isolates nodes, {len(isolates)/n_allnode:.1%} node of graph')
    print(isolates)

def draw_graph(G):
    # 此处使用了DeepSeek给出的代码
    nx.draw(G, nx.spring_layout(G), with_labels=True, node_size=300, node_color='red',
            edge_color='blue', font_size=12, font_family='sans-serif')
    plt.show()


if __name__ == '__main__':
    db = database.Database('data/imagery.db')
    G = nx.Graph()
    nodes = db.get_all_image_iid()
    G.add_nodes_from(nodes)
    edges = db.get_all_match_iidpair()
    G.add_edges_from(edges)
    print_info(G)
    draw_graph(G)
