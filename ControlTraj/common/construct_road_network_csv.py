import os.path

import dgl
import numpy as np
import pandas as pd
import networkx as nx
from rtree import Rtree
from osgeo import ogr
from common.spatial_func import SPoint, distance
from common.mbr import MBR
import copy
from common.road_network import RoadNetwork, UndirRoadNetwork
import ast
import osmium as o
import argparse
import copy
import pickle
import re
from tqdm import tqdm



candi_highway_types = {'motorway': 7, 'trunk': 6, 'primary': 5, 'secondary': 4, 'tertiary': 3,
                       'unclassified': 2, 'residential': 1, 'motorway_link': 7, 'trunk_link': 6, 'primary_link': 5,
                       'secondary_link': 4, 'tertiary_link': 3, 'living_street': 2, 'other': 0, 'road': 2}
candi_node = {}




def create_wkt(coords, str_type='LineString'):
    # assert coords.shape[1] == 2
    # print("查看coords的长度:", len(coords))
    str_name = f"{str_type}("
    for i, cor in enumerate(coords):
        # print(cor)
        str_get = f'{cor.lng} {cor.lat}'
        if i != len(coords) - 1:
            str_get += ','
        str_name = str_name + str_get
    str_name = str_name + ")"
    # print(str_name)
    geom = ogr.CreateGeometryFromWkt(str_name)
    return geom

def wkt2coords(wkt):
    coordinates = re.findall(r"(-?\d+\.\d+)\s(-?\d+\.\d+)", wkt)
    coords = [[float(lat), float(lng)] for lng, lat in coordinates]
    # print(coords)
    return coords

def load_rn_csv(node_path, edge_path, graph_path = 'road_newtork', is_directed=True, save = False):
    edge_spatial_idx = Rtree()
    edge_idx = {}
    # node uses coordinate as key
    # edge uses coordinate tuple as key

    # if graph_path!=None:
    #     with open(graph_path, 'rb') as f:
    #         g = pickle.load(f)
    # nodes_get
    df_nodes = pd.read_csv(f'{node_path}', encoding='utf-8')
    # print(df_nodes.Index)
    node_feature = ['osmid','y', 'x','street_count', 'highway']


    df_nodes = df_nodes[node_feature]
    df_nodes['ID'] = list(range(df_nodes.shape[0]))
    df_nodes['highway'] = df_nodes['highway'].fillna('other')
    df_nodes['street_count'] = df_nodes['street_count'].fillna(0)

    nodes_hash = dict(zip(df_nodes['osmid'].to_numpy(), df_nodes['ID'].to_numpy()))
    places_hash = dict(zip(df_nodes['ID'].to_numpy(), df_nodes[['y', 'x']].to_numpy()))

    # edges_get
    df_edges = pd.read_csv(f'{edge_path}', encoding='utf-8')
    # print(df_edges.Index)
    edge_feature = ['fid', 'u', 'v','osmid','highway','geometry']
    print(max(df_nodes['ID']), min(df_nodes['ID']), max(df_edges['fid']), min(df_edges['fid']))

    df_edges = df_edges[edge_feature]
    u_data = df_edges['u'].to_numpy()
    u_node = []
    for uid in u_data:
        u_node.append(uid)
    df_edges['u'] = u_node

    v_data = df_edges['v'].to_numpy()
    v_node = []
    for vid in v_data:
        v_node.append(vid)
    df_edges['v'] = v_node

    df_edges['highway'] = df_edges['highway'].fillna('other_way')

    geo_features = df_edges['geometry'].to_numpy().tolist()
    geo_lines = []
    for data_slice in tqdm(geo_features):
        geo_lines.append(wkt2coords(data_slice))
    df_edges['coords'] = geo_lines

    nodes_data = df_nodes.to_numpy()
    edges_data = df_edges.to_numpy()

    if is_directed:
        G = nx.MultiDiGraph()
    else:
        G = nx.MultiGraph()

    find_rec = 1
    print("查看edges_data长度:", len(edges_data))
    node_ids = []
    for i, env in enumerate(nodes_data):
        if i == 0:
            print(env[1], env[2])
        node_id = nodes_hash[env[0]]
        # if env[4] not in candi_node.keys():
        #     candi_node[env[4]] = find_rec
        #     find_rec += 1
        G.add_node(node_id, nid=node_id, pt=SPoint(env[1], env[2]), count=env[3])
        node_ids.append(node_id)
    # G.ndata[dgl.NID] = node_ids

    count = 0
    edge_ids = []
    for i, env in enumerate(edges_data):
        # eid, coords, length
        # 'eid', 'u', 'v', 'ids', 'highway','coords'
        if i == 0:
            print("查看数据env:",env)
        coords = []
        u, v = nodes_hash[env[1]], nodes_hash[env[2]]
        # coord_list = ast.literal_eval(env[-1])
        coord_list = env[-1]
        # print(coord_list)
        if isinstance(coord_list, int):
            coord_list = [coord_list]
        for coord in coord_list:
            coords.append(SPoint(coord[0], coord[1]))
        # coords = np.array(coords)
        # coords = coords.reshape(-1, 2)
        # print(coords)
        geom_line = create_wkt(coords, str_type='LineString')
        envs = geom_line.GetEnvelope()
        if i == 0:
            print("查看edge：",envs)
        eid = env[0]
        length = sum(distance(coords[i], coords[i+1]) for i in range(len(coords) - 1))

        # edge_spatial_idx.insert(eid, (envs[0], envs[2], envs[1], envs[3]))
        if i == 0:
            print("查看coords:", envs[0], envs[2], envs[1], envs[3])
        edge_spatial_idx.insert(eid, (envs[0], envs[2], envs[1], envs[3]))

        ##真实的顺序: min.lng, min.lat, max.lng, max.lat
        # print("查看coords:", envs[0], envs[2], envs[1], envs[3])
        edge_idx[eid] = (u, v)
        if env[4] not in candi_highway_types.keys():
            env[4] = 'other'
        G.add_edge(u, v, u = u, v = v,eid=eid, coords=coords, length=length, highway=env[4])
        if u == v:
            count += 1
        edge_ids.append(eid)
    # G.edata[dgl.EID] = edge_ids
    # print("查看edge_ids的情况:", edge_ids)
    # print("有self-loop!,个数为:", count)
    print('# of nodes:{}'.format(G.number_of_nodes()))
    print('# of edges:{}'.format(G.number_of_edges()))

    if save == True:
        if not is_directed:
            return UndirRoadNetwork(G, edge_spatial_idx, edge_idx, max(df_nodes['ID']),(max(df_edges['fid'])))
        else:
            return RoadNetwork(G, edge_spatial_idx, edge_idx, max(df_nodes['ID']),(max(df_edges['fid'])))
    if not is_directed:
        new_graph = UndirRoadNetwork(G, edge_spatial_idx, edge_idx, max(df_nodes['ID']),(max(df_edges['fid'])))
        store_rn_graph_index(new_graph, 'road_graph', 'Porto')
        print("保存成功！")
        return UndirRoadNetwork(G, edge_spatial_idx, edge_idx, max(df_nodes['ID']),(max(df_edges['fid'])))
    else:
        new_graph = RoadNetwork(G, edge_spatial_idx, edge_idx, max(df_nodes['ID']),(max(df_edges['fid'])))
        store_rn_graph_index(new_graph, 'road_graph', 'Porto')
        print("保存成功！")
        return RoadNetwork(G, edge_spatial_idx, edge_idx, max(df_nodes['ID']),(max(df_edges['fid'])))

def store_rn_graph(rn, target_path='road_graph', city_name='Porto', is_directed=True):
    import pickle
    # if not os.path.exists(os.path.join(target _path, city_name)):
    #     os.makedirs(city_name)
    print(rn.nodes[0])
    with open(f'../{target_path}/{city_name}/{city_name}_{is_directed}_another.pkl', 'wb') as f:
        pickle.dump(rn, f)

    with open(f'../{target_path}/{city_name}/{city_name}_{is_directed}_another.pkl', 'rb') as f:
        graph = pickle.load(f)
    print("保存成功！")
    print(graph.nodes[0])

def store_rn_graph_index(rn, target_path='road_graph', city_name='Porto', is_directed=True):
    import pickle
    # if not os.path.exists(os.path.join(target _path, city_name)):
    #     os.makedirs(city_name)
    print(rn.nodes[0])
    with open(f'../{target_path}/{city_name}/{city_name}_{is_directed}_index_another.pkl', 'wb') as f:
        pickle.dump(rn, f)

    with open(f'../{target_path}/{city_name}/{city_name}_{is_directed}_index_another.pkl', 'rb') as f:
        graph = pickle.load(f)
    print("保存成功！")
    print(graph.nodes[0])

if __name__ == '__main__':
    # wkt2coords('LINESTRING (-8.6314887 41.0996204, -8.6315913 41.0991539, -8.6316336 41.0989615)')
    load_rn_csv('../road_network/Porto/nodes.csv',
                '../road_network/Porto/edges.csv',None, True, True)