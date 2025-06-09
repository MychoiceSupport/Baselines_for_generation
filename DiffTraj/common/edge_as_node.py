import os.path
import sys
sys.path.append('../common')
sys.path.append('../../')
import dgl
import numpy as np
import pandas as pd
import networkx as nx
from rtree import Rtree
from osgeo import ogr
try:
    from common.spatial_func import SPoint, distance
    from common.road_network import RoadNetwork, UndirRoadNetwork
except:
    from spatial_func import SPoint, distance
    from road_network import RoadNetwork, UndirRoadNetwork
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
    ID2gps = dict(zip(df_nodes['ID'].to_numpy(), df_nodes[['y', 'x']].to_numpy()))

    nodes_data = df_nodes[['y', 'x']].to_numpy()
    tuple_nodes = [tuple(node) for node in nodes_data]
    gps2ID = dict(zip(tuple_nodes, df_nodes['ID'].to_numpy()))
    # print("查看id的格式:", gps2ID.keys())
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

    count = 0
    edge_ids = []

    all_coord_list = []
    for i, env in enumerate(edges_data):
        # eid, coords, length
        # 'eid', 'u', 'v', 'ids', 'highway','coords'
        print("当前的位置:",i)
        if i == 0:
            print("查看数据env:",env)
        coords = []
        coords_id_list = []
        u, v = nodes_hash[env[1]], nodes_hash[env[2]]
        # coord_list = ast.literal_eval(env[-1])
        coord_list = env[-1]
        # print(coord_list)
        if isinstance(coord_list, int):
            coord_list = [coord_list]
        for coord in coord_list:
            coords.append(SPoint(coord[0], coord[1]))
            try:
                coords_id_list.append(gps2ID[(coord[0], coord[1])])
            except:
                gps2ID[(coord[0], coord[1])] = max(list(gps2ID.values())) + 1
                coords_id_list.append(gps2ID[(coord[0], coord[1])])
        if i == 0:
            print("查看数据:",coord_list[0][0], coord_list[0][1])
        all_coord_list.append(coords_id_list)
        # coords = np.array(coords)
        # coords = coords.reshape(-1, 2)
        # print(coords)
        geom_line = create_wkt(coords, str_type='MultiPoint')
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
        G.add_node(eid, u=u, v=v,eid=eid, coords=coords, length=length, highway=env[4])
        edge_ids.append(eid)


    edge_id = 0
    for i in tqdm(range(len(all_coord_list))):
        for j in range(i + 1, len(all_coord_list)):
            common_elements = set(all_coord_list[i]) & set(all_coord_list[j])
            if common_elements:
                G.add_edge(G.nodes[i]['eid'], G.nodes[j]['eid'], eid=edge_id, u=G.nodes[i]['eid'], v=G.nodes[j]['eid'])
                edge_id = edge_id + 1
    # G.edata[dgl.EID] = edge_ids
    # print("查看edge_ids的情况:", edge_ids)
    # print("有self-loop!,个数为:", count)
    print('# of nodes:{}'.format(G.number_of_nodes()))
    print('# of edges:{}'.format(G.number_of_edges()))

    ####这一部分主要是看有没有重复的内容

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
    with open(f'../{target_path}/{city_name}/{city_name}_{is_directed}_use_road_as_node.pkl', 'wb') as f:
        pickle.dump(rn, f)

    with open(f'../{target_path}/{city_name}/{city_name}_{is_directed}_use_road_as_node.pkl', 'rb') as f:
        graph = pickle.load(f)
    print("保存成功！")
    print(graph.nodes[0])

def store_rn_graph_index(rn, target_path='road_graph', city_name='Porto', is_directed=True):
    import pickle
    # if not os.path.exists(os.path.join(target _path, city_name)):
    #     os.makedirs(city_name)
    print(rn.nodes[0])
    try:
        with open(f'../{target_path}/{city_name}/{city_name}_{is_directed}_use_road_as_node.pkl', 'wb') as f:
            pickle.dump(rn, f)

        with open(f'../{target_path}/{city_name}/{city_name}_{is_directed}_use_road_as_node.pkl', 'rb') as f:
            graph = pickle.load(f)
    except:
        with open(f'{target_path}/{city_name}/{city_name}_{is_directed}_use_road_as_node.pkl', 'wb') as f:
            pickle.dump(rn, f)

        with open(f'{target_path}/{city_name}/{city_name}_{is_directed}_use_road_as_node.pkl', 'rb') as f:
            graph = pickle.load(f)
    print("保存成功！")
    print(graph.nodes[0])

if __name__ == '__main__':
    # wkt2coords('LINESTRING (-8.6314887 41.0996204, -8.6315913 41.0991539, -8.6316336 41.0989615)')
    load_rn_csv('road_network/Porto/nodes.csv',
                'road_network/Porto/edges.csv',None, True, False)