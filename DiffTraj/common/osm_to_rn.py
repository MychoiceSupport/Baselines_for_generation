import argparse
import os
import networkx as nx
import osmium as o
import osmnx as ox
import pandas as pd

candi_highway_types = {'motorway': 7, 'trunk': 6, 'primary': 5, 'secondary': 4, 'tertiary': 3,
                       'unclassified': 2, 'residential': 1, 'motorway_link': 7, 'trunk_link': 6, 'primary_link': 5,
                       'secondary_link': 4, 'tertiary_link': 3, 'living_street': 2, 'service': 0, 'road': 2}

class OSM2RNHandler(o.SimpleHandler):


    def __init__(self, rn):
        super(OSM2RNHandler, self).__init__()
        self.candi_highway_types = {'motorway': 7, 'trunk': 6, 'primary': 5, 'secondary': 4, 'tertiary': 3,
                       'unclassified': 2, 'residential': 1, 'motorway_link': 7, 'trunk_link': 6, 'primary_link': 5,
                       'secondary_link': 4, 'tertiary_link': 3, 'living_street': 2, 'service': 0, 'road': 2}
        self.rn = rn
        self.eid = 0
        # self.node = {
        #     'place': [],
        #     'id': [],
        #     'highway': []
        # }
        print("初始化成功!")
        self.nid = 0

    def way(self, w):
        if 'highway' in w.tags and w.tags['highway'] in self.candi_highway_types:
            raw_eid = w.id
            full_coords = []
            full_ids = []
            for n in w.nodes:
                full_coords.append((n.lat, n.lon))
                full_ids.append(n.ref)
            if 'oneway' in w.tags:
                ###单向路，注意图
                edge_attr = {'eid': self.eid, 'coords': full_coords, 'raw_eid': raw_eid, 'highway': w.tags['highway'],
                             'u': full_ids[0], 'v': full_ids[-1], 'osmid': full_ids}
                rn.add_edge(full_ids[0], full_ids[-1], **edge_attr)
                self.eid += 1
            else:
                ####Bidirection_road
                edge_attr = {'eid': self.eid, 'coords': full_coords, 'raw_eid': raw_eid, 'highway': w.tags['highway'],
                             'u': full_ids[0], 'v': full_ids[-1], 'osmid': full_ids}
                rn.add_edge(full_ids[0], full_ids[-1], **edge_attr)
                self.eid += 1

                reversed_full_coords = full_coords.copy()
                reversed_full_coords.reverse()
                reversed_full_ids = full_ids.copy()
                reversed_full_ids.reverse()

                edge_attr = {'eid': self.eid, 'coords': reversed_full_coords, 'raw_eid': raw_eid, 'highway': w.tags['highway'],
                             'u': reversed_full_ids[0], 'v': reversed_full_ids[-1], 'osmid': reversed_full_ids}
                rn.add_edge(reversed_full_ids[0], reversed_full_ids[-1], **edge_attr)
                self.eid += 1
            # print("初始化成功2!")

    def node(self, n):
        node_dict = {

        }
        lat = n.location.lat
        lon = n.location.lon
        node_dict['place'] = [lat, lon]
        node_dict['id'] = n.id
        tag_found = False
        # self.node['place'].append([lat,lon])
        for tag in n.tags:
            if tag.k == 'highway':
                # self.node['highway'].append(tag.v)
                node_dict['highway'] = tag.v
                tag_found = True
        if not tag_found:
            node_dict['highway'] = 'other'
            # self.node['highway'].append('other')
        rn.add_node(node_dict['id'], **node_dict)
        # self.node.add_node(n.id, **node_dict)
        # self.node.append(node_dict)
        self.nid += 1
        # print("初始化成功3!")

def store_osm_csv(rn, target_path):

    #rn.remove_nodes_from(list(nx.isolates(rn)))
    rn.remove_nodes_from(list(nx.isolates(rn)))
    # print(len(rn.nodes()))
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    ###直接保存road_network

    """nodes: [lat, lng]"""
    print('# of nodes:{}'.format(rn.number_of_nodes()))
    print('# of edges:{}'.format(rn.number_of_edges()))
    # node_map = {k: idx for idx, k in enumerate(rn.nodes())}

    # nodes = rn.nodes
    #
    # nodes_hash_for = {idx:k for idx, k in enumerate(nodes['id'])}
    # nodes_hash_re = {k:idx for idx, k in enumerate(nodes['id'])}
    # places_hash = {nodes_hash_for[idx]: k for idx, k in enumerate(nodes['place'])}

    node_feats = {
        'id':[],
        'lat':[],
        'lng':[],
        'street_count':[],
        'highway':[]
    }
    # with open(os.path.join(target_path, 'nodeOSM.txt'), 'w+') as node:
    for idx, data_nodes in enumerate(rn.nodes(data=True)):
        # node.write(f'{"%-6d"%idx} {coord[0]} {coord[1]}\n')
        data_node = data_nodes[1]
        if idx == 0:
            print(data_node)
        tmp_id = data_node['id'] ###对应的是osmid的

        node_feats['id'].append(data_node['id'])
        node_feats['lat'].append(data_node['place'][0])
        node_feats['lng'].append(data_node['place'][1])
        node_feats['street_count'].append(rn.degree(tmp_id))
        node_feats['highway'].append(data_node['highway'])


    node_path = os.path.join(target_path, 'node.csv')
    print("保存node节点成功")
    df_node = pd.DataFrame(node_feats)
    df_node.to_csv(node_path, index = False)

    edges = {
        'eid':[],
        'u':[],
        'v':[],
        'ids':[],
        'highway':[],
        'coords':[],
    }
    for stcoord, encoord, data in rn.edges(data=True):
        # edges[data['eid']] = {'st': node_map[stcoord],
        #                       'en': node_map[encoord],
        #                       'ids': data["osmid"],
        #                       'type': candi_highway_types[data['highway']],
        #                       'coords':data['coords'],
        #                       'u':data['u'], 'v':data['v']}
        edges['eid'].append(data['eid'])
        edges['u'].append(data['u'])
        edges['v'].append(data['v'])
        edges['ids'].append(data['osmid'])
        edges['highway'].append(candi_highway_types[data['highway']])
        edges['coords'].append(data['coords'])


    edge_path = os.path.join(target_path, 'edge.csv')
    df_edge = pd.DataFrame(edges)
    df_edge.to_csv(edge_path, index = False)
    print("保存edge文件成功!")

    # with open(os.path.join(target_path, 'wayTypeOSM.txt'), 'w+') as waytype:
    #     for idx, k in enumerate(sorted(edges.keys())):
    #         waytype.write(f'{"%-6d"%idx} {"%-10s"%edges[k]["type"]} {"%-4d"%candi_highway_types[edges[k]["type"]]}\n')
    #     waytype.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='../road_network/Porto/Porto_city.osm', help='the input path of the original osm data')
    parser.add_argument('--output_path', type=str, default='../road_graph/Porto', help='the output directory of the constructed road network')
    opt = parser.parse_args()
    # print(opt)

    rn = nx.DiGraph()
    handler = OSM2RNHandler(rn)
    handler.apply_file(opt.input_path, locations=True)
    print(opt.input_path)
    # handler.apply_file(opt.input_path)
    import pickle
    with open(os.path.join(opt.output_path, 'Porto_graph.pkl'), 'wb') as f:
        pickle.dump(rn, f)
    store_osm_csv(rn, opt.output_path)
