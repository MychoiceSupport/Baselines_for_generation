import ast
import os
import sys
# sys.path.append('./')
# sys.path.append('././')
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import logging
import json
import pickle
import argparse
import holidays
import sys
sys.path.append('/home/huanghy/python_directories/Trajectory/TEVI')
sys.path.append('./')
sys.path.append('././')
sys.path.append('D:\BUAA_LAB\Traj_Rec_related\Trajectory\TEVI')
from common.grid import *
try:
    from utils.utils import *
except:
    from utils import *
from tqdm import tqdm
# from TEVI.road_net_for_edge_as_node_v2 import load_graph, construct_frequency_graph_not_dup, load_dgl_graph_v5
from torch.utils.data import TensorDataset, DataLoader
# from map_matching.candidate_point import *
class myDataset(Dataset):
    def __init__(self, gps_traj, grid_ids, edge_ids, ratios, time_feats, side_attr):
        self.len = len(gps_traj)
        self.gps_traj = gps_traj
        self.grid_ids = grid_ids
        self.edge_ids = edge_ids
        self.ratios = ratios
        self.time_feats = time_feats
        self.side_attr = side_attr

    def __getitem__(self, item):
        return self.gps_traj[item], self.grid_ids[item], self.edge_ids[item], self.ratios[item], self.time_feats[item], self.side_attr[item]

    def __len__(self):
        return self.len

def convert_timestamp_new(timestamps, predict_times, type_graph):
    batch, step = timestamps.shape[0], timestamps.shape[1]
    timestamps = timestamps.reshape(-1)
    timestamps = pd.to_datetime(timestamps, unit='s')

    predict_times = predict_times.reshape(-1)
    predict_times = pd.to_datetime(predict_times, unit='s')
    pre_hours, pre_mins, pre_secs = predict_times.hour.values, predict_times.minute.values, predict_times.second.values
    print(len(pre_hours), len(pre_mins), len(pre_secs))
    # print(timestamps[0:21], pd.Timedelta(1, unit='s'))
    hours, mins, secs = timestamps.hour.values, timestamps.minute.values, timestamps.second.values
    print(len(hours), len(mins), len(secs))
    time_start_end = np.array(timestamps).reshape(batch, step, 2)
    time_delta, mid_time_delta = get_the_time_delta(time_start_end, predict_times, type_graph)
    return np.stack([hours, mins, secs], axis=1).astype(np.float32), time_delta.astype(
        np.float32), mid_time_delta.astype(np.float32), np.stack([pre_hours, pre_mins, pre_secs], axis=1).astype(
        np.float32)


"""
之前算法 N * 21个time_delta的cumsum
本质计算的是end_time的
"""
def get_the_time_delta(time_all, predict_times, type_graph):
    # N * 21 * 2
    time_start_point = time_all[:, 0, 0]
    time_end_point = time_all[:, 0, 1]
    if type_graph == 'time_start':
        time_start = np.expand_dims(time_start_point, axis=-1)  # N * 1
        time_start = np.repeat(time_start, time_all.shape[1], axis=1).reshape(-1)  # N * 21
        time_end = time_all[:, :, 0].reshape(-1)
    elif type_graph == 'time_end':
        time_start = np.expand_dims(time_end_point, axis=-1)  # N * 1
        time_start = np.repeat(time_start, time_all.shape[1], axis=1).reshape(-1)  # N * 21
        time_end = time_all[:, :, 1].reshape(-1)
    elif type_graph == 'time_mid':
        time_start = np.expand_dims(time_start_point, axis=-1)
        time_start = np.repeat(time_start, time_all.shape[1], axis=1).reshape(-1)
        time_end = (time_all[:, :, 1] + time_all[:, :, 0]) / 2
        time_end = time_end.reshape(-1, )
    else:
        raise Exception('There is something wrong with type_graph')
    time_delta = (time_end - time_start) / pd.Timedelta(1, unit='s')

    mid_time_delta = (predict_times - time_start_point) / pd.Timedelta(1, unit='s')
    ###计算另一个
    # print(time_delta[0:21])
    return time_delta, mid_time_delta


def convert_timestamp(timestamps):
    timestamps = pd.to_datetime(timestamps, unit='s')
    hours, mins, secs = timestamps.hour.values, timestamps.minute.values, timestamps.second.values
    return np.stack([hours, mins, secs], axis=1).astype(np.float32)

def extract_time_features(timestamps):
    timestamps = timestamps.reshape(-1)
    timestamps = pd.to_datetime(timestamps)
    hours, mins, secs = timestamps.hour.values, timestamps.minute.values, timestamps.second.values
    holiday_result = holidays.Portugal()
    # is_weekend = np.where(timestamps in holiday_result, np.zeros_like(timestamps), np.ones_like(timestamps))
    is_holiday = np.array([int(date in holiday_result) for date in timestamps])
    weekday = timestamps.dayofweek.values

    #### 288 = 24 * 12
    return np.stack([hours, mins, secs], axis=1).astype(np.float32), np.stack([weekday, is_holiday], axis=1).astype(np.int32)

def get_time_features(timestamps):
    lens = timestamps.shape[-1]
    timestamps = timestamps.reshape(-1)
    timestamps = pd.to_datetime(timestamps)
    hours, mins, secs = timestamps.hour.values, timestamps.minute.values, timestamps.second.values
    time_values = (((hours * 60) + mins) // 5) % 288
    return time_values

def calculate_attr_result(gps_slice, time_interval, dataset='xian'):
    """
    :param gps_slice: N * 2, B * N * 2
    :return: 计算avg_speed, avg_dis
    """
    # print(gps_slice.shape)
    avg_dis, avg_time, travel_dis, avg_speed, num_point = calculate_route_dis(gps_slice, time_interval, dataset)
    travel_dis = np.array(travel_dis)
    # print(gps_slice.shape, avg_dis.shape, avg_speed.shape, travel_dis.shape)
    # departure_time, trip_distance,  trip_time, trip_length, avg_dis, avg_speed
    if dataset == 'xian':
        trip_time = np.array([(point * 3) for point in num_point])
        return np.concatenate(
            [travel_dis.reshape(-1, 1), avg_dis.reshape(-1, 1), trip_time.reshape(-1, 1),
             num_point.reshape(-1, 1),avg_speed.reshape(-1, 1)], axis=-1)
    elif dataset == 'chengdu':
        trip_time = np.array([(point * 2) for point in num_point])
        return np.concatenate(
            [travel_dis.reshape(-1, 1), avg_dis.reshape(-1, 1), trip_time.reshape(-1, 1),
             num_point.reshape(-1, 1),avg_speed.reshape(-1, 1)], axis=-1)
    elif dataset == 'Porto':
        trip_time = np.array([(point * 15) for point in num_point])
        # trip_length = np.ones_like((len(gps_slice), 1)) * 120
        print(travel_dis.shape, avg_dis.shape, trip_time.shape, avg_speed.shape)
        return np.concatenate(
            [travel_dis.reshape(-1, 1), avg_dis.reshape(-1, 1), trip_time.reshape(-1, 1),
             num_point.reshape(-1, 1),avg_speed.reshape(-1, 1)], axis=-1)
    else:
        raise Exception('Wrong dataset')

def get_area(dataset_name):
    if dataset_name == 'chengdu':
        min_lng = 104.025
        max_lng = 104.130
        min_lat = 30.65
        max_lat = 30.75
    elif dataset_name == 'xian':
        min_lng = 108.9
        max_lng = 109.0
        min_lat = 34.20
        max_lat = 34.28
    elif dataset_name == 'Porto':
        min_lng = -8.72
        max_lng = -8.52
        min_lat = 41.1
        max_lat = 41.2
    else:
        raise Exception("Wrong dataset!")
    return min_lat, max_lat, min_lng, max_lng

class MinMaxScaler:
    def __init__(self, min_lat=41.077, min_lng=-8.8056, max_lat=41.221, max_lng=-8.4585):
        self.min_x = min_lat
        self.max_x = max_lat
        self.min_y = min_lng
        self.max_y = max_lng

    def fit(self, min_lat=41.077, min_lng=-8.8056, max_lat=41.221, max_lng=-8.4585):
        self.min_x = min_lat
        self.max_x = max_lat
        self.min_y = min_lng
        self.max_y = max_lng

    def transform(self, data: torch.Tensor):
        """
        对数据进行标准化
        """
        raw_data = data.clone()
        raw_data[..., 0] = (data[..., 0] - self.min_x) / (self.max_x - self.min_x)
        raw_data[..., 1] = (data[..., 1] - self.min_y) / (self.max_y - self.min_y)
        return raw_data

    def fit_transform(self, data: torch.Tensor):
        """
        计算均值和标准差，并标准化数据
        """
        self.fit()
        return self.transform(data)

    def inverse_transform(self, data: torch.Tensor):
        """
        将标准化数据还原为原始分布
        """
        raw_data = data.clone()
        raw_data[..., 0] = data[..., 0] * (self.max_x - self.min_x) + self.min_x
        raw_data[..., 1] = data[..., 1] * (self.max_y - self.min_y) + self.min_y
        return raw_data


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: torch.Tensor):
        """
        计算数据的均值和标准差
        """
        len_data = data.shape[-1]
        self.mean = torch.mean(data.reshape(-1, len_data), dim=0)
        self.std = torch.std(data.reshape(-1, len_data), dim=0)
        print("查看mean和std的数值",self.mean, self.std)

    def transform(self, data: torch.Tensor):
        """
        对数据进行标准化
        """
        raw_data = data.clone()
        lens_raw = raw_data.shape[-1]
        for i in range(lens_raw):
            raw_data[..., i] = (data[..., i] - self.mean[i]) / (self.std[i] + 1e-10)
            # raw_data[..., 1] = (data[..., 1] - self.mean[1]) / (self.std[1] + 1e-10)
        return raw_data

    def fit_transform(self, data: torch.Tensor):
        """
        计算均值和标准差，并标准化数据
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: torch.Tensor):
        """
        将标准化数据还原为原始分布

        。
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet. Call `fit` first.")
        raw_data = data.clone()
        lens_raw = raw_data.shape[-1]
        for i in range(lens_raw):
            raw_data[..., i] = data[..., i] * self.std[i] + self.mean[i]
        # raw_data[..., 0] = data[..., 0] * self.std[0] + self.mean[0]
        # raw_data[..., 1] = data[..., 1] * self.std[1] + self.mean[1]
        return raw_data

def get_gaia_data(args, path, logger: logging.Logger):
    dtypes = ['train', 'val', 'test']

    dataset = args.dataset
    all_data_dict = dict()
    min_max_scaler = MinMaxScaler()
    std_scaler = StandardScaler() ## 对于距离的判断
    scaler_attr = StandardScaler()

    min_lat, max_lat, min_lng, max_lng = get_area(args.dataset)
    grid = create_grid_num(min_lat=min_lat, max_lat=max_lat, min_lng=min_lng, max_lng=max_lng,
                           nb_rows=16, nb_cols=16)

    for dtype in dtypes:
        all_gps_point = []
        gpses = []
        timestamps = []
        attr = []
        data_length = []
        data_path_raw = os.path.join(path, dataset)
        data_path = os.path.join(path, dataset, dtype)
        file_names = os.listdir(data_path)

        gps_flag = False
        time_flag = False
        lens_flag = False
        attr_flag = False
        save = True
        for file in tqdm(file_names):
            file_name = os.path.join(data_path, file)
            all_data = np.load(file_name, allow_pickle=True)
            if os.path.exists(os.path.join(data_path_raw, f'gps_{dtype}.npy')):
                gpses = np.load(os.path.join(data_path_raw, f'gps_{dtype}.npy'), allow_pickle=True)
                gps_flag = True
            if os.path.exists(os.path.join(data_path_raw, f'timestamps_{dtype}.npy')):
                timestamps = np.load(os.path.join(data_path_raw, f'timestamps_{dtype}.npy'), allow_pickle=True)
                time_flag = True
            if os.path.exists(os.path.join(data_path_raw, f'lens_{dtype}.npy')):
                data_length = np.load(os.path.join(data_path_raw, f'lens_{dtype}.npy'), allow_pickle=True)
                lens_flag = True
            if os.path.exists(os.path.join(data_path_raw, f'head_{dtype}.npy')):
                attr = np.load(os.path.join(data_path_raw, f'head_{dtype}.npy'), allow_pickle=True)
                attr_flag = True
            if gps_flag and time_flag and lens_flag and attr_flag:
                save = False
                break
            else:
                gps_data = all_data['coordinates']
                time_data = all_data['timestamps']
                for i, gps_slice in enumerate(gps_data):
                    gps_slice = np.array(gps_slice)
                    # gps_slice = gps_slice[:,[1,0]]
                    all_gps_point.append(torch.from_numpy(gps_slice))
                    if args.dataset == 'xian':
                        attr_0 = calculate_attr_result(gps_slice, 3, args.dataset)
                    elif args.dataset == 'chengdu':
                        attr_0 = calculate_attr_result(gps_slice, 2, args.dataset)
                    elif args.dataset == 'Porto':
                        attr_0 = calculate_attr_result(gps_slice, 15, args.dataset)
                    else:
                        raise Exception
                    data_length.append(gps_slice.shape[0])
                    attr.append(attr_0)
                    if len(gps_slice) <= 200:
                        if len(gps_slice) < 120:
                            continue
                        else:
                           gps_slice = resample_trajectory(gps_slice, 200)
                    else:
                        gps_slice = gps_slice[:200]
                    try:
                        gpses.append(gps_slice)
                    except:
                        pass
                # for i, time_stamp in enumerate(time_data):
                    time_slice = pd.to_datetime(np.array(time_data[i]))
                    time_features = get_time_features(np.array(time_slice))
                    if time_features.shape[0] <= 200:

                        pad_length = 200 - time_features.shape[0]
                        time_features = np.pad(time_features, (0, pad_length), mode='constant', constant_values=0)
                    else:
                        time_features = time_features[:200]
                    try:
                        timestamps.append(time_features)
                    except:
                        pass
            # break
        gpses = np.array(gpses)
        # print(gpses[0])
        timestamps = np.array(timestamps)
        traj_length = np.array(data_length)
        attr = np.array(attr).squeeze()

        # attr = scaler_attr.fit_transform(torch.from_numpy(attr)).numpy()
        # min_lng = np.min(gpses[:,:,1].reshape(-1)) - 0.001
        # max_lng = np.max(gpses[:,:,1].reshape(-1)) + 0.001
        # min_lat = np.min(gpses[:,:,0].reshape(-1)) - 0.001
        # max_lat = np.max(gpses[:,:,0].reshape(-1)) + 0.001
        grid_index = grid.get_1d_idx(gpses.reshape(-1, 2))
        grid_index = grid_index.reshape(-1,200,1)
        assert grid_index.all() < 257
        attr = np.concatenate([timestamps[:,0:1], attr, grid_index[:,0], grid_index[:,-1]], axis=-1)
        try:
            all_gps_point = torch.cat(all_gps_point, dim=0)
        except:
            pass
        if save:
            np.save(os.path.join(data_path_raw, f'gps_{dtype}.npy'), gpses)
            np.save(os.path.join(data_path_raw, f'timestamps_{dtype}.npy'), timestamps)
            np.save(os.path.join(data_path_raw, f'lens_{dtype}.npy'), traj_length)
            np.save(os.path.join(data_path_raw, f'head_{dtype}.npy'), attr)

        gpses = np.array(gpses)[:,:120]
        timestamps = np.array(timestamps)[:,:120]
        grid_index = np.array(grid_index)[:,:120]
        # print(gpses[0])
        timestamps = np.array(timestamps)
        if args.data_sample:
            random_index = torch.randperm(len(gpses))[:len(gpses) // 4].numpy()  # 无重复随机索引
            gpses = gpses[random_index]
            timestamps = timestamps[random_index]
            traj_length = traj_length[random_index]
            attr = attr[random_index]

        if dtype == 'train':
            try:
                np.save(os.path.join(data_path_raw, f'gps_point.npy'), all_gps_point)
            except:
                pass
            attr[:,1:6] = scaler_attr.fit_transform(torch.from_numpy(attr[:,1:6])).numpy()
            if args.transform:
                try:
                    std_scaler.fit(torch.cat(all_gps_point, dim=0))
                except:
                    std_scaler.fit(torch.from_numpy(gpses))
        else:
            attr[:, 1:6] = scaler_attr.transform(torch.from_numpy(attr[:, 1:6])).numpy()
        assert len(gpses) == len(timestamps)
        print("查看gps内容的特点:",gpses[0,0])
        logger.info(f"The data lens is {len(gpses)} in {dtype}.")
        all_data_dict[dtype] = myDataset(gpses, grid_index, gpses[:,:,0],
                                    gpses[:,:,0], timestamps, attr)
        print("查看最大最小:",np.max(gpses[:,:,0]),np.min(gpses[:,:,0]),np.max(gpses[:,:,1]),np.min(gpses[:,:,1]))
    return all_data_dict['train'], all_data_dict['val'], all_data_dict['test'], std_scaler, scaler_attr



def get_full_data(args, path, logger: logging.Logger):
    """
    :param path: 根据路网对于path进行处理
    :param dataset:
    :param logger:
    :param type_graph:
    :param device:

wqw    data_indice: raw_POLYLINE, matched_POLYLINE, edge_list, Origin, Destination,
                 timestamp, start_time, end_time

    :return:
    """
    dtypes = ['train', 'val', 'test']

    LAT_PER_KILOMETER = 8.993203677616966e-06 * 50
    LNG_PER_KILOMETER = 1.1700193970443768e-05 * 50
    min_lat, max_lat, min_lng, max_lng = get_area(args.dataset)
    grid = create_grid_num(min_lat=min_lat, max_lat=max_lat, min_lng=min_lng, max_lng=max_lng,
                           nb_rows=16, nb_cols=16)

    # 读取数据 #
    all_data = dict()
    scaler = MinMaxScaler()
    scaler1 = MinMaxScaler()
    scaler2 = StandardScaler() ## 对于距离的判断

    scaler_std = StandardScaler()

    raw_graph = load_graph('../road_graph')
    for dtype in dtypes:
        data_path = os.path.join(path,'new_matched',dtype)
        time_path = f'{path}'
        file_names = os.listdir(data_path)

        gps_files = [file for file in file_names if 'matched_p_poly_special' in file]
        edge_files = [file for file in file_names if 'matched_edge_list' in file]
        ratio_files = [file for file in file_names if 'ratio' in file]
        hash_files = [file for file in file_names if 'hash_list' in file]
        valid_files = [file for file in file_names if 'valid_index' in file]
        print(gps_files)
        print(edge_files)
        print(ratio_files)
        print(hash_files)

        gps_files = sorted(gps_files, key=lambda x: (len(x), x))
        edge_files = sorted(edge_files, key=lambda x: (len(x), x))
        ratio_files = sorted(ratio_files, key=lambda x: (len(x), x))
        hash_files = sorted(hash_files, key=lambda x: (len(x), x))
        valid_files = sorted(valid_files, key=lambda x:(len(x), x))
        time_data = os.path.join(time_path, f'{dtype}_raw_time.pkl')

        gpses = []
        edges = []
        ratios = []

        data_index = 0

        with open(time_data, 'rb') as f:
            time_series = pickle.load(f)
        time_series_data = []
        time_index = []

        data_lens = []
        for index, gps_file in enumerate(gps_files):
            # needed_index = []
            i = index
            valid_file_indexes = np.load(os.path.join(data_path, valid_files[i]), allow_pickle=True)
            print("查看当前gps_file:", gps_files[i], edge_files[i], ratio_files[i], hash_files[i])
            with open(os.path.join(data_path, hash_files[i]), 'rb') as f:
                hash_series = pickle.load(f)  ###主要是对应用的
                print("查看hash_series的长度:",len(hash_series))
            with open(os.path.join(data_path, gps_files[i]), 'rb') as f:
                gps_series = pickle.load(f)
                print("查看当前gps文件的长度:", len(gps_series))
                ### 读取小的数据集
                for j, val_idx in enumerate(valid_file_indexes):
                    # gps_slice = ast.literal_eval(gps_slice)
                    gps_slice = gps_series[val_idx]
                    gps_slice = np.array(gps_slice)
                    if gps_slice.shape[0] >= 120:
                        data_lens.append(len(gps_slice))
                        # print(np.array(gps_slice).shape)
                        # gpses.append(np.array(gps_slice[0:120]).astype(np.float32).reshape(1,120,2))
                        for k in range(120):
                            # print(gps_slice[k])
                            assert np.array(gps_slice[k]).shape[0] == 2
                            gpses.append(gps_slice[k])
                            time_series_data.append(time_series[hash_series[val_idx]][k])
                    else:
                        continue
                ##get_longer_series
            with open(os.path.join(data_path, edge_files[i]), 'rb') as f:
                edge_series = pickle.load(f)
                # edge_series = edge_series[valid_file_indexes]
                for j, val_idx in enumerate(valid_file_indexes):
                    edge_slice = edge_series[val_idx]
                    if len(edge_slice) >= 120:
                        edges.append(list(np.array(edge_slice)[0:120]))
                    else:
                        continue
                print("查看当前edges文件的长度:", len(edge_series))
                ##get_longer_series
            with open(os.path.join(data_path, ratio_files[i]), 'rb') as f:
                ratio_series = pickle.load(f)
                # ratio_series = ratio_series[valid_file_indexes]
                print("查看ratio_series的长度:", len(ratio_series))
                for j, val_idx in enumerate(valid_file_indexes):
                    ratio_slice = ratio_series[val_idx]
                    if len(ratio_slice) >= 120:
                        ratios.append(list(np.array(ratio_slice)[0:120]))
                    else:
                        continue
            # if i >= 3:
            #     break

                ##get_longer_series
            # gpses.append(gps_series)
            # edges.append(edge_series)
            # ratios.append(ratio_series)
            data_index += len(gps_series)
            data_index += len(gps_series)
            # print(time_series)

        # print(gpses[0][0], len(gpses[0]))
        gpses = np.array(gpses).reshape(-1, 120, 2)

        # gpses = [np.array(data) for data in gpses]
        edges = np.array(edges)
        ratios = np.array(ratios)
        # print("查看以下length的长度:",gpses.shape, edges.shape, ratios.shape, len(time_series_data))
        gpses[...,[0,1]] = gpses[...,[1,0]]
        # print(gpses[0])
        grid_index = grid.get_1d_idx(gpses.reshape(-1, 2))
        grid_index = grid_index.reshape(-1,120,1)
        # grid_index = np.array()
        # print("查看grid_coords的形状:", grid_index.shape)
        # print(time_series[0])
        time_series = np.array(time_series_data)
        # time_features = get_time_features(time_series)
        # time_features = time_features.reshape(-1, 120, 1).astype(np.int64)
        time_features_0, time_features_1 = extract_time_features(time_series)
        # time_features = np.concatenate([time_features_0, time_features_1], axis=-1)
        time_features = time_features_0.reshape(-1, 120, 3)
        attr_0 = calculate_attr_result(gpses,15, args.dataset)
        # gpses = torch.from_numpy(gpses)
        if dtype == 'train':
            if args.transform:
                scaler.fit()
                attr_0 = scaler1.fit_transform(torch.from_numpy(attr_0))
                scaler_std.fit(torch.from_numpy(gpses))
                gpses = torch.from_numpy(gpses)
                # scaler_2.fit(torch.from_numpy())
        else:
            # scaler.fit(torch.from_numpy(gpses))
            gpses = torch.from_numpy(gpses)
            attr_0 = scaler1.transform(torch.from_numpy(attr_0))
        # print("查看生成的特征:", attr_0.shape, gpses[:,0,:].shape, ratios[:, 0].reshape(-1, 1).shape,
        #       edges[:, 0].reshape(-1,1).shape, grid_index.shape)
        ## n * T * 2
        ##
        print(gpses[0,0])
        try:
            attr = np.concatenate([time_features[:, 0], attr_0, grid_index[:, 0], grid_index[:, -1]], axis=-1)
        except:
            attr = np.concatenate([time_features[:, 0], attr_0.numpy(), grid_index[:, 0], grid_index[:, -1]], axis=-1)
        print("查看特征:", time_features.shape, attr.shape)
        ####构建heads信息，这里构建多方面的信息
        try:
            assert len(gpses) == len(edges) == len(ratios) == len(time_features)
        except:
            print("查看这几个不同的地方:", len(gpses), time_features.shape)
        # edges = torch.from_numpy(edges).long()
        if dtype == 'train':
            all_needed_ids, id_num = torch.unique(torch.from_numpy(edges).long(), return_counts=True, sorted=True)
            # assert 50429 in all_needed_ids
            new_edge_ids = torch.searchsorted(all_needed_ids.long(), torch.from_numpy(edges).long())
            graph = construct_frequency_graph_not_dup(new_edge_ids, all_needed_ids, raw_graph, step=120)
            all_data[dtype] = myDataset(gpses, grid_index, new_edge_ids,
                                    ratios, time_features, attr)
        else:
            new_edge_ids = torch.searchsorted(all_needed_ids.long(), torch.from_numpy(edges).long())
            all_data[dtype] = myDataset(gpses, grid_index, new_edge_ids,
                                        ratios, time_features, attr)
        logger.info(f'{dtype} data num is {len(grid_index)}')
    assert graph != None
    return all_data['train'], all_data['val'], all_data['test'], scaler_std, scaler, graph, all_needed_ids.long(), id_num

def get_all_data(args, path, logger: logging.Logger):
    """
    :param path: 根据路网对于path进行处理
    :param dataset:
    :param logger:
    :param type_graph:
    :param device:

    data_indice: raw_POLYLINE, matched_POLYLINE, edge_list, Origin, Destination,
                 timestamp, start_time, end_time

    :return:
    """
    dtypes = ['train', 'val', 'test']

    LAT_PER_KILOMETER = 8.993203677616966e-06 * 50
    LNG_PER_KILOMETER = 1.1700193970443768e-05 * 50
    grid = create_grid_num(min_lat=41.077, min_lng=-8.8056, max_lat=41.221, max_lng=-8.4585,
                       nb_rows=16, nb_cols=16)

    # 读取数据 #
    all_data = dict()
    scaler = MinMaxScaler()
    scaler1 = MinMaxScaler()
    scaler2 = StandardScaler() ## 对于距离的判断

    scaler_std = StandardScaler()

    raw_graph = load_graph('../road_graph')
    for dtype in dtypes:
        data_path = os.path.join(path,'new_matched',dtype)
        time_path = f'{path}'
        file_names = os.listdir(data_path)

        gps_files = [file for file in file_names if 'matched_p_poly_special' in file]
        edge_files = [file for file in file_names if 'matched_edge_list' in file]
        ratio_files = [file for file in file_names if 'ratio' in file]
        hash_files = [file for file in file_names if 'hash_list' in file]
        print(gps_files)
        print(edge_files)
        print(ratio_files)
        print(hash_files)

        gps_files = sorted(gps_files, key=lambda x: (len(x), x))
        edge_files = sorted(edge_files, key=lambda x: (len(x), x))
        ratio_files = sorted(ratio_files, key=lambda x: (len(x), x))
        hash_files = sorted(hash_files, key=lambda x: (len(x), x))
        time_data = os.path.join(time_path, f'{dtype}_raw_time.pkl')

        gpses = []
        edges = []
        ratios = []

        data_index = 0

        with open(time_data, 'rb') as f:
            time_series = pickle.load(f)
        time_series_data = []
        time_index = []

        for index, gps_file in enumerate(gps_files):
            # needed_index = []
            i = index
            print("查看当前gps_file:", gps_files[i], edge_files[i], ratio_files[i], hash_files[i])
            with open(os.path.join(data_path, hash_files[i]), 'rb') as f:
                hash_series = pickle.load(f)  ###主要是对应用的
                print("查看hash_series的长度:",len(hash_series))
            with open(os.path.join(data_path, gps_files[i]), 'rb') as f:
                gps_series = pickle.load(f)
                # if dtype == 'train':
                #     if index <= 4:
                #         continue
                print("查看当前gps文件的长度:", len(gps_series))
                for j, gps_slice in enumerate(gps_series):
                    # gps_slice = ast.literal_eval(gps_slice)
                    gps_slice = np.array(gps_slice)


                    if gps_slice.shape[0] >= 120:
                        # print(np.array(gps_slice).shape)
                        # gpses.append(np.array(gps_slice[0:120]).astype(np.float32).reshape(1,120,2))
                        for k in range(120):
                            # print(gps_slice[k])
                            assert np.array(gps_slice[k]).shape[0] == 2
                            gpses.append(gps_slice[k])
                            time_series_data.append(time_series[hash_series[j]][k])
                    else:
                        continue
                ##get_longer_series
            with open(os.path.join(data_path, edge_files[i]), 'rb') as f:
                edge_series = pickle.load(f)
                for edge_slice in edge_series:
                    if len(edge_slice) >= 120:
                        edges.append(list(np.array(edge_slice)[0:120]))
                    else:
                        continue
                print("查看当前edges文件的长度:", len(edge_series))
                ##get_longer_series
            with open(os.path.join(data_path, ratio_files[i]), 'rb') as f:
                ratio_series = pickle.load(f)
                print("查看ratio_series的长度:", len(ratio_series))
                for ratio_slice in ratio_series:
                    if len(ratio_slice) >= 120:
                        ratios.append(list(np.array(ratio_slice)[0:120]))
                    else:
                        continue
            # if i >= 3:
            #     break

                ##get_longer_series
            # gpses.append(gps_series)
            # edges.append(edge_series)
            # ratios.append(ratio_series)
            data_index += len(gps_series)
            data_index += len(gps_series)
            # print(time_series)

        # print(gpses[0][0], len(gpses[0]))
        gpses = np.array(gpses).reshape(-1, 120, 2)
        # gpses = [np.array(data) for data in gpses]
        edges = np.array(edges)
        ratios = np.array(ratios)

        print("查看以下length的长度:",gpses.shape, edges.shape, ratios.shape, len(time_series_data))
        gpses[...,[0,1]] = gpses[...,[1,0]]
        print(gpses[0,0])
        grid_index = grid.get_1d_idx(gpses.reshape(-1, 2))
        grid_index = grid_index.reshape(-1,120,1)
        # grid_index = np.array()
        print("查看grid_coords的形状:", grid_index.shape)
        # print(time_series[0])
        time_series = np.array(time_series_data)
        # time_features = get_time_features(time_series)
        # time_features = time_features.reshape(-1, 120, 1).astype(np.int64)
        time_features_0, time_features_1 = extract_time_features(time_series)
        # time_features = np.concatenate([time_features_0, time_features_1], axis=-1)
        time_features = time_features_0.reshape(-1, 120, 3)
        attr_0 = calculate_attr_result(gpses,15)
        # gpses = torch.from_numpy(gpses)
        if dtype == 'train':
            if args.transform:
                scaler.fit()
                attr_0 = scaler1.fit_transform(torch.from_numpy(attr_0))
                scaler_std.fit(torch.from_numpy(gpses))
                gpses = torch.from_numpy(gpses)
                # scaler_2.fit(torch.from_numpy())
        else:
            # scaler.fit(torch.from_numpy(gpses))
            gpses = torch.from_numpy(gpses)
            attr_0 = scaler1.transform(torch.from_numpy(attr_0))
        # print("查看生成的特征:", attr_0.shape, gpses[:,0,:].shape, ratios[:, 0].reshape(-1, 1).shape,
        #       edges[:, 0].reshape(-1,1).shape, grid_index.shape)
        ## n * T * 2
        ##
        print(gpses[0,0])
        try:
            attr = np.concatenate([time_features[:, 0], attr_0, grid_index[:, 0], grid_index[:, -1]], axis=-1)
        except:
            attr = np.concatenate([time_features[:, 0], attr_0.numpy(), grid_index[:, 0], grid_index[:, -1]], axis=-1)
        print("查看特征:", time_features.shape, attr.shape)
        ####构建heads信息，这里构建多方面的信息
        try:
            assert len(gpses) == len(edges) == len(ratios) == len(time_features)
        except:
            print("查看这几个不同的地方:", len(gpses), time_features.shape)
        # edges = torch.from_numpy(edges).long()
        if dtype == 'train':
            all_needed_ids, id_num = torch.unique(torch.from_numpy(edges).long(), return_counts=True, sorted=True)
            assert 50429 in all_needed_ids
            new_edge_ids = torch.searchsorted(all_needed_ids.long(), torch.from_numpy(edges).long())
            graph = construct_frequency_graph_not_dup(new_edge_ids, all_needed_ids, raw_graph, step=120)
            all_data[dtype] = myDataset(gpses, grid_index, new_edge_ids,
                                    ratios, time_features, attr)
        else:
            new_edge_ids = torch.searchsorted(all_needed_ids.long(), torch.from_numpy(edges).long())
            all_data[dtype] = myDataset(gpses, grid_index, new_edge_ids,
                                        ratios, time_features, attr)
        logger.info(f'{dtype} data num is {len(grid_index)}')
    assert graph != None
    return all_data['train'], all_data['val'], all_data['test'], scaler, scaler_std, graph, all_needed_ids.long(), id_num


def preprocess_index(path):
    for dtype in ['train', 'val', 'test']:
        data_path = os.path.join(path, 'new_matched', dtype)
        file_names = os.listdir(data_path)
        gps_files = [file for file in file_names if 'matched_p_poly_special' in file]
        for index, gps_file in enumerate(gps_files):
            with open(os.path.join(data_path, gps_files[index]), 'rb') as f:
                gps_series = pickle.load(f)
                print("查看当前gps文件的长度:", len(gps_series))
                needed_idx = get_small_area_idx(gps_series, min_lat=41.10, min_lng=-8.72,
               max_lat=41.20, max_lng=-8.52)
                print("查看当前gps文件的长度:", len(needed_idx))
            np.save(os.path.join(path, 'new_matched', dtype, f"valid_index{gps_file.replace('matched_p_poly_special','')}.npy"), needed_idx)

def get_all_data_standard(args, path, logger: logging.Logger):
    """
    :param path: 根据路网对于path进行处理
    :param dataset:
    :param logger:
    :param type_graph:
    :param device:

    data_indice: raw_POLYLINE, matched_POLYLINE, edge_list, Origin, Destination,
                 timestamp, start_time, end_time

    :return:
    """
    dtypes = ['train', 'val', 'test']

    LAT_PER_KILOMETER = 8.993203677616966e-06 * 50
    LNG_PER_KILOMETER = 1.1700193970443768e-05 * 50
    grid = create_grid_num(min_lat=41.077, min_lng=-8.8056, max_lat=41.221, max_lng=-8.4585,
                       nb_rows=16, nb_cols=16)

    # 读取数据 #
    all_data = dict()
    scaler_x = StandardScaler()
    scaler1 = StandardScaler()

    scaler_std = StandardScaler()

    raw_graph = load_graph('../road_graph')
    for dtype in dtypes:
        data_path = os.path.join(path,'new_matched',dtype)
        time_path = f'{path}'
        file_names = os.listdir(data_path)

        gps_files = [file for file in file_names if 'matched_p_poly_special' in file]
        edge_files = [file for file in file_names if 'matched_edge_list' in file]
        ratio_files = [file for file in file_names if 'ratio' in file]
        hash_files = [file for file in file_names if 'hash_list' in file]
        print(gps_files)
        print(edge_files)
        print(ratio_files)
        print(hash_files)

        gps_files = sorted(gps_files, key=lambda x: (len(x), x))
        edge_files = sorted(edge_files, key=lambda x: (len(x), x))
        ratio_files = sorted(ratio_files, key=lambda x: (len(x), x))
        hash_files = sorted(hash_files, key=lambda x: (len(x), x))
        time_data = os.path.join(time_path, f'{dtype}_raw_time.pkl')

        gpses = []
        edges = []
        ratios = []

        data_index = 0

        with open(time_data, 'rb') as f:
            time_series = pickle.load(f)
        time_series_data = []
        time_index = []
        for index, gps_file in enumerate(gps_files):
            # needed_index = []
            i = index
            print("查看当前gps_file:", gps_files[i], edge_files[i], ratio_files[i], hash_files[i])
            with open(os.path.join(data_path, hash_files[i]), 'rb') as f:
                hash_series = pickle.load(f)  ###主要是对应用的
                print("查看hash_series的长度:",len(hash_series))
            with open(os.path.join(data_path, gps_files[i]), 'rb') as f:
                gps_series = pickle.load(f)


                print("查看当前gps文件的长度:", len(gps_series))
                for j, gps_slice in enumerate(gps_series):
                    # gps_slice = ast.literal_eval(gps_slice)
                    gps_slice = np.array(gps_slice)

                    if gps_slice.shape[0] >= 120:
                        # print(np.array(gps_slice).shape)
                        # gpses.append(np.array(gps_slice[0:120]).astype(np.float32).reshape(1,120,2))
                        for k in range(120):
                            # print(gps_slice[k])
                            assert np.array(gps_slice[k]).shape[0] == 2
                            gpses.append(gps_slice[k])
                            time_series_data.append(time_series[hash_series[j]][k])
                    else:
                        continue
                ##get_longer_series
            with open(os.path.join(data_path, edge_files[i]), 'rb') as f:
                edge_series = pickle.load(f)
                for edge_slice in edge_series:
                    if len(edge_slice) >= 120:
                        edges.append(list(np.array(edge_slice)[0:120]))
                    else:
                        continue
                print("查看当前edges文件的长度:", len(edge_series))
                ##get_longer_series
            with open(os.path.join(data_path, ratio_files[i]), 'rb') as f:
                ratio_series = pickle.load(f)
                print("查看ratio_series的长度:", len(ratio_series))
                for ratio_slice in ratio_series:
                    if len(ratio_slice) >= 120:
                        ratios.append(list(np.array(ratio_slice)[0:120]))
                    else:
                        continue
            data_index += len(gps_series)
            data_index += len(gps_series)
        gpses = np.array(gpses).reshape(-1, 120, 2)
        # gpses = [np.array(data) for data in gpses]
        edges = np.array(edges)
        ratios = np.array(ratios)

        print("查看以下length的长度:",gpses.shape, edges.shape, ratios.shape, len(time_series_data))
        gpses[...,[0,1]] = gpses[...,[1,0]]
        print(gpses[0,0])
        grid_index = grid.get_1d_idx(gpses.reshape(-1, 2))
        grid_index = grid_index.reshape(-1,120,1)
        # grid_index = np.array()
        print("查看grid_coords的形状:", grid_index.shape)
        # print(time_series[0])
        time_series = np.array(time_series_data)
        # time_features = get_time_features(time_series)
        # time_features = time_features.reshape(-1, 120, 1).astype(np.int64)
        time_features_0, time_features_1 = extract_time_features(time_series)
        # time_features = np.concatenate([time_features_0, time_features_1], axis=-1)
        time_features = time_features_0.reshape(-1, 120, 3)
        attr_0 = calculate_attr_result(gpses,15)
        # gpses = torch.from_numpy(gpses)
        if dtype == 'train':
            if args.transform:
                gpses = torch.from_numpy(gpses)
                scaler_x.fit(gpses)
                # scaler1.fit(torch.from_numpy(attr_0))
                # scaler_2.fit(torch.from_numpy())
        else:
            # scaler.fit(torch.from_numpy(gpses))
            gpses = torch.from_numpy(gpses)
            # attr_0 = scaler1.transform(torch.from_numpy(attr_0))
        # print("查看生成的特征:", attr_0.shape, gpses[:,0,:].shape, ratios[:, 0].reshape(-1, 1).shape,
        #       edges[:, 0].reshape(-1,1).shape, grid_index.shape)
        ## n * T * 2
        ##
        print(gpses[0,0])
        try:
            attr = np.concatenate([time_features[:, 0], attr_0, grid_index[:, 0], grid_index[:, -1]], axis=-1)
        except:
            attr = np.concatenate([time_features[:, 0], attr_0.numpy(), grid_index[:, 0], grid_index[:, -1]], axis=-1)
        print("查看特征:", time_features.shape, attr.shape)
        ####构建heads信息，这里构建多方面的信息
        try:
            assert len(gpses) == len(edges) == len(ratios) == len(time_features)
        except:
            print("查看这几个不同的地方:", len(gpses), time_features.shape)
        # edges = torch.from_numpy(edges).long()
        if dtype == 'train':
            all_needed_ids, id_num = torch.unique(torch.from_numpy(edges).long(), return_counts=True, sorted=True)
            assert 50429 in all_needed_ids
            new_edge_ids = torch.searchsorted(all_needed_ids.long(), torch.from_numpy(edges).long())
            graph = construct_frequency_graph_not_dup(new_edge_ids, all_needed_ids, raw_graph, step=120)
            all_data[dtype] = myDataset(gpses, grid_index, new_edge_ids,
                                    ratios, time_features, attr)
        else:
            new_edge_ids = torch.searchsorted(all_needed_ids.long(), torch.from_numpy(edges).long())
            all_data[dtype] = myDataset(gpses, grid_index, new_edge_ids,
                                        ratios, time_features, attr)
        logger.info(f'{dtype} data num is {len(grid_index)}')
    assert graph != None
    return all_data['train'], all_data['val'], all_data['test'], scaler_x, scaler1, graph, all_needed_ids.long(), id_num


def in_area(gps_slice, dataset_name):
    if dataset_name == 'chengdu':
        min_lng = 104.039
        max_lng = 104.128
        min_lat = 30.65
        max_lat = 31.731
    elif dataset_name == 'xian':
        min_lng = 108.91
        max_lng = 109.0
        min_lat = 34.20
        max_lat = 34.2802
    elif dataset_name == 'Porto':
        min_lng = -8.72
        max_lng = -8.52
        min_lat = 41.1
        max_lat = 41.2
    else:
        raise Exception("Wrong dataset!")
    for gps_point in gps_slice:
        # print(gps_point)
        if (gps_point[1] < min_lat or gps_point[1] > max_lat) or (gps_point[0] < min_lng or gps_point[0] > max_lng):
            return False
    return True

def get_small_area_idx(gps_slices, min_lat = 41.10, min_lng=-8.72,
               max_lat=41.20, max_lng=-8.52):
    """
    :param gps_slice: N, k, 2
    :param min_lat:
    :param min_lng:
    :param max_lat:
    :param max_lng:
    :return:
    """
    needed_idx = []
    for i, data in enumerate(gps_slices):
        if in_area(gps_slices[i],min_lat=min_lat, min_lng=min_lng,
               max_lat=max_lat, max_lng=max_lng):
            needed_idx.append(i)
    return needed_idx

def load_test_data(args, path):
    """
    :param path: 根据路网对于path进行处理
    :param dataset:
    :param logger:
    :param type_graph:
    :param device:

    data_indice: raw_POLYLINE, matched_POLYLINE, edge_list, Origin, Destination,
                 timestamp, start_time, end_time

    :return:
    """
    dtypes = ['train','test']

    LAT_PER_KILOMETER = 8.993203677616966e-06 * 1000
    LNG_PER_KILOMETER = 1.1700193970443768e-05 * 1000
    grid = create_grid_num(min_lat=41.077, min_lng=-8.8056, max_lat=41.221, max_lng=-8.4585,
                       nb_rows=16, nb_cols=16)

    # 读取数据 #
    all_data = dict()
    scaler = MinMaxScaler()
    scaler1 = MinMaxScaler()
    scaler2 = StandardScaler() ## 对于距离的判断
    for dtype in dtypes:
        data_path = os.path.join(path,'new_matched',dtype)
        time_path = f'{path}'
        file_names = os.listdir(data_path)

        gps_files = [file for file in file_names if 'matched_p_poly_special' in file]
        edge_files = [file for file in file_names if 'matched_edge_list' in file]
        ratio_files = [file for file in file_names if 'ratio' in file]
        hash_files = [file for file in file_names if 'hash_list' in file]
        print(gps_files)
        print(edge_files)
        print(ratio_files)
        print(hash_files)

        gps_files = sorted(gps_files, key=lambda x: (len(x), x))
        edge_files = sorted(edge_files, key=lambda x: (len(x), x))
        ratio_files = sorted(ratio_files, key=lambda x: (len(x), x))
        hash_files = sorted(hash_files, key=lambda x: (len(x), x))
        time_data = os.path.join(time_path, f'{dtype}_raw_time.pkl')

        gpses = []
        edges = []
        ratios = []

        data_index = 0

        with open(time_data, 'rb') as f:
            time_series = pickle.load(f)
        time_series_data = []
        time_index = []

        for index, gps_file in enumerate(gps_files):
            # needed_index = []
            i = index
            print("查看当前gps_file:", gps_files[i], edge_files[i], ratio_files[i], hash_files[i])
            with open(os.path.join(data_path, hash_files[i]), 'rb') as f:
                hash_series = pickle.load(f)  ###主要是对应用的
                print("查看hash_series的长度:",len(hash_series))
            with open(os.path.join(data_path, gps_files[i]), 'rb') as f:
                gps_series = pickle.load(f)
                # if dtype == 'train':
                #     if index <= 4:
                #         continue
                print("查看当前gps文件的长度:", len(gps_series))
                for j, gps_slice in enumerate(gps_series):
                    # gps_slice = ast.literal_eval(gps_slice)
                    gps_slice = np.array(gps_slice)


                    if gps_slice.shape[0] >= 120:
                        # print(np.array(gps_slice).shape)
                        # gpses.append(np.array(gps_slice[0:120]).astype(np.float32).reshape(1,120,2))
                        for k in range(120):
                            # print(gps_slice[k])
                            assert np.array(gps_slice[k]).shape[0] == 2
                            gpses.append(gps_slice[k])
                            time_series_data.append(time_series[hash_series[j]][k])
                    else:
                        continue
                ##get_longer_series
            with open(os.path.join(data_path, edge_files[i]), 'rb') as f:
                edge_series = pickle.load(f)
                for edge_slice in edge_series:
                    if len(edge_slice) >= 120:
                        edges.append(list(np.array(edge_slice)[0:120]))
                    else:
                        continue
                print("查看当前edges文件的长度:", len(edge_series))
                ##get_longer_series
            with open(os.path.join(data_path, ratio_files[i]), 'rb') as f:
                ratio_series = pickle.load(f)
                print("查看ratio_series的长度:", len(ratio_series))
                for ratio_slice in ratio_series:
                    if len(ratio_slice) >= 120:
                        ratios.append(list(np.array(ratio_slice)[0:120]))
                    else:
                        continue

            data_index += len(gps_series)
            data_index += len(gps_series)
            # print(time_series)

        # print(gpses[0][0], len(gpses[0]))
        gpses = np.array(gpses).reshape(-1, 120, 2)
        # gpses = [np.array(data) for data in gpses]
        edges = np.array(edges)
        ratios = np.array(ratios)

        print("查看以下length的长度:",gpses.shape, edges.shape, ratios.shape, len(time_series_data))
        gpses[...,[0,1]] = gpses[...,[1,0]]
        # print(gpses[0,0])
        # grid_index = grid.get_1d_idx(gpses.reshape(-1, 2))
        grid_index = torch.from_numpy(np.ones_like(gpses[...,0])).unsqueeze(-1)
        grid_index = grid_index.reshape(-1,120,1)
        # grid_index = np.array()
        print("查看grid_coords的形状:", grid_index.shape)
        # print(time_series[0])
        time_series = np.array(time_series_data)
        time_features = get_time_features(time_series)
        time_features = time_features.reshape(-1, 120, 1).astype(np.int64)
        attr_0 = calculate_attr_result(gpses,15)
        if dtype == 'train':
            if args.transform:
                scaler.fit()
                attr_0 = scaler1.fit_transform(torch.from_numpy(attr_0))
        else:
            gpses = torch.from_numpy(gpses)
            # scaler.fit(torch.from_numpy(gpses))
            attr_0 = scaler1.transform(torch.from_numpy(attr_0))
        print("查看生成的特征:", attr_0.shape, gpses[:,0,:].shape, ratios[:, 0].reshape(-1, 1).shape,
              edges[:, 0].reshape(-1,1).shape, grid_index.shape)
        ## n * T * 2
        ##
        try:
            attr = np.concatenate([time_features[:, 0], attr_0.numpy(), grid_index[:, 0], grid_index[:, -1]], axis=-1)
        except:
            attr = np.concatenate([time_features[:, 0], attr_0, grid_index[:, 0], grid_index[:, -1]], axis=-1)
        if dtype == 'train':
            all_needed_ids, id_num = torch.unique(torch.from_numpy(edges).long(), return_counts=True, sorted=True)
            assert 50429 in all_needed_ids
            new_edge_ids = torch.searchsorted(all_needed_ids.long(), torch.from_numpy(edges).long())
            all_data[dtype] = myDataset(gpses, grid_index, new_edge_ids,
                                    ratios, time_features, attr)
        else:
            new_edge_ids = torch.searchsorted(all_needed_ids.long(), torch.from_numpy(edges).long())
            all_data[dtype] = myDataset(gpses, grid_index, new_edge_ids,
                                        ratios, time_features, attr)
        print("查看特征:", time_features.shape, attr.shape)
        ####构建heads信息，这里构建多方面的信息
        try:
            assert len(gpses) == len(edges) == len(ratios) == len(time_features)
        except:
            print("查看这几个不同的地方:", len(gpses), time_features.shape)
        # all_data[dtype] = myDataset(gpses, grid_index, edges,
        #                             ratios, time_features, attr)
    # json_dict = json.load(open('raw2dict.json','r'))
    # print(json_dict)
    return all_data['test'], scaler, all_needed_ids

def seq_collate(batch):
    gps_traj, grid_ids, edge_ids, ratios, time_feats, side_attr = zip(*batch)
    gps_traj = torch.tensor(np.array(gps_traj).astype(np.float32), device=device)
    grid_ids = torch.tensor(np.array(grid_ids).astype(np.int64), device=device)
    # print("查看Grid_ids的形状:", grid_ids.shape)
    edge_ids = torch.tensor(np.array(edge_ids).astype(np.int64), device=device)
    ratios = torch.tensor(np.array(ratios).astype(np.float32), device=device)
    time_feats = torch.tensor(np.array(time_feats), device=device)
    side_attr = torch.tensor(np.array(side_attr).astype(np.float32), device=device)
    return gps_traj, grid_ids, edge_ids, ratios, time_feats, side_attr

def convert_back(edge_ids, all_unique_ids):
    for j in range(len(edge_ids)):
        for i,data in enumerate(edge_ids[j]):
            edge_ids[j][i] = all_unique_ids[data.long()]
    return edge_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser('parameter_selection')
    parser.add_argument('--training_epoch', type=int, default=200, help='training epoch')
    parser.add_argument('--device', type=str, default='cuda:2' if torch.cuda.is_available() else 'cpu')
    # parser.add_argument('--another_device', type=str, default='cuda:3' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_name', type=str, default='Transformer', help='the model name')
    # parser.add_argument('--device', type=str, default='cuda:3' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_temb', action='store_true', help='whether use temb')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning_rate')
    parser.add_argument('--use_time', action='store_true', help='use_time_feats')
    parser.add_argument('--emb_dim', type=int, default=128, help='hid_emb_dim')
    parser.add_argument('--layer', type=int, default=1, help='layer_num')
    parser.add_argument('--bs', type=int, default=512, help='batch_size')
    parser.add_argument('--use_gps', action='store_true', help='whether to use gps_traj')
    parser.add_argument('--not_proj', action='store_true', help='whether to proj the raw emb')
    parser.add_argument('--step', type=int, default=120, help='step_lens')
    parser.add_argument('--transform', action='store_true', help='whether to scale')
    parser.add_argument('--use_x0_loss', action='store_true', help='whether to use x0')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--norm_type', type=str, default='Batch', help='normalization_type')
    parser.add_argument('--mlp_layer', type=int, default=1, help='get the layer for MLP')
    parser.add_argument('--kernel_size', type=int, default=5, help='conv kernel size')
    parser.add_argument('--pretrain', action='store_true', help='whether to pretrain')
    parser.add_argument('--use_len', action='store_true', help='use the length feature')

    parser.add_argument('--use_road_emb', action='store_true', help='get_gps_emb')
    parser.add_argument('--only_noise', action='store_true', help='whether to use only the noise loss')
    parser.add_argument('--use_ratio', action='store_true', help='whether to use ratio emb for full road')
    parser.add_argument('--device_ids', type=int, nargs='+', default=[2,2],
                        help='GPU device ids to use')
    ### data_fort: gps_traj
    device = 'cpu'
    from tqdm import tqdm
    args = parser.parse_args()
    preprocess_index('../data')
    # test_data, scaler, all_needed_ids = load_test_data(args, '../data')
    #
    # graph = load_graph('road_graph', 'Porto', 'True')
    # (_, all_node_coords, edge_neighbors, node_lens,
    #  edge_lens, road_length, road_type, row_num, col_num) = load_dgl_graph_v5(
    #     graph, all_needed_ids, False)
    #
    # test_loader = DataLoader(test_data, batch_size=args.bs, shuffle=True, collate_fn=seq_collate)
    # for gps_traj, grid_ids, edge_ids, ratios, time_feats, head in tqdm(test_loader):
    #     # edge_ids = convert_back(edge_ids, all_needed_ids)
    #     for i in range(len(edge_ids)):
    #         print(i)
    #         single_result = rate2gps_torch(torch.from_numpy(all_node_coords), torch.from_numpy(road_length), edge_ids[i], edge_lens,
    #                                    ratios[i])
    #         if i == 4:
    #             print("这里的部分是展示GPS序列的内容:")
    #             print(single_result[:10])
    #             print(gps_traj[i][:10])
    #             break
    # pass