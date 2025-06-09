import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import pickle as pk
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", module="matplotlib\..*")


def geodesic(pos1, pos2):
    EARTH_RADIUS = 6371393
    """用haversine公式计算球面两点间的距离。"""
    lat0 = torch.deg2rad(pos1[:, 0])
    lat1 = torch.deg2rad(pos2[:, 0])
    lng0 = torch.deg2rad(pos1[:, 1])
    lng1 = torch.deg2rad(pos2[:, 1])
    dLat = lat1 - lat0
    dLon = lng1 - lng0
    a = torch.sin(dLat / 2) * torch.sin(dLat / 2) + torch.sin(dLon / 2) * torch.sin(dLon / 2) * torch.cos(
        lat0) * torch.cos(lat1)
    distance = 2 * EARTH_RADIUS * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return distance


def get_metrics(x, y):
    assert len(x) == len(y)
    counter = len(x)
    dis = geodesic(x, y)
    count = len(dis[dis <= 150.0])
    dis = torch.tensor(dis, requires_grad=True)
    rmse = torch.sqrt(torch.mean(torch.pow(dis, 2)))
    mae = torch.mean(dis)
    rate_150 = count / counter * 100
    return rmse.item(), mae.item(), rate_150


def save_model(model: nn.Module, save_path, run):
    torch.save(model.state_dict(), f'{save_path}_{run}.pth')


def load_model(model: nn.Module, load_path, run):
    model_dict = torch.load(f"{load_path}_{run}.pth")
    model.load_state_dict(model_dict)


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10, model_path=None, logger=None,
                 model: nn.Module = None,
                 run=0):
        self.max_round = max_round
        self.num_round = 0
        self.run = run

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance
        self.model_path = model_path
        self.logger = logger
        self.model = model

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
            save_model(self.model, self.model_path, self.run)
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
            save_model(self.model, self.model_path, self.run)
        else:
            self.num_round += 1
        self.epoch_count += 1
        if self.num_round <= self.max_round:
            return False
        return True


class Metric:
    def __init__(self, path, logger, fig_path):
        self.template = {'target': [], 'pred': [], 'label': [], 'rmse': 0, 'mae': 0, 'rating': 0, 'loss': 0}
        self.final = {'train': deepcopy(self.template), 'val': deepcopy(self.template), 'test': deepcopy(self.template)}
        self.history = {'train': {'rmse': [], 'mae': [], 'rating': [], 'loss': []},
                        'val': {'rmse': [], 'mae': [], 'rating': [], 'loss': []},
                        'test': {'rmse': [], 'mae': [], 'rating': [], 'loss': []},
                        }
        self.avg_metric = {'rmse': [], 'mae': [], 'rating': [], 'loss': []}
        self.path = path
        self.fig_path = fig_path
        self.logger = logger

    def finish(self):
        return_data = dict()
        for metric in ['rmse', 'mae', 'rating', 'loss']:
            return_data[metric] = np.mean(self.avg_metric[metric])
        return return_data

    def finish_run(self):
        for metric in ['rmse', 'mae', 'rating', 'loss']:
            self.avg_metric[metric].append(self.final['test'][metric])
        self.final = {'train': deepcopy(self.template), 'val': deepcopy(self.template), 'test': deepcopy(self.template)}
        self.history = {'train': {'rmse': [], 'mae': [], 'rating': [], 'loss': []},
                        'val': {'rmse': [], 'mae': [], 'rating': [], 'loss': []},
                        'test': {'rmse': [], 'mae': [], 'rating': [], 'loss': []},
                        }

    def fresh(self):
        self.temp = {'train': deepcopy(self.template), 'val': deepcopy(self.template), 'test': deepcopy(self.template)}

    def update(self, target, pred, label, dtype):
        self.temp[dtype]['target'].append(target)
        self.temp[dtype]['pred'].append(pred)
        self.temp[dtype]['label'].append(label)

    def caculate_metric(self, dtype, move_history=True, move_final=False, loss=0):
        targets, preds, labels = self.temp[dtype]['target'], self.temp[dtype]['pred'], self.temp[dtype]['label']

        targets, preds, labels = np.concatenate(targets, axis=0), \
                                 np.concatenate(preds, axis=0), \
                                 np.concatenate(labels, axis=0)
        rmse, mae, rating = get_metrics(torch.from_numpy(preds), torch.from_numpy(labels))
        self.temp[dtype]['target'] = targets
        self.temp[dtype]['pred'] = preds
        self.temp[dtype]['label'] = labels
        self.temp[dtype]['rmse'] = np.around(rmse, 2)
        self.temp[dtype]['mae'] = np.around(mae, 2)
        self.temp[dtype]['rating'] = np.around(rating, 2)
        self.temp[dtype]['loss'] = np.around(loss, 4)

        if move_history:
            for metric in ['rmse', 'mae', 'rating', 'loss']:
                self.history[dtype][metric].append(self.temp[dtype][metric])
        if move_final:
            self.move_final(dtype)
        return deepcopy({x: self.temp[dtype][x] for x in ['loss', 'rmse', 'mae', 'rating']})

    def move_final(self, dtype):
        self.final[dtype] = self.temp[dtype]

    def save(self, run):
        pk.dump(self.final, open(f'{self.path}_{run}.pkl', 'wb'))

    def plot(self, run):
        fig = plt.figure()
        epochs = np.arange(len(self.history['train']['loss']))
        for i, metric in enumerate(['loss', 'rmse', 'mae', 'rating']):
            plt.subplot(2, 2, i + 1)
            train_metric = np.stack(self.history['train'][metric], axis=0)
            val_metric = np.stack(self.history['val'][metric], axis=0)
            test_metric = np.stack(self.history['test'][metric], axis=0)
            plt.plot(epochs, train_metric, label=f'train')
            plt.plot(epochs, val_metric, label=f'val')
            plt.plot(epochs, test_metric, label=f'test')
            plt.title(metric)
        plt.legend()
        plt.suptitle(f'runs_{run}')
        plt.tight_layout()
        plt.savefig(f'{self.fig_path}_{run}.png')

    def info(self, dtype):
        s = []
        for metric in ['loss', 'rmse', 'mae', 'rating']:
            s.append(f'{metric}:{self.temp[dtype][metric]:.4f}')
        self.logger.info(f'{dtype}: ' + '\t'.join(s))
