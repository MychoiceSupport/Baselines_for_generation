import torch
import numpy as np
import torch.nn as nn
import pandas as pd

class TimeEncoder(torch.nn.Module):
    def __init__(self, dim, device):
        super(TimeEncoder, self).__init__()
        self.dim = dim
        self.device = device

    def time_encode(self, times):
        ...


class DeltaEncoder(TimeEncoder):
    def __init__(self, dimension, device):
        super(DeltaEncoder, self).__init__(dimension, device)
        self.w = nn.Linear(10, self.dim)

    def time_encode(self, times):
        return torch.cos(self.w(times))


# Semantic时间编码函数
class EmbEncoder(TimeEncoder):
    # Time Encoding proposed by TGAT
    def __init__(self, dimension, device):
        super(EmbEncoder, self).__init__(dimension, device)
        self.hour_emb = nn.Embedding(24, self.dim)
        self.min_emb = nn.Embedding(60, self.dim)
        self.sec_emb = nn.Embedding(60, self.dim)
        self.day_emb = nn.Embedding(7, self.dim)
        self.weekday_emb = nn.Embedding(2, self.dim)

    def time_encode(self, times):
        times = times.to(dtype=torch.long)
        # times: [batch,6]
        hour_emb = torch.sum(self.hour_emb(times[:, [0, 5]]), dim=1)
        min_emb = torch.sum(self.min_emb(times[:, [1, 6]]), dim=1)
        sec_emb = torch.sum(self.sec_emb(times[:, [2, 7]]), dim=1)
        day_emb = torch.sum(self.day_emb(times[:, [3, 8]]), dim=1)
        weekday_emb = torch.sum(self.weekday_emb(times[:, [4.9]]), dim = 1)

        return hour_emb + min_emb + sec_emb + day_emb + weekday_emb


class MLPEncoder(TimeEncoder):
    def __init__(self, dimension, device):
        super(MLPEncoder, self).__init__(dimension, device)
        self.encoder = nn.Sequential(nn.Linear(10, self.dim // 2), nn.ReLU(), nn.Linear(self.dim // 2, self.dim))

    def time_encode(self, times):
        return self.encoder(times)

class DayTimeEncoder(TimeEncoder):
    def __init__(self, dimension, device):
        super(DayTimeEncoder, self).__init__(dimension, device)
        self.dim = dimension
        self.hsmEncoder = nn.Sequential(nn.Linear(3, self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim))
        # self.dayEncoder = nn.Sequential(nn.Linear(4, self.dim // 2), nn.ReLU(), nn.Linear(self.dim // 2, self.dim))
        self.dayEncoder =  nn.Embedding(7, self.dim)
        self.weekDayEncoder = nn.Embedding(2, self.dim)

    def time_encode(self, times):
        """
        :param times: day, hour, minute, second, is_weekday
        :return:
        """
        xq_feats = torch.tensor(times[...,-2], dtype=torch.int64)
        week_feats = torch.tensor(times[...,-1], dtype=torch.int64)

        day_feats = torch.cat([torch.cos(self.hsmEncoder(times[...,0:3])), torch.sin(self.hsmEncoder(times[..., 0:3]))], dim=-1)

        xq_feats = self.dayEncoder(xq_feats)
        week_feats = self.weekDayEncoder(week_feats)
        # print("查看各种特征形状:", day_feats.shape, xq_feats.shape, week_feats.shape)
        ### 4 * dim
        return torch.cat([day_feats, xq_feats, week_feats], dim = -1)

class DayEncoder(TimeEncoder):
    def __init__(self, dimension, device):
        super(DayEncoder, self).__init__(dimension, device)
        self.dim = dimension
        self.hidden = nn.Linear(3, self.dim // 2)

    def forward(self, t):
        time_emb = self.hidden(t[:,:,:3].float())
        time_emb = torch.cat([torch.cos(time_emb), torch.sin(time_emb)], dim=-1)
        return time_emb

def get_time_encoder(encoder_type, dimension, device):
    if encoder_type == 'delta':
        return DeltaEncoder(dimension, device)
    elif encoder_type == 'emb':
        return EmbEncoder(dimension, device)
    elif encoder_type == 'mlp':
        return MLPEncoder(dimension, device)
    elif encoder_type == 'Int':
        return DayTimeEncoder(dimension, device)
    elif encoder_type == 'day':
        return DayEncoder(dimension, device)
    else:
        raise ValueError("Not implemented time encoder")
