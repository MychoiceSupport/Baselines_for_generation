import torch
import torch.nn as nn
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
from einops import repeat, rearrange
import os
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from types import SimpleNamespace
from utils.config import args
from utils.EMA import EMAHelper
from utils.road_encoder import *
from utils.geounet import *
from utils.logger import Logger, log_info
from pathlib import Path
import shutil
from utils.dataloader import *
from tqdm import tqdm
from common.construct_road_network_csv import load_rn_csv
from utils.road_seq_emb_pretrained_best_now import *
from utils.road_net_emb import *
from utils import road_net_emb

# set the GPU environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class MinMaxScaler:
    def __init__(self, min_lat=41.077, min_lng=-8.8056, max_lat=41.221, max_lng=-8.4585):
        self.min_lat = min_lat
        self.min_lng = min_lng
        self.max_lat = max_lat
        self.max_lng = max_lng

    def transform(self, data):
        raw_data = data
        raw_data[...,0] = (data[...,0] - self.min_lat) / (self.max_lat - self.min_lat)
        raw_data[...,1] = (data[...,1] - self.min_lng) / (self.max_lng - self.min_lng)
        return raw_data

    def inverse_transform(self, data):
        raw_data = data
        lat_std = data[..., 0]
        lng_std = data[..., 1]
        raw_data[..., 0] = lat_std * (self.max_lat - self.min_lat) + self.min_lat
        raw_data[..., 1] = lng_std * (self.max_lng - self.min_lng) + self.min_lng
        return raw_data
def load_graph(file_path='/home/huanghy/python_directories/Trajectory/road_graph',
               city_name='Porto', is_directed= True):
    file_path_name = f'{file_path}/{city_name}/{city_name}_{is_directed}_index_another.pkl'
    with open(file_path_name, 'rb') as f:
        graph = pickle.load(f)
    return graph


class MinMaxScaler:
    def __init__(self, min_lat=41.077, min_lng=-8.8056, max_lat=41.221, max_lng=-8.4585):
        self.min_x = min_lat
        self.max_x = max_lat
        self.min_y = min_lng
        self.max_y = max_lng

    def fit(self,min_lat=41.077, min_lng=-8.8056, max_lat=41.221, max_lng=-8.4585):
        self.min_x = min_lat
        self.max_x = max_lat
        self.min_y = min_lng
        self.max_y = max_lng
    def transform(self, data: torch.Tensor):
        """
        对数据进行标准化
        """
        data[..., 0] = (data[..., 0] - self.min_x) / (self.max_x - self.min_x)
        data[..., 1] = (data[..., 1] - self.min_y) / (self.max_y - self.min_y)
        return data

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
        data[..., 0] = data[..., 0] * (self.max_x - self.min_x) + self.min_x
        data[..., 1] = data[..., 1] * (self.max_y - self.min_y) + self.min_y
        return data


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: torch.Tensor):
        """
        计算数据的均值和标准差
        """
        self.mean = torch.mean(data, dim=-1)
        self.std = torch.std(data, dim=-1)
        print(self.mean.shape, self.std.shape)

    def transform(self, data: torch.Tensor):
        """
        对数据进行标准化
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet. Call `fit` first.")
        data[..., 0] = (data[..., 0] - self.mean[0]) / (self.std[0] + 1e-8)
        data[..., 1] = (data[..., 1] - self.mean[1]) / (self.std[1] + 1e-8)
        return (data - self.mean) / (self.std + 1e-8)

    def fit_transform(self, data: torch.Tensor):
        """
        计算均值和标准差，并标准化数据
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: torch.Tensor):
        """
        将标准化数据还原为原始分布
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet. Call `fit` first.")
        data[..., 0] = data[..., 0] * self.std[0] + self.mean[0]
        data[..., 1] = data[..., 1] * self.std[1] + self.mean[1]
        return data


def resample_trajectory(x, length=100):
    """
    Resample a trajectory to a fixed length using linear interpolation
    :param x: trajectory to resample
    :param length: length of the resampled trajectory
    :return: resampled trajectory
    """
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    resampled_trajectory = np.zeros((2, length))
    for i in range(2):
        resampled_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return resampled_trajectory.T

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1).to(device)
def compute_alpha(beta, t):
    """
    compute alpha for a given beta and t
    :param beta: tensor of shape (T,)
    :param t: tensor of shape (B,)
    :return: tensor of shape (B, 1, 1)
    """
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a

def p_xt(xt, noise, t, next_t, beta, eta=0):
    at = compute_alpha(beta.to(device), t.long())
    at_next = compute_alpha(beta, next_t.long())
    x0_t = (xt - noise * (1 - at).sqrt()) / at.sqrt()
    c1 = (eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    eps = torch.randn(xt.shape, device=xt.device)
    xt_next = at_next.sqrt() * x0_t + c1 * eps + c2 * noise
    return xt_next


def setup_experiment_directories(config, Exp_name='ControlTraj', model_name="ControlTraj"):
    """
    setup the directories for the experiment
    :param config: configuration file
    :param Exp_name: Experiment name
    file_save: directory to save the files
    result_save: directory to save the results
    model_save: directory to save the models during training
    """
    root_dir = Path(__file__).resolve().parent
    result_name = f"{config.data.dataset}_bs={config.training.batch_size}"
    exp_dir = root_dir / Exp_name / result_name
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    exp_time_dir = exp_dir / timestamp
    files_save = exp_time_dir / 'Files'
    result_save = exp_time_dir / 'Results'
    model_save = exp_time_dir / 'models'

    # Creating directories
    for directory in [files_save, result_save, model_save]:
        directory.mkdir(parents=True, exist_ok=True)

    # Copying files
    for filename in os.listdir(root_dir / 'utils'):
        if filename.endswith('.py'):
            shutil.copy(root_dir / 'utils' / filename, files_save)
    # Copying the current file itself
    this_file = Path(__file__)
    shutil.copy(this_file, files_save)

    print("All files saved path ---->>", exp_time_dir)
    logger = Logger( __name__, log_path=exp_dir /  (timestamp + '/out.log'),colorize=True)
    return logger, files_save, result_save, model_save


def seq_collate(batch):
    gps_traj, grid_ids, edge_ids, ratios, time_feats, side_attr = zip(*batch)
    gps_traj = torch.tensor(np.array(gps_traj), device=device)
    grid_ids = torch.tensor(np.array(grid_ids).astype(np.int64), device=device)
    # print("查看Grid_ids的形状:", grid_ids.shape)
    edge_ids = torch.tensor(np.array(edge_ids).astype(np.int64), device=device)
    ratios = torch.tensor(np.array(ratios).astype(np.float32), device=device)
    time_feats = torch.tensor(np.array(time_feats), device=device)
    side_attr = torch.tensor(np.array(side_attr).astype(np.float32), device=device)
    return gps_traj, grid_ids, edge_ids, ratios, time_feats, side_attr

def main(config, logger, device):
        # Modified to return the noise itself as well

    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start,
                              config.diffusion.beta_end, n_steps).to(device)
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    def q_xt_x0(x0, t):
        mean = gather(alpha_bar, t)**0.5 * x0
        var = 1 - gather(alpha_bar, t)
        eps = torch.randn_like(x0).to(x0.device)
        return mean + (var**0.5) * eps, eps  # also returns noise

    #initialize_graph
    graph = load_graph()

    edge_sorted = sorted(graph.edges(data=True), key=lambda x: x[2]['eid'])
    i = 0
    lens_cal = []
    edge_coords = []
    edge_feats = np.zeros((53826, 100, 2))
    scaler = MinMaxScaler()
    for u, v, data in edge_sorted:
            # data = edge_sorted[j]
        if i == 0:
            print(data)
        cord = []
        for k in range(len(data['coords'])):
            cord.append([data['coords'][k].lat, data['coords'][k].lng])
        cord = np.array(cord)
        lens_cal.append(len(cord))
        cord = resample_trajectory(cord, length=100)  ###长度补充，作为edge的预训练
        edge_coords.append(cord)
            # cord = np.mean(cord, axis = 1)
            ###获取每条边的内部的ID个数
            # edge_feats.append(np.array([cord, data['length'], data['eid']]))
        edge_feats[data['eid']] = edge_feats[data['eid']] + np.array(cord)
            # edge_index.append((data['u'], data['v']))


    edge_feats = torch.tensor(edge_feats).to(device)
    edge_feats = scaler.transform(edge_feats)
    # initialize the model with the configuration
    unet = UNetModel(
        in_channels = config.model.in_channels,
        out_channels = config.model.out_channels,
        channels = config.model.channels,
        n_res_blocks = config.model.num_res_blocks,
        attention_levels = config.model.attention_levels,
        channel_multipliers = config.model.channel_multipliers,
        n_heads = config.model.n_heads,
        tf_layers = config.model.tf_layers,
        d_cond=32
    ).to(device)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f'{total_params:,} total parameters.')
    # initialize the road encoder with RoadMAE


    autoencoder =  Graph_MAE_ViT(args, graph,
        image_size = 100,
        patch_size = 4,  ###only mask_some
        emb_dim = 128,
        encoder_layer = 12,
        encoder_head = 4,
        decoder_layer = 12,
        decoder_head = 4,
        mask_ratio = 0.00,
        pre_trained = False,
        device = device)


    lr = 4e-4  # Explore this - might want it lower when training on the full dataset

    losses = []  # Store losses for later plotting
    # optimizer
    optim = torch.optim.AdamW(unet.parameters(), lr=lr)  # Optimizer
    
    # EMA
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(unet)
    else:
        ema_helper = None

    train_data, val_data, test_data, scaler = get_all_data("../data", logger)
        # Training params
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=seq_collate)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True, collate_fn=seq_collate)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True, collate_fn=seq_collate)

    autoencoder.load_state_dict(
        torch.load('/home/huanghy/python_directories/Trajectory/TEVI/vit-t-mae.pt', map_location=device))
    autoencoder.to(device)
    ###这里应该需要生成轨迹
    # config.training.n_epochs = 1

    best_loss = 1e9
    for epoch in range(0, config.training.n_epochs):
        losses = []  # Store losses for later plotting
        logger.info("<----Epoch-{}---->".format(epoch))

        unet.train()

        for gps_traj, _, edge_ids, _, time_feats, head in tqdm(train_loader):
            torch.cuda.empty_cache()
            x0 = gps_traj
            x0 = x0.transpose(1, 2)
            batch, step = gps_traj.shape[0], gps_traj.shape[1]
            # for i in range(len(road)):
            #     new_roads.append(resample_trajectory(road[i]))
            new_roads = edge_feats[edge_ids]
            # print("查看new_roads.shape:", new_roads.shape)
            new_roads = new_roads.reshape(-1, 100, 2)
            # guide = new_roads.permute(0, 2, 1)
            # guide = torch.from_numpy(new_roads).to(device)
            guide = scaler.transform(new_roads.float())
            # get the road embeddings by RoadMAE
            ## 每个路段表示成为n * 2个embedding
            # guide = torch.from_numpy(new_roads).float()
            with torch.no_grad():

                guide = autoencoder.pre_forward(guide)
                guide = guide.transpose(1,2)
                guide, _ = autoencoder.encoder(guide)
                guide = guide[1:, :, :]
                guide = rearrange(guide, 't b c -> b t c')

                guide = guide.view(batch, step, 100, -1)
                guide = torch.mean(guide, dim = 2)
            # print("查看guide的形状:",guide.shape)
            ###由于原本的模型并没有告诉我们data的情况，因此只能采取edge_id是我们自己搞的

            t = torch.randint(low=0, high=n_steps,
                              size=(len(x0) // 2 + 1,)).to(device)
            t = torch.cat([t, n_steps - t - 1], dim=0)[:len(x0)].to(device)
            # Get the noised images (xt) and the noise (our target)
            xt, noise = q_xt_x0(x0, t)
            # print("分别查看形状:", t.shape, guide.shape, head.shape)
            # print("分别查看形状:", xt.shape, t.shape, noise.shape)
            pred_noise = unet(xt.float(), t, guide, head)
            # Compare the predictions with the targets
            loss = F.mse_loss(noise.float(), pred_noise)
            # Store the loss for later viewing
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            if config.model.ema:
                ema_helper.update(unet)
        train_loss = np.mean(losses)
        logger.info(f"The result of the train_dataset is: {train_loss}")


        unet.eval()
        losses = []
        for gps_traj, _, edge_ids, _, time_feats, head in tqdm(val_loader):
            torch.cuda.empty_cache()
            x0 = gps_traj
            x0 = x0.transpose(1, 2)
            batch, step = gps_traj.shape[0], gps_traj.shape[1]
            new_roads = edge_feats[edge_ids]
            # print("查看new_roads.shape:", new_roads.shape)
            new_roads = new_roads.reshape(-1, 100, 2)
            # guide = new_roads.permute(0, 2, 1)
            # guide = torch.from_numpy(new_roads).to(device)
            guide = new_roads.float()
            # get the road embeddings by RoadMAE
            ## 每个路段表示成为n * 2个embedding
            # guide = torch.from_numpy(new_roads).float()
            with torch.no_grad():
                guide = autoencoder.pre_forward(guide)
                guide = guide.transpose(1, 2)
                guide, _ = autoencoder.encoder(guide)
                guide = guide[1:, :, :]
                guide = rearrange(guide, 't b c -> b t c')
                guide = guide.view(batch, step, 100, -1)
                guide = torch.mean(guide, dim=2)


            t = torch.randint(low=0, high=n_steps,
                              size=(len(x0) // 2 + 1, )).to(device)
            t = torch.cat([t, n_steps - t - 1], dim=0)[:len(x0)].to(device)
            # Get the noised images (xt) and the noise (our target)
            xt, noise = q_xt_x0(x0, t)
            pred_noise = unet(xt.float(), t, guide, head)
            # Compare the predictions with the targets
            loss = F.mse_loss(noise.float(), pred_noise)
            # Store the loss for later viewing
            losses.append(loss.item())
            # optim.zero_grad()
            # loss.backward()
            # optim.step()
            if config.model.ema:
                ema_helper.update(unet)
        val_loss = np.mean(losses)
        logger.info(f"The result of the train_dataset is: {val_loss}")
        

        if best_loss > val_loss:
            all_trajs = []
            best_loss = val_loss
            m_path = model_save / f"ControlTraj_result.pt"
            torch.save(unet.state_dict(), m_path)
            # Start with random noise

            all_gps_traj, all_guide, all_head = [], [], []
            for gps_traj, _, edge_ids, _, time_feats, head in tqdm(test_loader):
                batch = gps_traj.shape[0]
                # print("查看形状:", len(gps_traj), len(edge_ids))
                batch, step = gps_traj.shape[0], gps_traj.shape[1]
                new_roads = edge_feats[edge_ids]
                # print("查看new_roads.shape:", new_roads.shape)
                new_roads = new_roads.reshape(-1, 100, 2)
                # guide = new_roads.permute(0, 2, 1)a
                guide = new_roads.float()
                # get the road embeddings by RoadMAE
                ## 每个路段表示成为n * 2个embedding
                # guide = torch.from_numpy(new_roads).float()
                with torch.no_grad():
                    guide = autoencoder.pre_forward(guide)
                    guide = guide.transpose(1, 2)
                    guide, _ = autoencoder.encoder(guide)
                    guide = guide[1:, :, :]
                    guide = rearrange(guide, 't b c -> b t c')
                    guide = guide.view(batch, step, 100, -1)
                    guide = torch.mean(guide, dim=2)

                all_gps_traj.append(gps_traj)
                all_guide.append(guide)
                all_head.append(head)
            guide = torch.cat(all_guide, dim=0)
            head = torch.cat(all_head, dim = 0)
            sample = torch.randn(guide.shape[0], 2, 120).to(device)
            ims = []
            n = sample.size(0)
            eta=0.0
            timesteps=100
            skip = n_steps // timesteps
            seq = range(0, n_steps, skip)
            seq_next = [-1] + list(seq[:-1])
            # print("查看一下形状:", seq)
            # print("查看一下形状:", seq_next)


            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(device)
                next_t = (torch.ones(n) * j).to(device)
                with torch.no_grad():
                    # print("分别查看形状:",sample.shape, t.shape, guide.shape, head.shape)
                    pred_noise = unet(sample, t, guide, head)
                    # print(pred_noise.shape)
                    sample = p_xt(sample, pred_noise, t, next_t, beta, eta)
                    if i % 10 == 0:
                        ims.append(sample.squeeze(0))
            trajs = ims[-1].cpu().numpy()
            print(np.array(all_trajs).shape)
            all_trajs.append(trajs) ###注意变化，每一次
            print(np.array(all_trajs).shape)
            np.save(f'ControlTraj_trajs_MinMax.npy', np.array(all_trajs))

            del ims
            # plt.figure(figsize=(8,8))
            # for i in range(len(trajs)):
            #     tj = trajs[i]
            #     plt.plot(tj[0,:],tj[1,:],color='#3f72af',alpha=0.1)
            # plt.tight_layout()
            # m_path = result_save / f"r_{epoch}.png"
            # plt.savefig(m_path)
        
            
if __name__ == "__main__":
    # Load configuration
    import argparse
    from utils import config
    parser = argparse.ArgumentParser('parameter_selection')
    parser.add_argument('--device', type=str, default='cuda:2' if torch.cuda.is_available() else 'cpu')
    # parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    device = args.device

    args = config.args

    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)
    config = SimpleNamespace(**temp)
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger,files_save, result_save, model_save = setup_experiment_directories(config, Exp_name='Control_Porto')
    
    log_info(config, logger)
    main(config, logger, device)
