import torch
import torch.nn as nn
import numpy as np
import math
import sys
import os
sys.path.append('../')
sys.path.append(os.path.join(os.getcwd(), 'utils'))
import utils
print(sys.path)
import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from types import SimpleNamespace
from utils.config import raw_args
from utils.EMA import EMAHelper
from utils.Traj_UNet_MAE import *
from utils.logger import Logger, log_info
from pathlib import Path
import shutil
from utils.dataloader import *
from tqdm import tqdm
import dgl
import time
from road_mae_encoder import MAE_ViT, get_parse_args
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
# This code part from https://github.com/sunlin-ai/diffusion_tutorial


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumpred(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1).to(device)


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
# departure_time, trip_distance,  trip_time, trip_length, avg_dis, avg_speed

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

def load_mae_model(args, device):
    if args.use_time:
        from road_mae_encoder_with_time import MAE_ViT, get_parse_args
        model_name = f'GPS_Pretrain_std_{args.dataset}_with_time_mask_ratio_{args.mask_ratio}_patch_size_{args.patch_size}.pt'
    else:
        print("使用没有时间的")
        model_name = f'GPS_Pretrain_std_without_time_{args.dataset}_{args.use_time}_{args.emb_dim}_{args.mask_ratio}_patch_size_{args.patch_size}.pt'
    print("查看你参数",args.patch_size)
    model_dict = torch.load(model_name, map_location=device)
    mae_model = MAE_ViT(args=args,
                    image_size=args.image_size,
                    patch_size=args.patch_size,
                    emb_dim=args.emb_dim,
                    encoder_layer=args.encoder_layer,
                    encoder_head=args.encoder_head,
                    decoder_layer=args.decoder_layer,
                    decoder_head=args.decoder_head,
                    mask_ratio=0.00,
                    input_dim=2,
                    ).to(device)
    mae_model.load_state_dict(model_dict)
    return mae_model

def evaluate_loader(data_loader, unet, mae_model, optimizer, n_steps, q_xt_x0, p_x0_xt,
                    ema_helper=None, mode='train', epoch = 0, args = None, scaler=None):
    losses = []
    i = 0
    generator = torch.Generator(device=args.device)
    for gps_traj, grid_ids, edge_ids, ratios, time_feats, head in tqdm(data_loader):
        i = i + 1
        torch.cuda.empty_cache()
        batch, step = gps_traj.shape[0], gps_traj.shape[1]
        if args.transform:
            x0 = scaler.transform(gps_traj)
        else:
            x0 = gps_traj
        x0 = x0.transpose(1, 2)
        # generator.manual_seed(args.seed + epoch)
        t = torch.randint(low=0, high=n_steps, size=(len(x0) // 2 + 1,),device=args.device)
        t = torch.cat([t, n_steps - t - 1], dim=0)[:len(x0)].to(device)
        xt, noise = q_xt_x0(x0, t)
        torch.cuda.empty_cache()
        # Run xt through the network to get its predictions
        pred_noise = unet(xt.float(), t, head)
        # Compare the predictions with the targets
        loss = F.mse_loss(pred_noise.transpose(-2,-1), noise.float().transpose(-2,-1))
        # final_x0 = scaler.inverse_transform(recon_x0)
        losses.append(loss.item())
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if ema_helper != None:
            ema_helper.update(unet)
        torch.cuda.empty_cache()
    train_loss = np.mean(losses)
    logger.info(
            f'The result of the {mode} at epoch {epoch} is {train_loss}.')
    return train_loss


def generate_loader(data_loader, unet, n_steps, beta, eta, dataset_name,
                    args = None, scaler = None):
    print("查看scaler的特征:",scaler.mean, scaler.std)
    k = 0
    skip = 5
    all_trajs = []
    for gps_traj, grid_ids, edge_ids, ratios, time_feats, head in tqdm(data_loader):
        if args.transform:
            gps_traj = scaler.transform(gps_traj)
        print(gps_traj[0,0])
        k = k + 1
        torch.cuda.empty_cache()
        batch, step = gps_traj.shape[0], gps_traj.shape[1]
        setup_seed(k * 1000)
        x = torch.randn(batch, 2, step).to(args.device)
        ims = []
        seq = range(0, n_steps, skip)
        seq_next = [-1] + list(seq[:-1])
        # x = x.transpose(-2,-1)
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(batch) * i).to(x.device)
            next_t = (torch.ones(batch) * j).to(x.device)
            with torch.no_grad():
                pred_noise = unet(x, t, head)
                # print(pred_noise.shape)
                x = p_xt(x, pred_noise, t, next_t, beta.to(args.device), eta)
                if i % 10 == 0:
                    ims.append(x.cpu().squeeze(0))
        trajs = ims[-1]
        trajs = trajs.transpose(-2,-1)
        print(trajs[0,0])
        trajs = scaler.inverse_transform(trajs)
        print(trajs[0,0])
        all_trajs.append(trajs)
    all_trajs = np.concatenate(all_trajs, axis=0)
    np.save(f'DiffTraj_newest_{args.dataset}_use_cond.npy', np.array(all_trajs))
    return all_trajs

def generate_data_with_mae(data_loader, unet, n_steps, beta, eta, dataset_name, mae_model,
                    args = None, scaler = None):
    print("查看scaler的特征:", scaler.mean, scaler.std)
    k = 0
    skip = 5
    all_trajs = []
    unet.eval()
    mae_model.eval()
    for gps_traj, grid_ids, edge_ids, ratios, time_feats, head in tqdm(data_loader):
        k = k + 1
        if k >= 60:
            break
        torch.cuda.empty_cache()
        batch, step = gps_traj.shape[0], gps_traj.shape[1]
        backward_indexes = torch.Tensor([range(step // 3) for i in range(batch)])
        backward_indexes = backward_indexes.long().transpose(1, 0).to(args.device)
        setup_seed(k * 100)
        x = torch.randn(batch, args.emb_dim, step // 5).to(args.device)
        # x = x.permute(2, 0, 1)
        ims = []
        seq = range(0, n_steps, skip)
        seq_next = [-1] + list(seq[:-1])
        # x = x.transpose(-2,-1)
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(batch) * i).to(x.device)
            next_t = (torch.ones(batch) * j).to(x.device)
            with torch.no_grad():
                pred_noise = unet(x, t, head)
                # print(pred_noise.shape)
                x = p_xt(x, pred_noise, t, next_t, beta.to(args.device), eta)
                if i % 10 == 0:
                    ims.append(x.squeeze(0))
        trajs = ims[-1]
        # trajs, b, c, t
        trajs = rearrange(trajs, 'b c t -> t b c')
        # mask = torch.ones((step // 5, batch))
        trajs, _ = mae_model.decoder(trajs, backward_indexes)
        print(trajs.shape)
        trajs = trajs.transpose(-2, -1)
        trajs = scaler.inverse_transform(trajs)
        torch.cuda.empty_cache()
        print(trajs[0])
        all_trajs.append(trajs.detach().cpu().numpy())

    all_trajs = np.concatenate(all_trajs, axis=0)
    if not args.use_mae:
        np.save(f'DiffTraj_newest_{args.dataset}.npy', np.array(all_trajs))
    else:
        np.save(f'DiffTraj_newest_use_mae_{args.dataset}.npy', np.array(all_trajs))
    return all_trajs


def evaluate_loader_with_mae(data_loader, unet, mae_model, optimizer, n_steps, q_xt_x0, p_x0_xt,
                    ema_helper=None, mode='train', epoch = 0, args = None, scaler=None):
    losses = []
    i = 0
    generator = torch.Generator(device=args.device)
    mae_model.eval()
    for gps_traj, grid_ids, edge_ids, ratios, time_feats, head in tqdm(data_loader):
        i = i + 1
        torch.cuda.empty_cache()
        batch, step = gps_traj.shape[0], gps_traj.shape[1]
        if args.transform:
            x0 = scaler.transform(gps_traj)
        else:
            x0 = gps_traj
        # x0 = x0.transpose(1, 2)
        # generator.manual_seed(args.seed + epoch)
        with torch.no_grad():
            # time_emb = mae_model.time_enc(time_feats)
            # x0 = torch.cat([x0, time_emb], dim=-1)
            if args.use_time:
                x0 = mae_model.get_time_fused_pos_emb(x0, time_feats)
            x0 = x0.transpose(2, 1)
            x0, _ = mae_model.encoder(x0)
            x0 = x0[1:, :, :]
            x0 = rearrange(x0, 't b c -> b c t')
            ## this is the raw_emb
        # print(x0.shape)
        t = torch.randint(low=0, high=n_steps, size=(len(x0) // 2 + 1,),device=args.device)
        t = torch.cat([t, n_steps - t - 1], dim=0)[:len(x0)].to(device)
        xt, noise = q_xt_x0(x0, t)
        torch.cuda.empty_cache()
        # Run xt through the network to get its predictions

        pred_noise = unet(xt.float(), t, head)
        # Compare the predictions with the targets
        loss = F.mse_loss(pred_noise.transpose(-2,-1), noise.float().transpose(-2,-1))
        # recon_x0 = p_x0_xt(xt.float(), t, pred_noise)
        # recon_x0 = scaler.inverse_transform(recon_x0.transpose(-2,-1))
        # final_x0 = scaler.inverse_transform(recon_x0)ASW
        losses.append(loss.item())
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if ema_helper != None:
            ema_helper.update(unet)
        torch.cuda.empty_cache()
    train_loss = np.mean(losses)
    logger.info(
            f'The result of the {mode} at epoch {epoch} is {train_loss}.')
    return train_loss


def main(args, config, logger, exp_dir):
    # Modified to return the noise itself as well
    def q_xt_x0(x0, t):
        # print(alpha_bar.device, t.device,x0.device)
        mean = gather(alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(alpha_bar, t)
        eps = torch.randn_like(x0)
        return mean + (var ** 0.5) * eps, eps  # also returns noise

    def p_x0_xt(xt, t, eps):
        sqrt_alpha_bar = gather(alpha_bar, t) ** 0.5
        sqrt_var = (1 - gather(alpha_bar, t)) ** 0.5
        # 重构 x0
        x0_recon = (xt - sqrt_var * eps) / (sqrt_alpha_bar)

        return x0_recon

    if not args.use_time:
        from road_mae_encoder import MAE_ViT, get_parse_args
        mae_args = get_parse_args()
        mae_model = load_mae_model(mae_args, device=args.device)
    else:
        from road_mae_encoder_with_time import MAE_ViT, get_parse_args
        mae_args = get_parse_args()
        mae_model = load_mae_model(mae_args, device=args.device)
    # Create the model
    if args.dataset == 'Porto':
        if not args.use_small_area:
            train_data, val_data, test_data, scaler_std, scaler, graph, all_unique_ids, all_unique_counts = get_all_data_standard(
                args, f"../data", logger)
        else:
            train_data, val_data, test_data, scaler_std, scaler, graph, all_unique_ids, all_unique_counts = get_full_data(
                args, f"../data", logger)
    elif args.dataset == 'chengdu' or args.dataset == 'xian':
        train_data, val_data, test_data, scaler_std, scaler = get_gaia_data(args, f'../data', logger)
    else:
        raise Exception('Dataloader wrong!')
    if not args.use_std:
        use_scaler = scaler
    else:
        use_scaler = scaler_std

    unet = Guide_UNet(args, config, use_scaler, 0).to(args.device)
    # m_path = f"DiffTraj_Pretrain_std_lr_trained_{args.lr}_{args.emb_dim}.pt"
    # unet.load_state_dict(torch.load(m_path, map_location=device))
    # Training params
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True, collate_fn=seq_collate)
    val_loader = DataLoader(val_data, batch_size=args.bs, shuffle=True, collate_fn=seq_collate)
    test_loader = DataLoader(test_data, batch_size=args.bs, shuffle=True, collate_fn=seq_collate)
    del train_data, val_data, test_data
    # Set up some parameters
    n_steps = config.diffusion.num_diffusion_timesteps

    beta = torch.linspace(config.diffusion.beta_start,
                          config.diffusion.beta_end, n_steps).to(args.device)
    alpha = 1. - beta
    # alpha_bar = torch.cumprod(alpha, dim=0).to(device)
    alpha_bar = torch.cumprod(alpha, dim=0).to(args.device)
    ema_helper = EMAHelper(mu=config.model.ema_rate)
    ema_helper.register(unet)
    # Explore this - might want it lower when training on the full dataset
    # Store losses for later plotting
    # optimizer
    # ema_helper = None
    optim = torch.optim.AdamW(unet.parameters(), lr=args.lr)  # Optimizer
    torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[150],
                                         gamma=0.1)

    # new filefold for save model pt
    model_save = exp_dir / 'models' / (timestamp + '/')
    if not os.path.exists(model_save):
        os.makedirs(model_save)

    # config.training.n_epochs = 1

    best_loss = 10000

    for epoch in range(1, args.training_epoch + 1):
        logger.info("<----Epoch-{}---->".format(epoch))
        # if epoch >= 100:
        #     optim.lr = 2e-4
        unet.train()
        if not args.use_mae:
            train_loss = evaluate_loader(train_loader, unet, mae_model, optim, n_steps, q_xt_x0, p_x0_xt, ema_helper, mode='train', epoch=epoch, args=args, scaler=use_scaler)
        else:
            train_loss = evaluate_loader_with_mae(train_loader, unet, mae_model, optim, n_steps, q_xt_x0, p_x0_xt, ema_helper, mode='train', epoch=epoch, args=args, scaler=use_scaler)
        unet.eval()
        with torch.no_grad():
            if not args.use_mae:
                val_loss = evaluate_loader(val_loader, unet, mae_model, optim, n_steps, q_xt_x0, p_x0_xt, ema_helper, mode='val', epoch=epoch, args=args, scaler=use_scaler)
                test_loss = evaluate_loader(test_loader, unet, mae_model, optim, n_steps, q_xt_x0, p_x0_xt, ema_helper,mode='test', epoch=epoch, args=args, scaler=use_scaler)
            else:
                val_loss = evaluate_loader_with_mae(val_loader, unet, mae_model, optim, n_steps, q_xt_x0, p_x0_xt, ema_helper,
                                           mode='val', epoch=epoch, args=args, scaler=use_scaler)
                test_loss = evaluate_loader_with_mae(test_loader, unet, mae_model, optim, n_steps, q_xt_x0, p_x0_xt, ema_helper,
                                            mode='test', epoch=epoch, args=args, scaler=use_scaler)
            torch.cuda.empty_cache()

        if test_loss < best_loss:
            logger.info('Time for saving the model.')
            best_loss = test_loss
            if args.use_mae:
                m_path = f"Timed_DiffTraj_Pretrain_std_lr_trained_{args.lr}_{args.emb_dim}_{args.dataset}_sample_{args.data_sample}_use_mae_patch1.pt"
            else:
                print("存储的名字没有use_mae")
                m_path = f'Timed_DiffTraj_std_lr_trained_{args.lr}_{args.emb_dim}_{args.dataset}_sample_{args.data_sample}_patch1.pt'
            torch.save(unet.state_dict(), m_path)
            unet.epoch = epoch
            # m_path = f"{files_save} / loss_best.npy"
            # np.save(m_path, np.array(losses))

            print("进入生成界面")

def main_generate(args, config, logger):
    # Modified to return the noise itself as well
    # Create the model
    if args.dataset == 'Porto':
        if not args.use_small_area:
            train_data, val_data, test_data, scaler_std, scaler, graph, all_unique_ids, all_unique_counts = get_all_data_standard(
                args, f"../data", logger)
        else:
            train_data, val_data, test_data, scaler_std, scaler_attr, graph, all_unique_ids, all_unique_counts = get_full_data(
                args, f"../data", logger)
    elif args.dataset == 'chengdu' or args.dataset == 'xian':
        train_data, val_data, test_data, scaler_std, scaler_attr = get_gaia_data(args, f'../data', logger)
    else:
        raise Exception('Dataloader Wrong!')

    mae_args = get_parse_args()
    if args.use_mae:
        mae_model = load_mae_model(mae_args, device=args.device)
    else:
        mae_model = None
    use_scaler = scaler_std
    unet = Guide_UNet(args, config, use_scaler, 0).to(args.device)
    if args.use_mae:
        m_path = f"DiffTraj_Pretrain_std_lr_trained_{args.lr}_{args.emb_dim}_{args.dataset}_use_cond_sample_{args.data_sample}_use_mae.pt"
    else:
        print("存储的名字没有use_mae")
        m_path = f'DiffTraj_std_lr_trained_{args.lr}_{args.emb_dim}_{args.dataset}_use_cond_sample_{args.data_sample}.pt'
    unet.load_state_dict(torch.load(m_path, map_location=device))
    test_loader = DataLoader(test_data, batch_size=args.bs, shuffle=True, collate_fn=seq_collate)
    del train_data, val_data, test_data
    # Set up some parameters
    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start,
                          config.diffusion.beta_end, n_steps).to(args.device)
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    timesteps = 100
    skip = n_steps // timesteps
    # alpha_bar = torch.cumprod(alpha, dim=0).to(device)
    alpha_bar = torch.cumprod(alpha, dim=0).to(args.device)

    # new filefold for save model pt
    # data_loader, unet, n_steps, beta, eta,
    # args = None, scaler = None
    eta = 0.0
    if not args.use_mae:
        generate_loader(test_loader, unet, n_steps, beta, eta, args.dataset, args, use_scaler)
    else:
        generate_data_with_mae(test_loader, unet, n_steps, beta, eta, args.dataset, mae_model, args, use_scaler)



def setup_seed(seed, torch_only_deterministic=True):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True  # the picked algorithm of cuda is deterministic
    torch.autograd.set_detect_anomaly(True)
    if torch_only_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


if __name__ == "__main__":
    # Load configuration
    parser = argparse.ArgumentParser('parameter_selection')
    parser.add_argument('--training_epoch', type=int, default=200, help='training epoch')
    parser.add_argument('--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu')
    # parser.add_argument('--another_device', type=str, default='cuda:3' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_name', type=str, default='Transformer', help='the model name')
    # parser.add_argument('--device', type=str, default='cuda:3' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_temb', action='store_true', help='whether use temb')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning_rate')
    parser.add_argument('--emb_dim', type=int, default=64, help='hid_emb_dim')
    parser.add_argument('--layer', type=int, default=1, help='layer_num')
    parser.add_argument('--bs', type=int, default=512, help='batch_size')
    parser.add_argument('--use_gps', action='store_true', help='whether to use gps_traj')
    parser.add_argument('--not_proj', action='store_true', help='whether to proj the raw emb')
    parser.add_argument('--step', type=int, default=120, help='step_lens')
    parser.add_argument('--transform', action='store_true', help='whether to scale')
    parser.add_argument('--use_x0_loss', action='store_true', help='whether to use x0')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--norm_type', type=str, default='Batch', help='normalization_type')
    parser.add_argument('--mlp_layer', type=int, default=1, help='get the layer for MLP')
    parser.add_argument('--kernel_size', type=int, default=3, help='conv kernel size')
    parser.add_argument('--pretrain', action='store_true', help='whether to pretrain')
    parser.add_argument('--use_len', action='store_true', help='use the length feature')

    parser.add_argument('--use_road_emb', action='store_true', help='get_gps_emb')
    parser.add_argument('--only_noise', action='store_true', help='whether to use only the noise loss')
    parser.add_argument('--use_ratio', action='store_true', help='whether to use ratio emb for full road')
    parser.add_argument('--device_ids', type=int, nargs='+', default=[1,1],
                        help='GPU device ids to use')
    parser.add_argument('--use_std', action='store_true', help='whether to use standard scaler')
    parser.add_argument('--use_small_area', action='store_true', help='use smaller area')

    parser.add_argument('--dataset', default='xian', type=str)
    parser.add_argument('--use_mae', action='store_true', help='whether to use the mae enc and dec')
    parser.add_argument('--data_sample', action='store_true', help='sample_data')
    parser.add_argument('--generate_mae', action='store_true', help='whether_to_generate_using_mae')
    parser.add_argument('--use_cond', action='store_true', help='whether_to_use_condition`')
    parser.add_argument('--mask_ratio',type=float, default=0.75, help='masking dat')
    parser.add_argument('--use_time', action='store_true', help='use_time_feats')

    temp = {}
    for k, v in raw_args.items():
        temp[k] = SimpleNamespace(**v)
    config = SimpleNamespace(**temp)

    args = parser.parse_args()
    setup_seed(args.seed)
    root_dir = Path(__name__).resolve().parents[0]
    result_name = '{}_steps={}_len={}_{}_bs={}'.format(
        config.data.dataset, config.diffusion.num_diffusion_timesteps,
        config.data.traj_length, config.diffusion.beta_end,
        config.training.batch_size)
    exp_dir = root_dir / "DiffTraj" / result_name
    for d in ["results", "models", "logs", "Files"]:
        os.makedirs(exp_dir / d, exist_ok=True)
    print("All files saved path ---->>", exp_dir)
    timestamp = datetime.datetime.now().strftime(f"MAE-use_small_{args.use_small_area}-gps-use-cond-{args.use_cond}-%m-%d-%H-%M")
    files_save = exp_dir / 'Files' / (timestamp + '/')
    if not os.path.exists(files_save):
        os.makedirs(files_save)
    # shutil.copy('./utils/config.py', files_save)
    # shutil.copy('./utils/Traj_UNet.py', files_save)
    # device = 'cpu'
    logger = Logger(
        __name__,
        log_path=exp_dir / "logs" / (timestamp + f'_{args.dataset}.log'),
        colorize=True,
    )
    device = args.device
    # device = 'cpu'
    log_info(config, logger)

    mode = 'train'
    if mode != 'test':
        main(args, config, logger, exp_dir)
    else:
        main_generate(args, config, logger)