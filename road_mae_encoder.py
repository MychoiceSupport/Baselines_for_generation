'''
code derived from https://github.com/IcarusWizard/MAE
'''
import torch
import timm
import numpy as np
from utils.logger import Logger
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
import torch.nn as nn
import dgl
from tqdm import tqdm
from torch.utils.data import DataLoader
try:
    from time_encoder import get_time_encoder
except:
    from utils.time_encoder import get_time_encoder
import argparse
import os
import datetime
from pathlib import Path
from utils.utils import *
from utils.dataloader import get_all_data_standard, get_all_data, get_full_data, get_gaia_data
import utils
import sys
sys.path.append('./')
sys.path.append('././')
sys.path.append('/home/huanghy/python_directories/Trajectory/TEVI')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(
            patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(
            patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes


class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=120,
                 patch_size=5,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 input_dim=192):
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size), 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv1d(input_dim, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.init_weight()


    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c l-> l b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)
        # need_index = backward_indexes[:, 0]
        # for data in need_index:
        #     print(data)
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes


class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=120,
                 patch_size=5,
                 emb_dim=192,
                 num_layer=4,
                 num_head=2,
                 input_dim=192
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, input_dim * patch_size)
        self.patch2img = Rearrange('h b (c p) -> b c (h p)', p=patch_size, h=image_size // patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes=None):
        T = features.shape[0]
        # if backward_indexes != None
        backward_indexes = torch.cat(
            [torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat(
            [features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)],
            dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:]  # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T - 1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: torch.Tensor):
        """
        计算数据的均值和标准差
        """
        self.mean = torch.mean(data.reshape(-1,2), dim=0)
        self.std = torch.std(data.reshape(-1,2), dim=0)

    def transform(self, data: torch.Tensor):
        """
        对数据进行标准化
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet. Call `fit` first.")
        raw_data = data.clone()
        raw_data[..., 0] = (data[..., 0] - self.mean[0]) / (self.std[0])
        raw_data[..., 1] = (data[..., 1] - self.mean[1]) / (self.std[1])
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
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet. Call `fit` first.")
        raw_data = data.clone()
        raw_data[..., 0] = data[..., 0] * self.std[0] + self.mean[0]
        raw_data[..., 1] = data[..., 1] * self.std[1] + self.mean[1]
        return raw_data


class MAE_ViT(torch.nn.Module):
    def __init__(self, args,
                 image_size=120,
                 patch_size=2,
                 emb_dim=128,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 input_dim = 2
                 ):
        super().__init__()

        self.args = args
        self.emb_dim = emb_dim
        self.encoder_layer = encoder_layer
        self.encoder_head = encoder_head
        self.decoder_layer = decoder_layer
        self.decoder_head = decoder_head
        self.mask_ratio = mask_ratio
        self.device0 = f'cuda:{args.device_ids[0]}'
        self.device1 = f'cuda:{args.device_ids[1]}'

        if self.args.use_time:
            self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio, input_dim + self.emb_dim)
        else:
            self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio,
                                       input_dim)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head, input_dim)
        # self.time_enc = get_time_encoder('day', self.emb_dim, args.device)
        self.time_enc = nn.Embedding(288, self.emb_dim)
        self.time_lin = nn.Linear(input_dim + self.emb_dim, self.emb_dim)
        self.pos_emb = nn.Sequential(
            nn.Linear(2, self.emb_dim), nn.ReLU(), nn.Linear(self.emb_dim, self.emb_dim)
        )
        self.pos_dec = nn.Sequential(
            nn.Linear(self.emb_dim, 2)
        )


    def forward(self, img, time_feats):
        """
        :param img: N * T * 2 -> N * T * dim
        :param time_feats: N * T
        :return:
        """
        if self.args.use_time:
            time_feats = self.time_enc(time_feats)
        # img = self.pos_emb(img)
            img = torch.cat([img, time_feats], dim=-1)
        # img = self.time_lin(img)
        img = img.transpose(-2, -1)
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img.transpose(-2,-1), mask


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


def evaluate_loader(args, logger, data_loader, model, optimizer, epoch, max_epoch, scaler, mode='train'):
    loss_fn = nn.MSELoss()
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}/{max_epoch}")
    all_loss = []
    all_masked_geo_loss = []
    all_unmasked_geo_loss = []
    for gps_traj, grid_ids, edge_ids, ratios, time_feats, head in pbar:
        # print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        # print(gps_traj[0,0])
        raw_gps_traj = gps_traj
        if args.transform:
            gps_traj = scaler.transform(gps_traj)
        else:
            pass
        # print(raw_gps_traj[0, 0], gps_traj[0, 0])
        rec_traj, mask = model(gps_traj, time_feats)
        mask = mask.transpose(-2, -1)
        if epoch == 0:
            assert mask[0,:,0].sum() == 120 * args.mask_ratio
        loss_unmasked = 1e3 * loss_fn(rec_traj * mask[..., 0].unsqueeze(-1), gps_traj * mask[..., 0].unsqueeze(-1)) / args.mask_ratio
        loss_real = 1e3 * loss_fn(rec_traj, gps_traj)
        loss = loss_unmasked + loss_real
        all_loss.append(loss.item())
        if args.transform:
            rev_traj = scaler.inverse_transform(rec_traj)
            # geo_loss = torch.mean(geodesic(rev_traj[torch.where(mask[..., 0] == 1)], raw_gps_traj[torch.where(mask[..., 0] == 1)]))
        else:
            rev_traj = rec_traj
        geo_loss_masked = torch.mean(geodesic(rev_traj[torch.where(mask[..., 0] == 1)], raw_gps_traj[torch.where(mask[..., 0] == 1)]))
        geo_loss_unmasked = torch.mean(
            geodesic(rev_traj[torch.where(mask[..., 0] == 0)], raw_gps_traj[torch.where(mask[..., 0] == 0)]))
        all_masked_geo_loss.append(geo_loss_masked.item())
        all_unmasked_geo_loss.append(geo_loss_unmasked.item())
        # print(raw_gps_traj[0,0], rec_traj[0,0], rev_traj[0,0])
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # torch.cuda.empty_cache()
    train_loss = np.mean(all_loss)
    train_geo_loss_masked = np.mean(all_masked_geo_loss)
    train_geo_loss_unmasked = np.mean(all_unmasked_geo_loss)
    logger.info(f'The result for the {mode} dataset is {train_loss} at epoch {epoch}, the recovered trajectory pos error for masked part is {train_geo_loss_masked},'
                f'the result for unmasked part is {train_geo_loss_unmasked}.')
    return train_loss


def get_args(args):
    print("\n====== 参数列表 ======")
    for arg in vars(args):
        print(f"{arg:>12}: {getattr(args, arg)}")
    print("===============\n")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args, logger):
    model = MAE_ViT(args=args,
                    image_size=args.image_size,
                    patch_size=args.patch_size,
                    emb_dim=args.emb_dim,
                    encoder_layer=args.encoder_layer,
                    encoder_head=args.encoder_head,
                    decoder_layer=args.decoder_layer,
                    decoder_head=args.decoder_head,
                    mask_ratio=args.mask_ratio,
                    input_dim=2
                    ).to(args.device)

    # if os.path.exists(f'GPS_Pretrain_false_std_{args.dataset}_{args.use_time}_{args.emb_dim}_{args.mask_ratio}.pt'):
    #     checkpoint = torch.load(f'GPS_Pretrain_false_std_{args.dataset}_{args.use_time}_{args.emb_dim}_{args.mask_ratio}.pt')
    #     model.load_state_dict(checkpoint)
    #     start_epoch = 142
    # else:
    #     start_epoch = 1
    start_epoch = 0
    print(f"Total trainable parameters: {get_model_memory_size(model)}")
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150], gamma=0.1)
    best_loss = 1e10
    if args.dataset == 'Porto':
        train_data, val_data, test_data, scaler, scaler_std, graph, all_unique_ids, all_unique_counts= get_full_data(args,f"../data",logger)
    else:
        train_data, val_data, test_data, scaler_std, scaler = get_gaia_data(args, f"../data",logger)
    if args.use_std:
        use_scaler = scaler_std
    else:
        use_scaler = scaler_std
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True, collate_fn=seq_collate,pin_memory=False)
    val_loader = DataLoader(val_data, batch_size=args.bs, shuffle=True, collate_fn=seq_collate,pin_memory=False)
    test_loader = DataLoader(test_data, batch_size=args.bs, shuffle=True, collate_fn=seq_collate,pin_memory=False)
    del train_data, val_data, test_data
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    for epoch in range(start_epoch, args.training_epoch + 1):
        logger.info("<----Epoch-{}---->".format(epoch))
        # if epoch >= 100:
        #     optimizer.lr = 1e-5
        model.train()
        _ = evaluate_loader(args, logger, train_loader, model, optimizer, epoch, args.training_epoch, use_scaler,
                            mode='train')
        model.eval()
        _ = evaluate_loader(args, logger, val_loader, model, optimizer, epoch, args.training_epoch, use_scaler, mode='val')
        test_loss = evaluate_loader(args, logger, test_loader, model, optimizer, epoch, args.training_epoch, use_scaler,
                                    mode='test')
        torch.cuda.empty_cache()
        # try:
        #     torch.cuda.empty_cache()  # 可能会失败
        # except RuntimeError:
        #     torch.cuda.synchronize()  # 等待所有 CUDA 操作完成
        #     torch.cuda.empty_cache()  # 再次尝
        # torch.cuda.synchronize()
        if test_loss < best_loss:
            best_loss = test_loss
            logger.info(f"Saving the model at epoch {epoch}! with transform {args.transform}.")
            if args.transform:
                torch.save(model.state_dict(), f'GPS_Pretrain_std_without_time_{args.dataset}_{args.use_time}_{args.emb_dim}_{args.mask_ratio}_patch_size_{args.patch_size}.pt')
            else:
                torch.save(model.state_dict(), f'GPS_Pretrain_false_std_without_time_{args.dataset}_{args.use_time}_{args.emb_dim}_{args.mask_ratio}_patch_size_{args.patch_size}.pt')


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


def get_model_memory_size(model, dtype_bytes=4):
    """计算模型参数占用的内存（MB）"""
    param_mem = sum(p.numel() * dtype_bytes for p in model.parameters()) / (1024 ** 3)
    print(f"Model parameters memory: {param_mem:.2f} GB (dtype={dtype_bytes} bytes/param)")
    return param_mem

class Args:
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)
def get_parse_args():
    params_dict = {
        'training_epoch': 200,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'device_ids': [3, 3],
        'lr': 1e-3,
        'bs': 128,
        'image_size': 120,
        'patch_size': 3,
        'emb_dim': 128,
        'encoder_layer': 3,
        'encoder_head': 4,
        'decoder_layer': 3,
        'decoder_head': 4,
        'mask_ratio': 0.75,
        'transform': True,
        'use_std': True,
        'dataset': 'xian',
        'use_time': False,
    }
    args = Args(params_dict)
    return args

def get_info(logger, args):
    """使用 logger.info 打印解析后的参数"""
    logger.info("运行参数：")
    args_dict = vars(args)
    for key, value in args_dict.items():
        logger.info(f"{key}: {value}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('parameter_selection')
    parser.add_argument('--training_epoch', type=int, default=200, help='training epoch')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device_ids', default=[0,0], help='all_device_ids')
    parser.add_argument('--lr', type=float, default=1e-3, help='input length')
    parser.add_argument('--bs', type=int, default=2048, help='batch size')

    parser.add_argument("--image_size", type=int, default=120, help="输入序列长度（时间步数）")
    parser.add_argument("--patch_size", type=int, default=3,help="每个patch覆盖的时间步数（必须能整除image_size）")
    parser.add_argument("--emb_dim", type=int, default=128, help="嵌入维度（特征维度）")
    parser.add_argument("--encoder_layer", type=int, default=3,help="Encoder的Transformer层数")
    parser.add_argument("--encoder_head", type=int, default=4,help="Encoder的注意力头数")
    parser.add_argument("--decoder_layer", type=int, default=3,help="Decoder的Transformer层数")
    parser.add_argument("--decoder_head", type=int, default=4,help="Decoder的注意力头数")
    parser.add_argument("--mask_ratio", type=float, default=0.75,help="掩码比例（0-1之间）")
    parser.add_argument('--dataset', type=str, default='xian', help='数据读取')

    parser.add_argument('--transform', action='store_true', help='get the transformation')
    parser.add_argument('--use_std', action='store_true', help='get_std_output')
    parser.add_argument('--data_sample', action='store_true', help='sample_data')
    parser.add_argument('--use_time', action='store_true', help='whether to use time feats for rec')

    args = parser.parse_args()
    result_name = 'pretrain_gps_another'
    exp_dir = result_name
    get_args(args)
    root_dir = Path(__name__).resolve().parents[0]
    logger = Logger(
        __name__,
        log_path= root_dir / exp_dir / f"Pretrain_use_std_{args.dataset}_use_MAE_test_new_patch_1.log",
        colorize=True,
    )

    setup_seed(42)
    device = args.device
    get_info(logger, args)
    main(args, logger)
    # 训练参数

    # image_size = args.emb_dim,
    # patch_size = args.patch_size,
    # emb_dim = args.emb_dim,
    # encoder_layer = args.encoder_layer,
    # encoder_head = args.encoder_head,
    # decoder_layer = args.decoder_layer,
    # decoder_head = args.decoder_head,
    # mask_ratio = args.mask_ratio

