import torch.nn as nn

from tqdm import tqdm
from einops import rearrange
from utils.logger import Logger
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
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
import torch.nn as nn
import math
sys.path.append('./')
sys.path.append('././')
sys.path.append('/home/huanghy/python_directories/Trajectory/TEVI')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def mask_it(x, masks):
    """
    :param x:
    :param masks: N * T
    :return:
    """
    b, l, f = x.shape
    x_visible = x[~masks, :].reshape(b, -1, f)  # (bs, vis_size, z_dim)
    return x_visible

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


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        """
        embed_dim: 输出维度（必须是偶数）
        max_len: 支持的最大序列长度
        """
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.embed_dim = embed_dim
        self.max_len = max_len

        # 创建位置嵌入缓冲区（不可训练）
        self.register_buffer('pos_embed', self.create_pos_embed(max_len, embed_dim))

    @staticmethod
    def create_pos_embed(max_len, embed_dim):
        """创建位置嵌入矩阵"""
        # 生成位置索引 [0, 1, 2, ..., max_len-1]
        pos = np.arange(max_len)

        # 计算正弦/余弦位置嵌入
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000 ** omega

        # 计算外积: pos * omega
        out = np.einsum('m,d->md', pos, omega)

        # 计算正弦和余弦部分
        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        # 拼接正弦和余弦部分
        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return torch.from_numpy(emb).float()

    def forward(self, x):
        """
        x: 输入张量，形状为 (batch_size, seq_len, embed_dim) 或 (seq_len, batch_size, embed_dim)
        返回: 添加了位置编码的张量
        """
        seq_len = x.size(1) if x.dim() == 3 and x.size(0) != self.max_len else x.size(0)
        if seq_len > self.max_len:
            raise ValueError(f"序列长度 {seq_len} 超过了最大支持长度 {self.max_len}")

        # 获取位置嵌入
        pos_embed = self.pos_embed[:seq_len]

        # 根据输入形状调整位置嵌入
        if x.dim() == 3:
            if x.size(1) == seq_len:  # (batch, seq, dim)
                pos_embed = pos_embed.unsqueeze(0)  # (1, seq, dim)
            else:  # (seq, batch, dim)
                pos_embed = pos_embed.unsqueeze(1)  # (seq, 1, dim)
        return x + pos_embed.to(x.device)

    def get_pos_embed(self, seq_len):
        """获取指定长度的位置嵌入"""
        if seq_len > self.max_len:
            raise ValueError(f"序列长度 {seq_len} 超过了最大支持长度 {self.max_len}")
        return self.pos_embed[:seq_len]


class MAE_Encoder(torch.nn.Module):
    def __init__(self,input_dim=2, hidden_dim=128, layer=3, nhead=4):
        super(MAE_Encoder, self).__init__()
        self.in_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer = layer
        self.nhead = nhead

        self.pos_emb = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(batch_first=True, d_model=self.hidden_dim, nhead=self.nhead, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.layer)
        self.positional_encoding = PositionalEmbedding(self.hidden_dim, max_len=500)
        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x):
        """
        :param x: N * T * 2
        :return:
        """
        x = self.pos_emb(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)

        return self.fc(x)

def generate_random_masks(mask_ratio, patch_size, batch_size, seed=0):
    """
    生成随机patch mask的张量

    参数:
    mask_ratio: 需要被mask的patch比例 (0-1)
    patch_size: 每个patch包含的时间步数
    batch_size: 批量大小
    seed: 随机种子 (确保可复现性)

    返回:
    masks: 形状为(B, T)的布尔张量, True表示被mask的位置
    """
    # 设置随机种子确保可复现性
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 时间序列总长度（假设为120）
    ts_size = 120
    # 计算patch数量
    num_patches = ts_size // patch_size
    # 计算需要mask的patch数量
    num_masks = int(num_patches * mask_ratio)
    # 初始化全0 mask张量 (B, T)
    masks = torch.zeros((batch_size, ts_size), dtype=torch.bool)
    # 为每个batch生成独立的mask
    for i in range(batch_size):
        # 随机选择要mask的patch索引
        mask_indices = torch.randperm(num_patches)[:num_masks]

        # 将选中的patch位置设为1 (True)
        for idx in mask_indices:
            start = idx * patch_size
            end = (idx + 1) * patch_size
            masks[i, start:end] = True

    return masks

class Interpolator(nn.Module):
    def __init__(self, mask_ratio, hidden_dim):
        super(Interpolator, self).__init__()
        self.ts_size = 120
        self.mask_ratio = mask_ratio
        self.total_mask_size = int(self.ts_size * self.mask_ratio)
        self.hidden_dim = hidden_dim

        self.sequence_inter = nn.Linear(in_features=(self.ts_size - self.total_mask_size),
                                        out_features=self.ts_size)
        self.feature_inter = nn.Linear(in_features=self.hidden_dim,
                                       out_features=self.hidden_dim)

    def forward(self, x):
        if self.sequence_inter.in_features != self.sequence_inter.out_features:
            x = rearrange(x, 'b l f -> b f l')
            x = self.sequence_inter(x)
            x = rearrange(x, 'b f l -> b l f')
        x = self.feature_inter(x)
        return x

class MAE_Decoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, layer = 3, nhead=4):
        super(MAE_Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer = layer
        self.nhead = nhead

        self.pos_encoder = PositionalEmbedding(self.hidden_dim, max_len=500)
        self.decoder_layer = nn.TransformerDecoderLayer(batch_first=True, nhead=self.nhead, d_model=self.hidden_dim, dropout=0.1)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=self.layer
        )

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim)
        )
        self.query = nn.Parameter(torch.zeros(1, 120, self.hidden_dim))
        nn.init.normal_(self.query, mean=0, std=0.02)
        pass

    def forward(self, x_enc):
        x_enc = self.pos_encoder(x_enc)

        # 创建查询向量 (复制到batch size)
        query = self.query.repeat(x_enc.size(0), 1, 1)

        # 添加位置编码到查询向量
        query = self.pos_encoder(query)

        # Transformer解码
        # 使用可学习查询向量从编码器输出中解码
        x = self.decoder(
            tgt=query,  # 可学习查询向量
            memory=x_enc  # 编码器输出
        )  # (batch, ts_size, hidden_dim)

        # 映射到原始空间
        x_dec = self.fc(x)
        return x_dec


class MAE_ViT(nn.Module):
    def __init__(self, args):
        super(MAE_ViT, self).__init__()
        self.input_dim = 2
        self.hidden_dim = args.emb_dim
        self.encoder_head = args.encoder_head
        self.decoder_head = args.decoder_head
        self.encoder_layer = args.encoder_layer
        self.decoder_layer = args.decoder_layer
        self.encoder_head = args.encoder_head
        self.decoder_head = args.decoder_head
        self.mask_ratio = args.mask_ratio

        self.masks = generate_random_masks(args.mask_ratio, args.patch_size, args.bs, args.seed)
        self.encoder = MAE_Encoder(self.input_dim, self.hidden_dim, self.encoder_layer, self.encoder_head)
        self.decoder = MAE_Decoder(self.input_dim, self.hidden_dim, self.decoder_layer, self.decoder_head)
        self.interpolator = Interpolator(self.mask_ratio, self.hidden_dim)

    def forward(self, x, masks):
        x_vis = mask_it(x, masks)  # (bs, vis_size, z_dim)
        # 编码可见部分
        x_enc = self.encoder(x_vis)  # (bs, vis_size, hidden_dim)
        # 插值到完整序列长度
        x_inter = self.interpolator(x_enc)  # (bs, ts_size, hidden_dim)
        # 解码重建完整序列
        x_dec = self.decoder(x_inter)  # (bs, ts_size, z_dim)
        return x_dec, masks

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


def main(args, logger):
    model = MAE_ViT(args=args).to(args.device)

    # if os.path.exists(f'GPS_Pretrain_false_std_{args.dataset}_{args.use_time}_{args.emb_dim}_{args.mask_ratio}.pt'):
    #     checkpoint = torch.load(f'GPS_Pretrain_false_std_{args.dataset}_{args.use_time}_{args.emb_dim}_{args.mask_ratio}.pt')
    #     model.load_state_dict(checkpoint)
    #     start_epoch = 142
    # else:
    #     start_epoch = 1
    start_epoch = 0
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
                torch.save(model.state_dict(), f'GPS_new_Pretrain_std_without_time_{args.dataset}_{args.use_time}_{args.emb_dim}_{args.mask_ratio}_patch_size_{args.patch_size}.pt')
            else:
                torch.save(model.state_dict(), f'GPS_new_Pretrain_false_std_without_time_{args.dataset}_{args.use_time}_{args.emb_dim}_{args.mask_ratio}_patch_size_{args.patch_size}.pt')


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

def evaluate_loader(args, logger, data_loader, model, optimizer, epoch, max_epoch, scaler, mode='train'):
    loss_fn = nn.MSELoss()
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}/{max_epoch}")
    all_loss = []
    all_masked_geo_loss = []
    all_unmasked_geo_loss = []
    masks = generate_random_masks(args.mask_ratio, args.patch_size, args.bs, args.seed).to(args.device)
    for gps_traj, grid_ids, edge_ids, ratios, time_feats, head in pbar:
        # print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        # print(gps_traj[0,0])
        batch = gps_traj.shape[0]
        masks = masks[:batch]
        raw_gps_traj = gps_traj
        if args.transform:
            gps_traj = scaler.transform(gps_traj)
        else:
            pass
        # print(raw_gps_traj[0, 0], gps_traj[0, 0])
        rec_traj, mask = model(gps_traj, masks)
        # mask = mask.transpose(-2, -1)
        if epoch == 0:
            assert mask[0,:].sum() == 120 * args.mask_ratio
        loss_unmasked = 1e3 * loss_fn(rec_traj * mask.unsqueeze(-1), gps_traj * mask.unsqueeze(-1)) / args.mask_ratio
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser('parameter_selection')
    parser.add_argument('--training_epoch', type=int, default=200, help='training epoch')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device_ids', default=[0,0], help='all_device_ids')
    parser.add_argument('--lr', type=float, default=1e-3, help='input length')
    parser.add_argument('--bs', type=int, default=1024, help='batch size')

    parser.add_argument("--image_size", type=int, default=120, help="输入序列长度（时间步数）")
    parser.add_argument("--patch_size", type=int, default=3,help="每个patch覆盖的时间步数（必须能整除image_size）")
    parser.add_argument("--emb_dim", type=int, default=128, help="嵌入维度（特征维度）")
    parser.add_argument("--encoder_layer", type=int, default=3,help="Encoder的Transformer层数")
    parser.add_argument("--encoder_head", type=int, default=2,help="Encoder的注意力头数")
    parser.add_argument("--decoder_layer", type=int, default=3,help="Decoder的Transformer层数")
    parser.add_argument("--decoder_head", type=int, default=2,help="Decoder的注意力头数")
    parser.add_argument("--mask_ratio", type=float, default=0.75,help="掩码比例（0-1之间）")
    parser.add_argument('--dataset', type=str, default='xian', help='数据读取')

    parser.add_argument('--transform', action='store_true', help='get the transformation')
    parser.add_argument('--use_std', action='store_true', help='get_std_output')
    parser.add_argument('--data_sample', action='store_true', help='sample_data')
    parser.add_argument('--use_time', action='store_true', help='whether to use time feats for rec')
    parser.add_argument('--seed', type=int, default=42, help='the seed for training')

    args = parser.parse_args()
    result_name = 'pretrain_gps_another'
    exp_dir = result_name
    setup_seed(args.seed)

    root_dir = Path(__name__).resolve().parents[0]
    logger = Logger(
        __name__,
        log_path= root_dir / exp_dir / f"Pretrain_use_std_{args.dataset}_use_MAE_test_new_patch_1.log",
        colorize=True,
    )


    device = args.device
    get_info(logger, args)
    main(args, logger)