import math
import sys
sys.path.append('./')
import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
import torch.nn.functional as F
from utils.dataloader import *
import dgl
from time_encoder import get_time_encoder
from TEVI.road_net_for_edge_as_node_v2 import load_graph, load_dgl_graph_v4


def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Attention(nn.Module):
    def __init__(self, embedding_dim):
        super(Attention, self).__init__()
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, num_attributes, embedding_dim)
        weights = self.fc(x)  # shape: (batch_size, num_attributes, 1)
        # apply softmax along the attributes dimension
        weights = F.softmax(weights, dim=1)
        return weights


class WideAndDeep(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=256):
        super(WideAndDeep, self).__init__()

        # Wide part (linear model for continuous attributes)
        self.wide_fc = nn.Linear(5, embedding_dim)

        # Deep part (neural network for categorical attributes)
        self.depature_embedding = nn.Embedding(288, hidden_dim)
        self.sid_embedding = nn.Embedding(257, hidden_dim)
        self.eid_embedding = nn.Embedding(257, hidden_dim)
        self.deep_fc1 = nn.Linear(hidden_dim, embedding_dim)
        self.deep_fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, attr):
        # Continuous attributes
        attr = attr.squeeze()
        continuous_attrs = attr[:, 1:6]

        # Categorical attributes
        depature = attr[:, 0].long()

        # Wide part
        wide_out = self.wide_fc(continuous_attrs)

        # Deep part
        depature_embed = self.depature_embedding(depature)
        # sid_embed = self.sid_embedding(sid)
        # eid_embed = self.eid_embedding(eid)
        # categorical_embed = torch.cat(
        #     (depature_embed, sid_embed, eid_embed), dim=1)
        deep_out = F.relu(self.deep_fc1(depature_embed))
        deep_out = self.deep_fc2(deep_out)
        # Combine wide and deep embeddings
        combined_embed = wide_out + deep_out

        return combined_embed


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32,
                              num_channels=in_channels,
                              eps=1e-6,
                              affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x,
                                            scale_factor=2.0,
                                            mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (1, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 conv_shortcut=False,
                 dropout=0.1,
                 temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        # print("查看in_channels:", self.in_channels)
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, w = q.shape
        q = q.permute(0, 2, 1)  # b,hw,c
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        # attend to values
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, w)

        h_ = self.proj_out(h_)

        return x + h_



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer, kernel_size=3, mode = 'enc'):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer = layer
        self.kernel_size = kernel_size

        self.MLP = nn.ModuleList()
        assert layer >= 1
        for i in range(layer):
            if i == 0:
                if mode == 'dec':
                    if layer == 1:
                        self.MLP.append(nn.Conv1d(self.input_dim, self.hidden_dim, self.kernel_size, 1, (self.kernel_size - 1) // 2))
                    else:
                        self.MLP.append(nn.Conv1d(self.input_dim, self.input_dim, self.kernel_size, 1, (self.kernel_size - 1) // 2))
                elif mode == 'enc':
                    self.MLP.append(nn.Conv1d(self.input_dim, self.hidden_dim, self.kernel_size, 1, (self.kernel_size - 1) // 2))
            else:
                if mode == 'dec':
                    if i == layer - 1:
                        self.MLP.append(nn.Conv1d(self.input_dim, self.hidden_dim, self.kernel_size, 1, (self.kernel_size - 1) // 2))
                    else:
                        self.MLP.append(nn.Conv1d(self.input_dim, self.input_dim, self.kernel_size, 1, (self.kernel_size - 1) // 2))
                elif mode == 'enc':
                    self.MLP.append(nn.Conv1d(self.hidden_dim, self.hidden_dim, self.kernel_size, 1, (self.kernel_size - 1) // 2))
            if i != layer - 1:
                self.MLP.append(nn.SiLU())

    def forward(self,x):
        for i, layer_form in enumerate(self.MLP):
            x = layer_form(x)
        return x



class Model(nn.Module):
    def __init__(self, args, config, scaler):
        super().__init__()
        self.config = config
        self.args = args
        self.scaler = scaler
        ch, out_ch, ch_mult = self.args.emb_dim, config.model.out_ch, tuple(
            config.model.ch_mult)

        if self.args.use_mae:
            out_ch = self.args.emb_dim

        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        if not self.args.use_mae:
            in_channels = config.model.in_channels
        else:
            in_channels = self.args.emb_dim
        resolution = config.data.traj_length
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps

        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))

        self.ch = self.args.emb_dim
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        if self.args.use_mae:
            self.in_channels = self.args.emb_dim
        else:
            self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv1d(self.in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1, ) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(in_channels=block_in + skip_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t, extra_embed=None):
        # print("查看x的形状:",x.shape)
        # assert x.shape[2] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        if extra_embed is not None:
            temb = temb + extra_embed

        # downsampling
        hs = [self.conv_in(x)]
        # print(hs[-1].shape)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # print(i_level, i_block, h.shape)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        # print(hs[-1].shape)
        # print(len(hs))
        h = hs[-1]  # [10, 256, 4, 4]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # print(h.shape)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                ht = hs.pop()
                if ht.size(-1) != h.size(-1):
                    h = torch.nn.functional.pad(h,
                                                (0, ht.size(-1) - h.size(-1)))
                h = self.up[i_level].block[i_block](torch.cat([h, ht], dim=1),
                                                    temb)
                # print(i_level, i_block, h.shape)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Guide_UNet(nn.Module):
    def __init__(self,args, config, use_scaler, epoch):
        super(Guide_UNet, self).__init__()
        self.epoch = epoch
        self.config = config
        self.args = args
        self.ch = self.args.emb_dim * 4
        self.attr_dim = config.model.attr_dim
        self.guidance_scale = config.model.guidance_scale
        self.unet = Model(args, config, use_scaler)
        # self.guide_emb = Guide_Embedding(self.attr_dim, self.ch)
        # self.place_emb = Place_Embedding(self.attr_dim, self.ch)
        self.guide_emb = WideAndDeep(self.ch)
        self.place_emb = WideAndDeep(self.ch)

    def forward(self, x, t, attr):
        guide_emb = self.guide_emb(attr)
        place_vector = torch.zeros(attr.shape, device=attr.device)
        place_emb = self.place_emb(place_vector)
        cond_noise = self.unet(x, t, guide_emb)
        uncond_noise = self.unet(x, t, place_emb)
        if self.args.use_cond:
            pred_noise = cond_noise + self.guidance_scale * (cond_noise -
                                                         uncond_noise)
        else:
            pred_noise = uncond_noise
        # pred_noise = uncond_noise
        return pred_noise

if __name__ == '__main__':
    from utils.config import args

    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)

    config = SimpleNamespace(**temp)
    t = torch.randn(10)
    depature = torch.zeros(10)
    avg_dis = torch.zeros(10)
    avg_speed = torch.zeros(10)
    total_dis = torch.zeros(10)
    total_time = torch.zeros(10)
    total_len = torch.zeros(10)
    sid = torch.zeros(10)
    eid = torch.zeros(10)
    attr = torch.stack(
        [depature, total_dis, total_time, total_len, avg_dis, avg_speed, sid, eid], dim=1)
    unet = Guide_UNet(args,config)
    x = torch.randn(10, 2, 200)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f'{total_params:,} total parameters.')
    out = unet(x, t, attr)
    print(out.shape)
