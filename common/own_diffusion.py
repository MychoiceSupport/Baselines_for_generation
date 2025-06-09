import numpy as np
import pandas as pd
import torch
from new_baselines import config

diffusion_step = config.diffusion.num_diffusion_timesteps
betas = torch.linspace(config.diffusion.beta_start,
                          config.diffusion.beta_end, diffusion_step)
alphas = 1. - betas
alphas_bar = torch.cumprod(alphas, dim=0).requires_grad_(False)


def q_xt_x0_v1(x0, t):
    noise = torch.randn_like(x0)



