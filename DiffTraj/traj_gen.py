import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm
from types import SimpleNamespace
from utils.Traj_UNet import *
from utils.config import raw_args
from utils.utils import *
from torch.utils.data import DataLoader

head = np.load('heads.npy',
                   allow_pickle=True)
head = torch.from_numpy(head).float()
print(head[0])
