import numpy as np
import pandas as pd
import torch.nn as nn
import torch

trajs = torch.load('models/road_encoder_CD.pt')
print(trajs.shape)