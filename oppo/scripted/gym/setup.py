from sys import path as spath
from os import path as opath
spath.append(opath.dirname(opath.abspath(__file__)) + "/../..")

import numpy as np
import torch
import d4rl
import random

def seed(seed: int = 0):
  RANDOM_SEED = seed
  np.random.seed(RANDOM_SEED)
  torch.manual_seed(RANDOM_SEED)
  torch.cuda.manual_seed_all(RANDOM_SEED)
  random.seed(RANDOM_SEED)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")