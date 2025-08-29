import os, json, random
import numpy as np
import torch
from pathlib import Path

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
