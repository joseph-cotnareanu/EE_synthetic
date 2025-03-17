
import torch
import numpy as np
import random
from tqdm import tqdm

from train import train_two_stage_experiment
seed = 42  # or any number you choose

# Python random seed
random.seed(seed)

# NumPy random seed
np.random.seed(seed)

# PyTorch random seed
torch.manual_seed(seed)

# If you are using CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
from create_model import create_two_stage_model
from generate_data import load_data


if __name__ == '__main__':
    costs = [0]
    test_n = 32*100
    train_n = 32*100
    mc_posterior_n = 32*10
    num_trails = 1
    
    for trial in range(num_trails):
        data_dict = load_data(trial = trial, train_n=train_n, test_n=test_n, mc_posterior_n=mc_posterior_n)
        for cost in costs:
            two_stage_model = create_two_stage_model(x_dim=1, z_dim=1, num_classes=2)
            train_two_stage_experiment(data_dict, cost, two_stage_model)