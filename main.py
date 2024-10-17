
import torch
import numpy as np
import random

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
    
from generate_data import load_data


if __name__ == '__main__':
    costs = [0, 0.002, 0.004, 0.006, 0.008, 0.01,0.05, 0.1]
    test_n = 32*10000
    train_n = 32*10000
    mc_posterior_n = 32*10000
    num_trails = 2
    for trial in range(num_trails):
        data_dict = load_data(trial = trial, train_n=train_n, test_n=test_n, mc_posterior_n=mc_posterior_n)