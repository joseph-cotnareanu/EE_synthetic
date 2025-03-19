import torch
import numpy as np
from tqdm import tqdm
import os
import pickle as pk
from torch.nn.functional import sigmoid


def posterior_p_y_given_x(x, n_samples):
    # Sample z from U[-1,1]
    z_samples = np.random.uniform(-10, 10, n_samples)

    # Compute sigmoid(x + z) for each z
    prob_y_given_xz = sigmoid(x + z_samples)

    # Compute p(y=1 | x)
    p_y1_x = torch.mean(prob_y_given_xz)

    # Return max(p(y=1 | x), p(y=0 | x))
    # return max(p_y1_x, 1 - p_y1_x)
    return p_y1_x

# Monte Carlo approximation of expected max_y p(y | x, z)


def expected_max_p_y_given_xz(x, n_samples):
    # Sample z from U[-1,1]
    z_samples = np.random.uniform(-10, 10, n_samples)

    # Compute sigmoid(x + z) for each z
    prob_y_given_xz = sigmoid(x + z_samples)

    # Compute max(p(y=1 | x, z), p(y=0 | x, z)) for each z
    max_prob_y_given_xz = np.maximum(prob_y_given_xz, 1 - prob_y_given_xz)

    # Return the expected value of max_y p(y | x, z)
    return torch.mean(max_prob_y_given_xz)


def posterior_p_y_given_x_z(x, z):
    # Return and p_y1_x_z
    return sigmoid(x + z)


def generate_xzy(n: int):
    r1, r2 = -10, 10
    # sample on the uniform distribution
    x = (r1 - r2) * torch.rand((n, 1)) + r2
    z = (r1 - r2) * torch.rand((n, 1)) + r2
    # y \sim Bernouilli(p)

    # prob
    p_y = sigmoid(x+z)

    flat_y_train = torch.bernoulli(p_y).type(torch.LongTensor).flatten()
    y = torch.nn.functional.one_hot(flat_y_train, num_classes=2)
    return x, z, y


def create_data(trial: int, train_n: int, test_n: int, mc_posterior_n: int):
    
    """
        This generate data and return it, following our  model
        X \\sim Uniform[-1,1]
        Z \\sim Uniform[-1,1]
        Y \\sim Bernouilli(p =sigmoid(X+Z))
    
    Args:
        trial (int): trial index
        train_n (int): number of training samples
        test_n (int): number of test samples
        mc_posterior_n (int): number of smaples used in the posterior approx
        path (str, optional): Path where we store the data. Defaults to 'synth_data'.

    Returns:
        data_dict: dict containing x,z,y for train,val,test, and precomputed posteriors quantitiy for the test set.
    """

    # we set the seed with the trial index.
    torch.manual_seed(trial)
    # bounds of the Uniform
    x_train, z_train, y_train = generate_xzy(train_n)
    x_val, z_val, y_val = generate_xzy(train_n)
    x_test, z_test, y_test = generate_xzy(test_n)

    data_dict = {'y_train': y_train, 'z_train': z_train, 'x_train': x_train, 'y_val': y_val, 'z_val': z_val,
                 'x_val': x_val, 'y_test': y_test, 'z_test': z_test, 'x_test': x_test}
    data_dict['train_n'] = train_n
    data_dict['test_n'] = test_n

    print('precomputing posteriors on the test test...')
    E_max_py_xz, max_y_x, py_xz = precompute_posterior_and_store(
        x_test, z_test, num_samples=mc_posterior_n)
    data_dict['test_E'] = E_max_py_xz
    data_dict['test_max'] = max_y_x
    data_dict['test_py_xz'] = py_xz
    return data_dict


def load_data(trial: int, train_n: int, test_n: int, mc_posterior_n: int, path: str = 'synth_data'):
    """This will try to load the datasets with the right number of samples, for each trial.
        If not, it will generate it and store it into path.

    Args:
        trial (int): trial index
        train_n (int): number of training samples
        test_n (int): number of test samples
        mc_posterior_n (int): number of smaples used in the posterior approx
        path (str, optional): Path where we store the data. Defaults to 'synth_data'.

    Returns:
        data_dict: dict containing x,z,y for train,val,test, and precomputed posteriors quantitiy for the test set.
    """
    # check if the file was store
    data_filename = '_'.join(
        [str(m) for m in [trial, train_n, test_n, mc_posterior_n]]) + '_data.pk'
    path_file = os.path.join(path, data_filename)
    if os.path.exists(path_file):
        # Load the existing pickle file
        with open(path_file, 'rb') as f:
            data_dict = pk.load(f)
    else:
        data_dict = create_data(trial, train_n, test_n, mc_posterior_n)
        with open(path_file, 'wb') as f:
            pk.dump(data_dict, f)

    return data_dict


def precompute_posterior_and_store(x_values, z_values, num_samples):
    E_max_py_xz = []
    max_y_x = []
    py_xz = []
    # compute the boundary
    for i in tqdm(range(len(x_values))):
        x = x_values[i]

        # Compute the expected max p(y | x, z) over z
        expected_max_prob = expected_max_p_y_given_xz(x, num_samples)

        # Compute the marginalized max_y p(y | x)
        max_prob = posterior_p_y_given_x(x, num_samples)
        xz_post = posterior_p_y_given_x_z(x, z_values[i])

        E_max_py_xz.append(expected_max_prob)
        max_y_x.append(max_prob)
        py_xz.append(xz_post)

    return E_max_py_xz, max_y_x, py_xz
