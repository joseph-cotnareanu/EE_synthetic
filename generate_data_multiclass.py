import torch
import numpy as np
from tqdm import tqdm
import os
import pickle as pk
from torch.nn.functional import sigmoid



def expected_max_p_y_given_xz_and_posterior_p_y_given_x_simple(x, n_samples, n_classes):
    x = x.item()
    # Sample z from U[-1,1]
    z_samples = np.random.uniform(-1, 1, n_samples)

    # Compute prob y given x,z
    thetas = np.zeros((n_classes, n_samples))
    
    slope = n_classes - 1
    shift = [n_classes - 2 -2*i for i in range(n_classes-1)]
    
    x_bins = np.linspace(-1, 1, n_classes)
    
    #find the bin for x
    bin_index = np.digitize(x, x_bins) - 1
    if bin_index == n_classes - 1: #correct in case x = 1
        bin_index = n_classes - 2
    
    #assign probability to the two classes that are non-zero for this bin
    thetas[bin_index] =     logistic( -1*((slope*x + shift[bin_index]) + z_samples))
    thetas[bin_index + 1] = logistic(     (slope*x + shift[bin_index]) + z_samples )
    
    #thetas is prob_y_given_xz

    # Compute max(p(y=k | x, z), for each class, z
    max_prob_y_given_xz = np.max(thetas, axis=0) # axis 

    # Return the expected value of max_y p(y | x, z), and the p(y|x)
    return torch.tensor(np.mean(max_prob_y_given_xz)), torch.tensor(np.mean(thetas, axis=1))


def posterior_p_y_given_x_z_simple(x, z, n_classes):
    x = x.item()
    z = z.item()
    
    thetas = np.zeros((n_classes))
    slope = n_classes - 1
    shift = [n_classes - 2 -2*i for i in range(n_classes-1)]
    
    x_bins = np.linspace(-1, 1, n_classes)
    
    #find the bin for x
    bin_index = np.digitize(x, x_bins) - 1
    if bin_index == n_classes - 1: #correct in case x = 1
        bin_index = n_classes - 2
    
    thetas[bin_index] =     logistic( -1*((slope*x + shift[bin_index]) + z))
    thetas[bin_index + 1] = logistic(     (slope*x + shift[bin_index]) + z )
    
    
    # Return and p_y_given_x_z
    return torch.tensor(thetas)


def logistic(k):
    return 1 / (1 + np.exp(-k))

def m_generate_xzy_simple(n: int, n_classes: int):

    # sample on the uniform distribution
    x = np.random.uniform(-1, 1, n)
    z = np.random.uniform(-1, 1, n)
    
    thetas = [np.zeros_like(x) for _ in range(n_classes)]
    
    #slope and shift define the lines at each class boundary
    slope = n_classes - 1
    shift = [n_classes - 2 -2*i for i in range(n_classes-1)]
    
    x_bins = np.linspace(-1, 1, n_classes)

    y_flat = np.zeros_like(x)
    
    labels = [i for i in range(n_classes)]
    
    for i in range(n):
        #find the bin for x_samples[i]
        bin_index = np.digitize(x[i], x_bins) - 1
        if bin_index == n_classes - 1: #correct in case x = 1
            bin_index = n_classes - 2
        
        thetas[bin_index][i] =     logistic( -1*((slope*x[i] + shift[bin_index]) + z[i]))
        thetas[bin_index + 1][i] = logistic(     (slope*x[i] + shift[bin_index]) + z[i] )
        
        y_flat[i] = np.random.choice(labels, p=[thetas[k][i] for k in range(n_classes)])
    
    y_flat = torch.tensor(y_flat).long()
    
    y = torch.nn.functional.one_hot(y_flat, num_classes=n_classes)
    
    x = x.reshape(-1, 1)
    z = z.reshape(-1, 1)
    
    return torch.tensor(x).float(), torch.tensor(z).float(), y


def create_data(trial: int, train_n: int, test_n: int, mc_posterior_n: int, n_classes: int, type: str = 'simple'):
    
    """
        This generate data and return it, following our  model
        X \\sim Uniform[-1,1]
        Z \\sim Uniform[-1,1]
        Y \\sim Depends on the type
    
    Args:
        trial (int): trial index
        train_n (int): number of training samples
        test_n (int): number of test samples
        mc_posterior_n (int): number of smaples used in the posterior approx
        n_classes (int): number of classes
        type (str, optional): type of data to generate. Defaults to 'simple'.

    Returns:
        data_dict: dict containing x,z,y for train,val,test, and precomputed posteriors quantitiy for the test set.
    """

    # we set the seed with the trial index.
    torch.manual_seed(trial)
    #switch case statement
    if type == 'simple':
        x_train, z_train, y_train = m_generate_xzy_simple(train_n, n_classes)
        x_val, z_val, y_val = m_generate_xzy_simple(train_n, n_classes)
        x_test, z_test, y_test = m_generate_xzy_simple(test_n, n_classes)

    data_dict = {'y_train': y_train, 'z_train': z_train, 'x_train': x_train, 'y_val': y_val, 'z_val': z_val,
                 'x_val': x_val, 'y_test': y_test, 'z_test': z_test, 'x_test': x_test}
    data_dict['train_n'] = train_n
    data_dict['test_n'] = test_n

    print('precomputing posteriors on the test test...')
    E_max_py_xz, max_y_x, py_xz, py_x = m_precompute_posterior_and_store_simple(
        x_test, z_test, num_samples=mc_posterior_n, n_classes=n_classes)
    data_dict['test_E'] = E_max_py_xz
    data_dict['test_max'] = max_y_x
    data_dict['test_py_xz'] = py_xz
    data_dict['test_py_x'] = py_x
    
    return data_dict


def load_data(trial: int, train_n: int, test_n: int, mc_posterior_n: int, n_classes: int, path: str = 'synth_data'):
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
        [str(m) for m in [trial, train_n, test_n, mc_posterior_n, 'multiclass', n_classes]]) + '_data.pk'
    path_file = os.path.join(path, data_filename)
    if os.path.exists(path_file):
        # Load the existing pickle file
        with open(path_file, 'rb') as f:
            data_dict = pk.load(f)
    else:
        data_dict = create_data(trial, train_n, test_n, mc_posterior_n, n_classes=n_classes)
        with open(path_file, 'wb') as f:
            pk.dump(data_dict, f)

    return data_dict


def m_precompute_posterior_and_store_simple(x_values, z_values, num_samples, n_classes):
    E_max_py_xz = []
    max_y_x = []
    py_xz = torch.zeros((len(x_values), n_classes))
    py_x = torch.zeros((len(x_values), n_classes))
    # compute the boundary
    for i in tqdm(range(len(x_values))):
        x = x_values[i]

        # Compute the expected max p(y | x, z) over z and the marginalized max_y p(y | x)
        expected_max_prob, prob_y_x = expected_max_p_y_given_xz_and_posterior_p_y_given_x_simple(x, num_samples, n_classes)

        py_x[i] = prob_y_x
        max_prob = torch.max(prob_y_x)
        xz_post = posterior_p_y_given_x_z_simple(x, z_values[i], n_classes)

        E_max_py_xz.append(expected_max_prob)
        max_y_x.append(max_prob)
        py_xz[i] = xz_post

    return torch.tensor(E_max_py_xz), torch.tensor(max_y_x), py_xz, py_x
