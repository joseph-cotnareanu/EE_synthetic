import torch 
import numpy as np
from tqdm import tqdm 
sigmoid_f = torch.nn.Sigmoid()

def sigmoid(x):
    return sigmoid_f(x)
def posterior_p_y_given_x(x, n_samples):
    # Sample z from U[-1,1]
    z_samples = np.random.uniform(-1, 1, n_samples)
    
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
    z_samples = np.random.uniform(-1, 1, n_samples)
    
    # Compute sigmoid(x + z) for each z
    prob_y_given_xz = sigmoid(x + z_samples)
    
    # Compute max(p(y=1 | x, z), p(y=0 | x, z)) for each z
    max_prob_y_given_xz = np.maximum(prob_y_given_xz, 1 - prob_y_given_xz)
    
    # Return the expected value of max_y p(y | x, z)
    return torch.mean(max_prob_y_given_xz)
def posterior_p_y_given_x_z(x, z):
    # Return and p_y1_x_z
    return sigmoid(x + z)

def create_data(train_d, test_d):
    r1, r2 = -1,1
    # (r1 - r2) * torch.rand((train_d,1)) + r2
    x_train = (r1 - r2) * torch.rand((train_d,1)) + r2
    z_train = (r1 - r2) * torch.rand((train_d,1)) + r2

    y_train = (torch.ones((train_d, 1)) * torch.bernoulli(torch.nn.Sigmoid()(x_train+z_train))).type(torch.LongTensor)
    new_y = torch.zeros((train_d, 2))
    for i in range(len(y_train)):
        if y_train[i] == 0:
            new_y[i] = torch.tensor([1,0])
        elif y_train[i] == 1:
            new_y[i] = torch.tensor([0,1])

    y_train = new_y
    
    x_val = (r1 - r2) * torch.rand((train_d,1)) + r2
    z_val = (r1 - r2) * torch.rand((train_d,1)) + r2

    y_val = torch.ones((train_d, 1)) * torch.bernoulli(torch.nn.Sigmoid()(x_val+z_val))

    new_y = torch.zeros((train_d, 2))
    for i in range(len(y_val)):
        if y_val[i] == 0:
            new_y[i] = torch.tensor([1,0])
        elif y_val[i] == 1:
            new_y[i] = torch.tensor([0,1])

    y_val = new_y
   

    x_test = (r1 - r2) * torch.rand((test_d,1)) + r2
    z_test = (r1 - r2) * torch.rand((test_d,1)) + r2

    # print('l. 280', len(x_test))
    y_test = torch.ones((test_d, 1)) * torch.bernoulli(torch.nn.Sigmoid()(x_test+z_test))
    # print('l. 282', len(y_test))
    new_y = torch.zeros((test_d, 2))
    for i in range(len(y_test)):
        if y_test[i] == 0:
            new_y[i] = torch.tensor([1,0])
        elif y_test[i] == 1:
            new_y[i] = torch.tensor([0,1])

    y_test = new_y
    data_dict = {'y':y_train, 'z':z_train, 'x':x_train, 'y_val':y_val, 'z_val':z_val, 'x_val':x_val, 'y_test':y_test, 'z_test':z_test, 'x_test':x_test}
    data_dict['train_d'] = train_d
    data_dict['test_d'] = test_d
    
    # E_max_py_xz, max_y_x, py_xz = precompute_posterior_and_store(x_train, z_train)
    # data_dict['train_E'] = E_max_py_xz
    # data_dict['train_max'] = max_y_x
    # data_dict['train_py_xz'] = py_xz
    # E_max_py_xz, max_y_x, py_xz = precompute_posterior_and_store(x_val, z_val)
    # data_dict['val_E'] = E_max_py_xz
    # data_dict['val_max'] = max_y_x
    # data_dict['val_py_xz'] = py_xz
    print('precomputing posteriors on the test test...')
    E_max_py_xz, max_y_x, py_xz = precompute_posterior_and_store(x_test, z_test, num_samples=32*10000)
    data_dict['test_E'] = E_max_py_xz
    data_dict['test_max'] = max_y_x
    data_dict['test_py_xz'] = py_xz
                
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
        
    