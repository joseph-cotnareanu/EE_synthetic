
from storing_plotting import storing_and_plotting
from torch.utils.data import  DataLoader, TensorDataset
from train import train_two_stage_experiment
seed = 42  # or any number you choose

# # Python random seed
# random.seed(seed)

# # NumPy random seed
# np.random.seed(seed)

# # PyTorch random seed
# torch.manual_seed(seed)

# # If you are using CUDA
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
from create_model import create_two_stage_model
from generate_data import load_data

def data_dict_to_dataloader(data_dict):
    x_train = data_dict['x_train']
    y_train = data_dict['y_train']
    z_train = data_dict['z_train']
    x_test = data_dict['x_test']
    y_test = data_dict['y_test']
    z_test = data_dict['z_test']
    E_max_py_xz = data_dict['test_E']
    max_y_x = data_dict['test_max']
    py_xz = data_dict['test_py_xz']
    train_dataset = TensorDataset(x_train, z_train, y_train)
    # Create DataLoader with shuffling
    train_loader = DataLoader(train_dataset, batch_size=training_configs['batch_size'], shuffle=True)
    
    test_dataset = TensorDataset(x_test, z_test, y_test,E_max_py_xz, max_y_x, py_xz)
    # Create DataLoader with shuffling
    test_loader = DataLoader(test_dataset, batch_size=training_configs['batch_size'], shuffle=False)
    return train_loader, test_loader

if __name__ == '__main__':
    costs = [0.03, 0.05, 0.07] # between 0 and 0.1
    test_n = 32*100
    train_n = 32*100
    mc_posterior_n = 32*10
    num_trials = 1
    two_stage_model_name = 'NN' # NN
    training_configs = {'epoch':10, 'lr':0.001, 'batch_size':128}
    for trial in range(num_trials):
        data_dict = load_data(trial = trial, train_n=train_n, test_n=test_n, mc_posterior_n=mc_posterior_n)
        
        train_loader, test_loader = data_dict_to_dataloader(data_dict)
        
        for cost in costs:
            two_stage_model = create_two_stage_model(x_dim=1, z_dim=1, num_classes=2, two_stage_model_name=two_stage_model_name)
            two_stage_model, training_log_dict = train_two_stage_experiment(train_loader, test_loader, cost, two_stage_model, training_configs)


            storing_and_plotting(training_log_dict, prefix=str(cost)+'_')

