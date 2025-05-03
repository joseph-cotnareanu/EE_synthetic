
from storing_plotting import only_storing_plotting_latter
from torch.utils.data import  DataLoader, TensorDataset
from train import train_two_stage_experiment
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm
seed = 42  # or any number you choose

    
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
    py_x = data_dict['test_py_x']
    train_dataset = TensorDataset(x_train, z_train, y_train)
    # Create DataLoader with shuffling
    train_loader = DataLoader(train_dataset, batch_size=training_configs['batch_size'], shuffle=True)
    
    test_dataset = TensorDataset(x_test, z_test, y_test,E_max_py_xz, max_y_x, py_xz, py_x)
    # Create DataLoader with shuffling
    test_loader = DataLoader(test_dataset, batch_size=training_configs['batch_size'], shuffle=False)
    return train_loader, test_loader

if __name__ == '__main__':
   
    #costs = list(np.arange(0.01,0.09, 0.01))
    costs = [0.07, 0.03]
    test_n = 32*10000
    train_n = 32*10000
    hidden_dim = 8
    mc_posterior_n = 32*100
    num_trials = 1
    two_stage_model_name = 'NN' # NN
    training_configs = {'epoch':5, 'lr':0.001, 'batch_size':512}
    
    exp = 'two_stage_experiment'
    cost_plot_log_sep = {'name':'sep_hinge_experiment'}
    cost_plot_log_2s = {'name':'two_stage_experiment'}

    
    baseline_dicts = [cost_plot_log_2s, cost_plot_log_sep]
    
    for base_dict in baseline_dicts:
        base_dict['test_avg_l01c'] = []
        base_dict['df_testacc'] = []
        base_dict['df_testrate'] = []
        base_dict['f1 acc'] = []
        base_dict['f2 acc'] = []
    
    for trial in range(num_trials):
        data_dict = load_data(trial = trial, train_n=train_n, test_n=test_n, mc_posterior_n=mc_posterior_n)
        
        train_loader, test_loader = data_dict_to_dataloader(data_dict)
        
        for cost in tqdm(costs):
            
            if exp == 'two_stage_experiment' or  exp == 'both': 
                two_stage_model = create_two_stage_model(x_dim=1, z_dim=1, num_classes=1, hidden_dim=hidden_dim, two_stage_model_name=two_stage_model_name)
                training_configs['loss_type'] = 'hinge_surrogate'
                two_stage_model, training_log_dict = train_two_stage_experiment(train_loader, test_loader, cost, two_stage_model, training_configs)

                cost_plot_log_2s['test_avg_l01c'].append(training_log_dict['test_avg_l01c'])
                cost_plot_log_2s['df_testacc'].append(training_log_dict['df_testacc'])
                cost_plot_log_2s['df_testrate'].append(training_log_dict['df_testrate'])
                cost_plot_log_2s['f1 acc'].append(training_log_dict['f1 acc'])
                cost_plot_log_2s['f2 acc'].append(training_log_dict['f2 acc'])


                only_storing_plotting_latter(training_log_dict, prefix='2s_exp' + str(cost)+'_')

            if exp == 'sep_hinge_experiment' or  exp == 'both': 
                two_stage_model = create_two_stage_model(x_dim=1, z_dim=1, num_classes=2, two_stage_model_name=two_stage_model_name)
                training_configs['loss_type'] = 'separate'
                two_stage_model, training_log_dict = train_two_stage_experiment(train_loader, test_loader, cost, two_stage_model, training_configs)

                cost_plot_log_sep['test_avg_l01c'].append(training_log_dict['test_avg_l01c'])
                cost_plot_log_sep['df_testacc'].append(training_log_dict['df_testacc'])
                cost_plot_log_sep['df_testrate'].append(training_log_dict['df_testrate'])
                cost_plot_log_sep['f1 acc'].append(training_log_dict['f1 acc'])
                cost_plot_log_sep['f2 acc'].append(training_log_dict['f2 acc'])

                only_storing_plotting_latter(training_log_dict, prefix='sep_exp' + str(cost)+'_')
           
        only_storing_plotting_latter(baseline_dicts, prefix='baseline_' + str(costs) + '_')