
from storing_plotting import plotting_multi_costs, storing_and_plotting
from torch.utils.data import  DataLoader, TensorDataset
from train import train_two_stage_experiment
from train_llm import train_two_stage_experiment as train_llm 
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import torch
# seed = 42  # or any number you choose

    
from create_model import create_two_stage_model, create_llm_model
from generate_data import load_data

def data_dict_to_dataloader():
    # x_train = data_dict['x_train']
    # y_train = data_dict['y_train']
    # z_train = data_dict['z_train']
    # x_test = data_dict['x_test']
    # y_test = data_dict['y_test']
    # z_test = data_dict['z_test']
    # E_max_py_xz = data_dict['test_E']
    # max_y_x = data_dict['test_max']
    # py_xz = data_dict['test_py_xz']
    # py_x = data_dict['test_py_x']

    from datasets import load_dataset
    import json
    import os
    alphabet = {'A':[1,0,0,0,0], 'B':[0,1,0,0,0], 'C':[0,0,1,0,0], 'D':[0,0,0,1,0], 'E':[0,0,0,0,1]}
    alpha_num = {0: 'A', 1: 'B', 2:'C', 3:'D', 4: 'E'}
    ds = load_dataset("hails/agieval-aqua-rat")

    ds = ds['test']
    y_test = []
    # for d in ds:
    #     y_test.append(d[0])
    for d in ds['gold']:
        # print(d)
        y_test.append(alphabet[alpha_num[d[0]]])
    

    aqua_training_path = '/home/joseph/EE_llm/aqua_train.json'
    a_t = open(aqua_training_path, 'r',  encoding='utf-8-sig')
    aqua_training = []
    for line in a_t.readlines():
        datum=json.loads(line)
        tmp = {}
        tmp['query'] = datum['question']
        option_str = 'Answer Choices: '
        for option in datum['options']:
            option_str += '(' + option + ' '
        option_str = option_str[:-1] +'\n Among A through E, the answer is,'
        tmp['query'] += option_str
        tmp['gold'] = [datum['correct']]
        aqua_training.append(tmp)
    y_train = []
    for d in aqua_training[:1000]:
        y_train.append(alphabet[d['gold'][0]])
    # print(y_train)
    y_val = y_train[800:900]
    y_test = y_train[900:1000]
    y_train = y_train[:800]


    # c = torch.ones(len(y_train))*cost
    x_train_path = '/home/joseph/home/josephc/scratch/logits/aqua_logits_train_8B/'
    x_test_path = '//home/joseph/home/josephc/scratch/logits/aqua_logits_8B/'
    z_train_path = '/home/joseph/home/josephc/scratch/logits/aqua_logits_train_70B/'
    z_test_path = '/home/joseph/home/josephc/scratch/logits/aqua_logits_70B/'

    x_train = []
    z_train = []
    for i in range(1,801):
        try:
            x_train.append(np.load(open(x_train_path + str(i) + '_.npy', 'rb'))[-1, :].squeeze())
            z_train.append(np.load(open(z_train_path + str(i) + '_.npy', 'rb'))[-1, :].squeeze())
        except:
            print(i)
    x_val = []
    z_val = []
    for i in range(801, 901):
        x_val.append(np.load(open(x_train_path + str(i) + '_.npy', 'rb'))[-1, :].squeeze())
        z_val.append(np.load(open(z_train_path + str(i) + '_.npy', 'rb'))[-1, :].squeeze())

    x_test = []
    z_test = []
    for i in range(901,1001):
        x_test.append(np.load(open(x_test_path + str(i) + '_.npy', 'rb'))[-1, :].squeeze())
        z_test.append(np.load(open(z_test_path + str(i) + '_.npy', 'rb'))[-1, :].squeeze())

    # print('l. 291', len(y_test))
    # print(y_train)
    x_train = torch.from_numpy(np.stack(x_train)).float()
    xdim = x_train.shape[-1]
    z_train = torch.from_numpy(np.stack(z_train)).float()
    zdim = z_train.shape[-1]
    y_train = torch.from_numpy(np.stack(y_train)).float()
    # print(y_train)
    # breakpoint()

    x_val = torch.from_numpy(np.stack(x_val)).float()
    z_val = torch.from_numpy(np.stack(z_val)).float()
    y_val = torch.from_numpy(np.stack(y_val)).float()

    # breakpoint()
    x_test = torch.from_numpy(np.stack(x_test)).float()
    z_test = torch.from_numpy(np.stack(z_test)).float()
    y_test = torch.from_numpy(np.stack(y_test)).float()




    train_dataset = TensorDataset(x_train, z_train, y_train)




    # breakpoint()
    # Create DataLoader with shuffling
    train_loader = DataLoader(train_dataset, batch_size=training_configs['batch_size'], shuffle=True)
    
    # test_dataset = TensorDataset(x_test, z_test, y_test,E_max_py_xz, max_y_x, py_xz, py_x)
    test_dataset = TensorDataset(x_test, z_test, y_test)
    test_loader = DataLoader(test_dataset,batch_size=len(test_dataset), shuffle=False)

    val_dataset = TensorDataset(x_val, z_val, y_val)
    val_loader = DataLoader(val_dataset, shuffle=False)

    print('train balance:', np.unique(y_train.max(dim=-1).indices, return_counts=True)[1]/len(y_train))
    # Create DataLoader with shuffling
    # test_loader = DataLoader(test_dataset, batch_size=training_configs['batch_size'], shuffle=False)
    return train_loader, val_loader, test_loader, xdim, zdim

if __name__ == '__main__':
   
    # costs = list(np.arange(0.01,0.09, 0.01))
    costs = list(np.arange(0.001, 0.01, 0.002))
    # costs = [0.001, 0.01, 0.1,0.5]
    #costs = [0.05]
    test_n = 32*10000
    train_n = 32*10000
    mc_posterior_n = 32*100
    num_trials = 1
    two_stage_model_name = 'NN' # NN
    training_configs = {'epoch':50, 'lr':0.0001, 'batch_size':32, 'data': 'llm', 'nlayers': 3} #data: llm or toy
    
    exp = 'two_stage_experiment'
    # exp = 'sep_hinge_experiment'
    cost_plot_log_sep = {'name':'sep_hinge_experiment'}
    cost_plot_log_2s = {'name':'two_stage_experiment'}

    
    baseline_dicts = [cost_plot_log_2s, cost_plot_log_sep]
    # baseline_dicts = [cost_plot_log_2s]
    # baseline_dicts = [cost_plot_log_sep]
    
    for base_dict in baseline_dicts:
        base_dict['test_avg_l01c'] = []
        base_dict['df_testacc'] = []
        base_dict['df_testrate'] = []
        base_dict['f1 acc'] = []
        base_dict['f2 acc'] = []
    
    for trial in range(num_trials):
        # data_dict = load_data(trial = trial, train_n=train_n, test_n=test_n, mc_posterior_n=mc_posterior_n)
        
        train_loader, val_loader, test_loader, xdim, zdim = data_dict_to_dataloader()
        
        for cost in tqdm(costs):
            two_stage_model = create_llm_model(x_dim=xdim, z_dim=zdim, nlayers=training_configs['nlayers'], num_classes=5, hidden_dim=16, two_stage_model_name=two_stage_model_name)
            if exp == 'two_stage_experiment' or  exp == 'both': 
                training_configs['loss_type'] = 'hinge_surrogate'
                two_stage_model, training_log_dict = train_llm(train_loader, test_loader, cost, two_stage_model, training_configs)

                cost_plot_log_2s['test_avg_l01c'].append(training_log_dict['test_avg_l01c'])
                cost_plot_log_2s['df_testacc'].append(training_log_dict['df_testacc'])
                cost_plot_log_2s['df_testrate'].append(training_log_dict['df_testrate'])
                cost_plot_log_2s['f1 acc'].append(training_log_dict['f1 acc'])
                cost_plot_log_2s['f2 acc'].append(training_log_dict['f2 acc'])


                storing_and_plotting(training_log_dict, prefix='llm_2s_exp' + str(cost)+'_')

            if exp == 'sep_hinge_experiment' or  exp == 'both': 
                training_configs['loss_type'] = 'separate'
                two_stage_model, training_log_dict = train_llm(train_loader, test_loader, cost, two_stage_model, training_configs)

                cost_plot_log_sep['test_avg_l01c'].append(training_log_dict['test_avg_l01c'])
                cost_plot_log_sep['df_testacc'].append(training_log_dict['df_testacc'])
                cost_plot_log_sep['df_testrate'].append(training_log_dict['df_testrate'])
                cost_plot_log_sep['f1 acc'].append(training_log_dict['f1 acc'])
                cost_plot_log_sep['f2 acc'].append(training_log_dict['f2 acc'])

                storing_and_plotting(training_log_dict, prefix='sep_exp' + str(cost)+'_')
           
        plotting_multi_costs(costs, baseline_dicts)