
from storing_plotting import storing_and_plotting
from torch.utils.data import  DataLoader, TensorDataset
from train import train_two_stage_experiment, sep_hinge_experiment
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm
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
    # costs = [0.03, 0.05, 0.07] # between 0 and 0.1
    costs = list(np.arange(0.0,0.06, 0.001))
    # costs = [0.05]
    test_n = 32*1000
    train_n = 32*1000
    mc_posterior_n = 32*100
    num_trials = 1
    two_stage_model_name = 'NN' # NN
    training_configs = {'epoch':50, 'lr':0.001, 'batch_size':512}
    exp = 'both'
    cost_plot_log_sep = {}
    cost_plot_log_sep['test_avg_l01c'] = []
    cost_plot_log_sep['df_testacc'] = []
    cost_plot_log_sep['df_testrate'] = []

    cost_plot_log_2s = {}
    cost_plot_log_2s['test_avg_l01c'] = []
    cost_plot_log_2s['df_testacc'] = []
    cost_plot_log_2s['df_testrate'] = []
    for trial in range(num_trials):
        data_dict = load_data(trial = trial, train_n=train_n, test_n=test_n, mc_posterior_n=mc_posterior_n)
        
        train_loader, test_loader = data_dict_to_dataloader(data_dict)
        
        for cost in tqdm(costs):
            two_stage_model = create_two_stage_model(x_dim=1, z_dim=1, num_classes=2, two_stage_model_name=two_stage_model_name)
            if exp == 'two_stage_experiment': 
                two_stage_model, training_log_dict = train_two_stage_experiment(train_loader, test_loader, cost, two_stage_model, training_configs)

                cost_plot_log_2s['test_avg_l01c'].append(training_log_dict['test_avg_l01c'])
                cost_plot_log_2s['df_testacc'].append(training_log_dict['df_testacc'])
                cost_plot_log_2s['df_testrate'].append(training_log_dict['df_testrate'])
                cost_plot_log_2s['f1 acc'].append(training_log_dict['f1 acc'])
                cost_plot_log_2s['f2 acc'].append(training_log_dict['f2 acc'])


                storing_and_plotting(training_log_dict, prefix='2s_exp' + str(cost)+'_')

            elif exp == 'sep_hinge_experiment': 
                two_stage_model, training_log_dict = sep_hinge_experiment(train_loader, test_loader, cost, two_stage_model, training_configs)

                cost_plot_log_sep['test_avg_l01c'].append(training_log_dict['test_avg_l01c'])
                cost_plot_log_sep['df_testacc'].append(training_log_dict['df_testacc'])
                cost_plot_log_sep['df_testrate'].append(training_log_dict['df_testrate'])
                cost_plot_log_sep['f1 acc'].append(training_log_dict['f1 acc'])
                cost_plot_log_sep['f2 acc'].append(training_log_dict['f2 acc'])

                storing_and_plotting(training_log_dict, prefix='sep_exp' + str(cost)+'_')
            elif exp == 'both':
                two_stage_model, training_log_dict = train_two_stage_experiment(train_loader, test_loader, cost, two_stage_model, training_configs)

                cost_plot_log_2s['test_avg_l01c'].append(training_log_dict['test_avg_l01c'])
                cost_plot_log_2s['df_testacc'].append(training_log_dict['df_testacc'])
                cost_plot_log_2s['df_testrate'].append(training_log_dict['df_testrate'])
                cost_plot_log_2s['f1 acc'].append(training_log_dict['f1 acc'])
                cost_plot_log_2s['f2 acc'].append(training_log_dict['f2 acc'])

                storing_and_plotting(training_log_dict, prefix='2s_exp' + str(cost)+'_')

                two_stage_model, training_log_dict = sep_hinge_experiment(train_loader, test_loader, cost, two_stage_model, training_configs)

                cost_plot_log_sep['test_avg_l01c'].append(training_log_dict['test_avg_l01c'])
                cost_plot_log_sep['df_testacc'].append(training_log_dict['df_testacc'])
                cost_plot_log_sep['df_testrate'].append(training_log_dict['df_testrate'])
                cost_plot_log_sep['f1 acc'].append(training_log_dict['f1 acc'])
                cost_plot_log_sep['f2 acc'].append(training_log_dict['f2 acc'])

                storing_and_plotting(training_log_dict, prefix='sep_exp' + str(cost)+'_')
            print('========== done cost = ' + str(cost) + ' ==========')
        # breakpoint()
        fig, ax = plt.subplots(3, 2, figsize=(10,15))
        ax[0,0].scatter(x=costs, y=cost_plot_log_2s['test_avg_l01c'], label='2-stage experiment')
        ax[0,0].scatter(x=costs, y=cost_plot_log_sep['test_avg_l01c'], label='separate training experiment')
        ax[0,0].set_title('average test-set l01c')
        ax[0,0].set_ylabel('l01c')
        ax[0,0].set_xlabel('cost')
        ax[0,0].legend()

        ax[1,0].scatter(x=costs, y=cost_plot_log_2s['df_testacc'], label='2-stage')
        ax[1,0].scatter(x=costs, y=cost_plot_log_sep['df_testacc'], label='seperate')
        ax[1,0].set_title('average test-set deferral accuracy')
        ax[1,0].set_ylabel('deferral accuracy')
        ax[1,0].set_xlabel('cost')
        ax[1,0].legend()

        ax[1,1].scatter(x=costs, y=cost_plot_log_2s['df_testrate'], label='2-stage')
        ax[1,1].scatter(x=costs, y=cost_plot_log_sep['df_testrate'], label='seperate')
        ax[1,1].set_title('average test-set deferral rate')
        ax[1,1].set_ylabel('deferral rate')
        ax[1,1].set_xlabel('cost')
        ax[1,1].legend()

        ax[2,0].scatter(x=costs, y=cost_plot_log_2s['f1 acc'], label='2-stage')
        ax[2,0].scatter(x=costs, y=cost_plot_log_sep['f1 acc'], label='2-stage')
        ax[2,0].set_title('f1 acc')
        ax[2,0].set_ylabel('accuracy')
        ax[2,0].set_xlabel('cost')
        ax[2,0].legend()


        ax[2,1].scatter(x=costs, y=cost_plot_log_2s['f2 acc'], label='2-stage')
        ax[2,1].scatter(x=costs, y=cost_plot_log_sep['f2 acc'], label='2-stage')
        ax[2,1].set_title('f2 acc')
        ax[2,1].set_ylabel('accuracy')
        ax[2,1].set_xlabel('cost')
        ax[2,1].legend()



        plt.tight_layout
        plt.savefig('./figures/costfig.pdf')
        plt.close()




