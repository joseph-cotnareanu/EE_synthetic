from matplotlib import pyplot as plt
from matplotlib_venn import venn3
import os
import numpy as np
import pickle
figure_path = 'figures'

def venn_region_counts(A, B, C):
    A = np.array(A, dtype=bool)
    B = np.array(B, dtype=bool)
    C = np.array(C, dtype=bool)
    
    regions = {
        'Abc': np.sum(A & ~B & ~C),
        'aBc': np.sum(~A & B & ~C),
        'ABc': np.sum(A & B & ~C),
        'abC': np.sum(~A & ~B & C),
        'AbC': np.sum(A & ~B & C),
        'aBC': np.sum(~A & B & C),
        'ABC': np.sum(A & B & C),
    }

    return regions


def only_storing_plotting_latter(training_log_dict, prefix):
    """
    This function is used to store the training log dictionary and plot the results.
    """
    with open(os.path.join(figure_path, f"{prefix}_training_result.pkl"), "wb") as f:
        pickle.dump(training_log_dict, f)
    
    
def storing_and_plotting(training_log_dict, prefix):

   
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    ls = training_log_dict['ls']
    f1ls = training_log_dict['f1ls']
    f2ls = training_log_dict['f2ls']
    param_cs = training_log_dict['param_cs']
    param_ds = training_log_dict['param_ds']
    track_t1_acc = training_log_dict['track_t1_acc']
    track_t2_acc = training_log_dict['track_t2_acc']

    track_t1s_acc = training_log_dict['track_t1s_acc']
    track_t2s_acc = training_log_dict['track_t2s_acc']

    track_df  = training_log_dict['track_df']
    f1_s_acc = training_log_dict['f1s_acc']
    f2_s_acc = training_log_dict['f2s_acc']

    # l01cs = training_log_dict['l01cs']
    test_01c = training_log_dict['track_01c']
    if 'optimal_lo1c' in training_log_dict.keys(): 
        optimal_l01c = training_log_dict['optimal_l01c']
    # track_l01c = training_log_dict['track_01c']
    # fig1, ax1 = plt.subplots()
    # ax1.plot(param_cs)
    # ax1.set_title('cplot')
    # fig1.savefig(os.path.join(figure_path, prefix+'cplot.pdf'))
    # plt.close()
    # fig1, ax1 = plt.subplots()
    # ax1.plot(param_ds)
    # ax1.set_title('dplot')
    # fig1.savefig(os.path.join(figure_path, prefix+'dplot.pdf'))
    # plt.close()

    


    fig, ax = plt.subplots(3, 2, figsize=(10, 15))
    # breakpoint()
    ax[0,0].plot(track_t1_acc, label='Accuracy 1', marker='o')
    ax[0,0].set_title('f1')
    ax[0,0].set_xlabel('Epoch')
    ax[0,0].set_ylabel('Accuracy')
    ax[0,0].legend()
    
    ax[0,1].plot(track_t2_acc, label='Accuracy 2', marker='o', color='r')
    ax[0,1].set_title('f2')
    ax[0,1].set_xlabel('Epoch')
    ax[0,1].set_ylabel('Accuracy')
    ax[0,1].legend()

    ax[1,0].plot(track_df, label='deferral rate', marker='o', color='g')
    ax[1,0].set_title('deferral rate')
    ax[1,0].set_xlabel('Epoch')
    ax[1,0].set_ylabel('Rate of Deferral to f2')
    
    ax[2,0].plot(track_t1s_acc, label='Selected Accuracy 1', marker='o')
    ax[2,0].set_title('selected f1')
    ax[2,0].set_xlabel('Epoch')
    ax[2,0].set_ylabel('Accuracy')
    ax[2,0].legend()

    ax[2,1].plot(track_t2s_acc, label='Selected Accuracy 2', marker='o')
    ax[2,1].set_title('selected f2')
    ax[2,1].set_xlabel('Epoch')
    ax[2,1].set_ylabel('Accuracy')
    ax[2,1].legend()


    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, prefix+'acc.pdf'))
    plt.close()

    fig, ax = plt.subplots(3, 2, figsize=(10, 15))
    
    ax[0,0].plot(ls, label='surrogate', marker='o')
    ax[0,0].set_title('surrogate loss')
    ax[0,0].set_xlabel('Epoch')
    ax[0,0].set_ylabel('Accuracy')
    ax[0,0].legend()
    
    ax[0,1].plot(test_01c, label='l01c on decision rule', marker='o')
    ax[0,1].set_title('l01c on decision rule')
    ax[0,1].set_xlabel('Epoch')
    ax[0,1].set_ylabel('Accuracy')
    ax[0,1].legend()

    ax[1,0].plot(f1ls, label='f1 loss', marker='o', color='r')
    ax[1,0].set_title('f1 loss')
    ax[1,0].set_xlabel('Epoch')
    ax[1,0].set_ylabel('loss')
    ax[1,0].legend()
    
    ax[1,1].plot(f2ls, label='f2 loss', marker='o', color='r')
    ax[1,1].set_title('f2 loss')
    ax[1,1].set_xlabel('Epoch')
    ax[1,1].set_ylabel('loss')
    ax[1,1].legend()
    


    plt.tight_layout()
    plt.savefig(os.path.join(figure_path,prefix+'losses.pdf'))
    plt.close()

    if 'optimal_lo1c' in training_log_dict.keys(): 

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        plt.plot(test_01c, label=r'$R_{01c}(f)$', marker='o')
        plt.axhline(y=optimal_l01c, color='r', linestyle='--', linewidth=2, label=r'$R^*_{01c}$')
        plt.xlabel('Epoch')
        plt.ylabel(r'$R_{01c}(f)$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figure_path, prefix+'R01c.pdf'))
        plt.close()

def plot_xzy(x,z,y, prefix):
    num_points_max = 5000
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x[:num_points_max,:], z[:num_points_max,:], c=y[:num_points_max], cmap="viridis", edgecolor="k", alpha=0.5)
    plt.colorbar(sc)
    plt.xlabel("X")
    plt.ylabel("Z")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    plt.savefig(os.path.join(figure_path,prefix+'xyz.pdf'))
    plt.close()

def plotting_multi_costs(costs, baseline_dicts):
    
        # breakpoint()
        # breakpoint()
        fig, ax = plt.subplots(5, 2, figsize=(10,25))
        for base_dict in baseline_dicts:
            try:
                ax[0,0].scatter(x=costs, y=base_dict['test_avg_l01c'], label=base_dict['name'])
            except: breakpoint()
        
        ax[0,0].set_title('average test-set l01c')
        ax[0,0].set_ylabel('l01c')
        ax[0,0].set_xlabel('cost')
        ax[0,0].legend()

        for base_dict in baseline_dicts:
            ax[1,0].scatter(x=costs, y=base_dict['df_testacc'], label=base_dict['name'])
            # breakpoint()
        ax[1,0].set_title('average test-set deferral accuracy')
        ax[1,0].set_ylabel('deferral accuracy')
        ax[1,0].set_xlabel('cost')
        ax[1,0].legend()

        for base_dict in baseline_dicts:
            ax[1,1].scatter(x=costs, y=base_dict['df_testrate'], label=base_dict['name'])
        ax[1,1].set_title('average test-set deferral rate')
        ax[1,1].set_ylabel('deferral rate')
        ax[1,1].set_xlabel('cost')
        ax[1,1].legend()

        for base_dict in baseline_dicts:
            ax[2,0].scatter(x=costs, y=base_dict['f1 acc'], label=base_dict['name'])
    
        ax[2,0].set_title('f1 acc')
        ax[2,0].set_ylabel('accuracy')
        ax[2,0].set_xlabel('cost')
        ax[2,0].legend()

        for base_dict in baseline_dicts:
            ax[2,1].scatter(x=costs, y=base_dict['f2 acc'], label=base_dict['name'])
        ax[2,1].set_title('f2 acc')
        ax[2,1].set_ylabel('accuracy')
        ax[2,1].set_xlabel('cost')
        ax[2,1].legend()

        for base_dict in baseline_dicts:
            ax[3,0].scatter(x=costs, y=base_dict['f1 s acc'], label=base_dict['name'])
        ax[3,0].set_title('f1 selected acc')
        ax[3,0].set_ylabel('accuracy')
        ax[3,0].set_xlabel('cost')
        ax[3,0].legend()

        for base_dict in baseline_dicts:
            ax[3,1].scatter(x=costs, y=base_dict['f2 s acc'], label=base_dict['name'])
        ax[3,1].set_title('f2 selected acc')
        ax[3,1].set_ylabel('accuracy')
        ax[3,1].set_xlabel('cost')
        ax[3,1].legend()

        i = 0
        xes = []
        yes = []
        for c in costs:
            # breakpoint()    

            xes += [np.log(c)]*100
            yes += list(range(100))
        for base_dict in baseline_dicts:
            s = base_dict['s']
            ax[4,i].scatter(x=xes, y=yes, c = s)
            ax[4,i].set_title('deferral selection over testset')
            ax[4,i].set_ylabel('test-set indices')
            ax[4,i].set_xlabel('log cost')
            i += 1

       

        plt.tight_layout
        plt.savefig('./figures/costfig.pdf')
        plt.close()

        s0 = baseline_dicts[0]['s'][0]
        s1 = baseline_dicts[0]['s'][-2]
        s2 = baseline_dicts[0]['s'][-1]

        vrc = venn_region_counts(s0, s1, s2)

        v = venn3(subsets = list(vrc.values()), set_labels=('cost = ' + str(costs[0]), 'cost = ' + str(costs[-2]), 'cost = ' + str(costs[-1])))
        plt.title("Overlapping selections over costs")

        plt.tight_layout
        plt.savefig('./figures/venn.pdf')
        plt.close()