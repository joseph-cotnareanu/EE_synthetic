from matplotlib import pyplot as plt

import os
figure_path = 'figures'
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
    l01cs = training_log_dict['l01cs']
    test_01c = training_log_dict['track_01c']
    optimal_l01c = training_log_dict['optimal_l01c']
    # track_l01c = l01c()
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

    


    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].plot(track_t1_acc, label='Accuracy 1', marker='o')
    ax[0].set_title('f1')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    
    ax[1].plot(track_t2_acc, label='Accuracy 2', marker='o', color='r')
    ax[1].set_title('f2')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, prefix+'acc.pdf'))
    plt.close()

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    
    ax[0,0].plot(ls, label='surrogate', marker='o')
    ax[0,0].set_title('surrogate loss')
    ax[0,0].set_xlabel('Epoch')
    ax[0,0].set_ylabel('Accuracy')
    ax[0,0].legend()
    
    ax[0,1].plot(l01cs, label='l01c on decision rule', marker='o')
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
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
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
        fig, ax = plt.subplots(3, 2, figsize=(10,15))
        for base_dict in baseline_dicts:
            ax[0,0].scatter(x=costs, y=base_dict['test_avg_l01c'], label=base_dict['name'])
        
        ax[0,0].set_title('average test-set l01c')
        ax[0,0].set_ylabel('l01c')
        ax[0,0].set_xlabel('cost')
        ax[0,0].legend()

        for base_dict in baseline_dicts:
            ax[1,0].scatter(x=costs, y=base_dict['df_testacc'], label=base_dict['name'])
        
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



        plt.tight_layout
        plt.savefig('./figures/costfig.pdf')
        plt.close()
