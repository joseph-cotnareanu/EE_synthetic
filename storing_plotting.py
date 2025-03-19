from matplotlib import pyplot as plt

def storing_and_plotting(training_log_dict):

    ls = training_log_dict['ls']
    f1ls = training_log_dict['f1ls']
    f2ls = training_log_dict['f2ls']
    param_cs = training_log_dict['param_cs']
    param_ds = training_log_dict['param_ds']
    track_t1_acc = training_log_dict['track_t1_acc']
    track_t2_acc = training_log_dict['track_t2_acc']
    fig1, ax1 = plt.subplots()
    ax1.plot(param_cs)
    ax1.set_title('cplot')
    fig1.savefig('cplot.pdf')
    plt.close()
    fig1, ax1 = plt.subplots()
    ax1.plot(param_ds)
    ax1.set_title('dplot')
    fig1.savefig('dplot.pdf')
    plt.close()
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
    plt.savefig('acc.pdf')
    plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].plot(ls, label='surrogate', marker='o')
    ax[0].set_title('surrogate loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    
    ax[1].plot(f1ls, label='f1 loss', marker='o', color='r')
    ax[1].set_title('f1 loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('loss')
    ax[1].legend()
    
    ax[2].plot(f2ls, label='f2 loss', marker='o', color='r')
    ax[2].set_title('f2 loss')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('loss')
    ax[2].legend()

    plt.tight_layout()
    plt.savefig('losses.pdf')



def plot_xzy(x,z,y, prefix):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, z, c=y, cmap="viridis", edgecolor="k", alpha=0.75)
    plt.colorbar(sc, label="t1 or y")
    plt.xlabel("X")
    plt.ylabel("Z")

    plt.savefig(prefix+'xyz.pdf')
    plt.close()
