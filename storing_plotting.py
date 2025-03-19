from matplotlib import pyplot as plt

def storing_and_plotting(training_log_dict):
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


def plot_xzy(x,z,y, prefix):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, z, c=y, cmap="viridis", edgecolor="k", alpha=0.75)
    plt.colorbar(sc, label="t1 or y")
    plt.xlabel("X")
    plt.ylabel("Z")

    plt.savefig(prefix+'xyz.pdf')
    plt.close()
