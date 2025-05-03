
from matplotlib import pyplot as plt

import os
import pickle
figure_path = 'figures'

def only_storing_plotting_latter(training_log_dict, prefix):
    """
    This function is used to store the training log dictionary and plot the results.
    """
    storing_path = os.path.join(figure_path, 'results')
    if not os.path.exists(storing_path):
        os.makedirs(storing_path)
    with open(os.path.join(storing_path, f"{prefix}_training_result.pkl"), "wb") as f:
        pickle.dump(training_log_dict, f)
    
    
# def storing_and_plotting(training_log_dict, prefix):


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
