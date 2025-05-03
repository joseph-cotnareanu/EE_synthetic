
from matplotlib import pyplot as plt

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
    storing_path = os.path.join(figure_path, 'results')
    if not os.path.exists(storing_path):
        os.makedirs(storing_path)
    plot_path = os.path.join(figure_path, 'plots')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    with open(os.path.join(storing_path, f"{prefix}_training_result.pkl"), "wb") as f:
        pickle.dump(training_log_dict, f)
    

