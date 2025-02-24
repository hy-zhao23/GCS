import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.logging import log_info

from utils.settings import *
from utils.files import read_pkl, get_sampled_sigma
from utils.load_data import load_layerwise_training_dataset 
from utils.logging import log_info
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from utils.DistributeTasks import distribute_tasks
from utils.ProcessAccData import *
from utils.ProcessRankLog import *
from utils.plots import plot_concept_accuracies

SHOW = False # Set to True if show each plot
# LAYERS = [1,2]
# lock for heatmap drawing
plt_lock = threading.Lock()
all_weights = None

# Compute the average accuracy of all classifiers 
def layer_accuracy(c, l, weight_file, observed=True, inter=False):
    sigma = 0 if observed else get_sampled_sigma(weight_file)
    desc = f"{c}-{l} {'observed samples' if observed else f'{sigma}-sigma samples'}"
    accuracies = []
    
    log_info(f"Loading data and weights for {desc}...")
    
    hs_file = os.path.join(HS_DIR, f'{c}.pkl')
    data = load_layerwise_training_dataset(hs_file, l, observed) if observed else read_pkl(hs_file)[l][1]
    weights = all_weights[sigma][c][l]

    log_info(f"Computing accuracies for {desc}...")

    with ThreadPoolExecutor(max_workers=100) as executor:
        future_to_accuracy = {executor.submit(test_lr_accuracy, data, i, weights, observed, inter): i for i in range(len(weights))}
        for future in as_completed(future_to_accuracy):
            accuracies.append(future.result())

    log_info(f"Accuracy computation for {desc} is finished!")

    return sigma, c, l, np.mean(accuracies)

# compute the accuracy of classifiers on datasets
def compare_accuracy(comm, f:str, sigmas: list):
    rank = comm.Get_rank()
    size = comm.Get_size()
    combined_data = None
    combined_file = os.path.join(PARA_DIR, f"accuracy-{OBSERVED_NUM}.pkl")
    
    # if combined_file does not exist, compute accuracies and save it
    if not os.path.exists(combined_file):
        # weights initialization
        log_info(f"Initializing all observed and sampled weights...")
        global all_weights
        all_weights = get_all_weights()

        # task queue initialization 
        task_queue = get_acc_tasks(f, sigmas)

        # allocating tasks to different nodes
        desc = 'accuracy'
        distribute_tasks(comm, task_queue, layer_accuracy, True, desc, 6)

        # Combine results on rank 0
        if rank == 0:
            log_info("All accuracies have been computed! Now combining results")
            combined_data = combine_acc_rank_data(desc, size)   
        log_info(f"Accuray computation runs successfully!")
        
    else:
        log_info(f"Combined file {combined_file} already exists. Skipping processing.")

    if rank == 0:
        if not combined_data:
            combined_data = read_pkl(combined_file)
        label, x_label, y_label = "Averaged accuracy", 'Layer', "Average accuracy"
        val_name = ["Observed", r"1-$\sigma$", r"2-$\sigma$", r"3-$\sigma$", r"4-$\sigma$", r"5-$\sigma$"]
        plot_concept_accuracies(combined_data, label, val_name, x_label, y_label)
    # Synchronize all processes again
    comm.Barrier()