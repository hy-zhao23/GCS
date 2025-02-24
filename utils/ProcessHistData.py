import sys
import os
from utils.settings import LAYERS, CONCEPTS, TMP_DATA_DIR, MODEL_SHORT
from utils.files import write_pkl
from utils.ProcessRankLog import new_layer_dict
import numpy as np
from queue import Queue
from utils.logging import log_info, log_warning, log_error
import copy
from utils.plots import plot_concept_accuracies

# observed, sampled, o-s,
def average_sim(sims: set):
    if not sims or len(sims) != 3:
        log_error(f"Invalid sims data: {sims}.")
        return None    
    
    if any(sim is None for sim in sims):
        log_error(f"One or more elements in sims is None: {sims}")
        return None
    try:
        return [np.mean(sim) for sim in sims]
    except Exception as e:
        log_error(f"Error calculating average sim: {str(e)}")
        return None
    
def average_vector_similarity(combined_data, avg_sim):
    for l in LAYERS:
        for c in CONCEPTS:
            avg_sim[c][l] = average_sim(combined_data[c][l])

# Make sure read data is not empty
# change combined_data to layerwise
def get_layerwise_combined_data(combined_data):
    file = os.path.join(TMP_DATA_DIR, f'{MODEL_SHORT}-restructured_hist_data.pkl')
    if os.path.exists(file):
        return 
    
    restructured_data = new_layer_dict()
    for l in LAYERS:
        restructured_data[l] = []
        for c in CONCEPTS:
            if c in combined_data and l in combined_data[c]:
                restructured_data[l].append(copy.deepcopy(combined_data[c][l]))
            else:
                # If data for this concept and layer is missing, append None or a placeholder
                log_warning(f"{c}-{l} in combined file is empty!")
                restructured_data[l].append((None, None, None))

    write_pkl(restructured_data, file)
    log_info("Rank 0: Data restructuring completed")

def get_one_hist_tasks(task_queue):
    file = os.path.join(TMP_DATA_DIR, f'{MODEL_SHORT}-restructured_hist_data.pkl')
    # Distribute plot tasks to different nodes according to layer
    for layer in LAYERS:
        task_queue.put((layer, file))

# divide task into layer level, get parameter for each task
def get_file_hist_tasks():
    task_queue = Queue()
    for c in CONCEPTS:
        for l in LAYERS:
            task_queue.put((c, l))
    return task_queue

def plot_avg_sim(avg_sim):
    names = ["Observed", "Sampled", "Observed-Sampled"]
    desc = "Averaged similarities"
    x_label = 'Layers'
    y_label = 'Average cosine similarity'
    log_info(f"Plot average similarity of data from histogram...")
    plot_concept_accuracies(avg_sim, desc, names, x_label, y_label)