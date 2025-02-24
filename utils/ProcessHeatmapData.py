from utils.files import read_pkl
from utils.settings import *
import copy
from mpi4py import MPI
from queue import Queue
from utils.logging import log_info, log_error

def get_heatmap_tasks():
    task_queue = Queue()
    try:
        for l in LAYERS:
            task_queue.put((l,))
    except Exception as e:
        log_error(f"Error in get_heatmap_tasks: {str(e)}")

    return task_queue

# Load observed weights and sampled weights(n-sigma)
def load_all_weights(o_file, s_file):
    log_info("Loading weights...")
    
    o_weights = read_pkl(o_file)
    s_weights = read_pkl(s_file)

    if not o_weights:
        raise ValueError(f"{o_file} is empty.")
    if not s_weights:
        raise ValueError(f"{s_file} is empty.")
    
    return o_weights, s_weights

# restructure weights into layerwise
def get_layer_weights(r_o_weights, r_s_weights, o_file, s_file):

    o_weights, s_weights = load_all_weights(o_file, s_file)

    for l in LAYERS:
        r_o_weights[l], r_s_weights[l] = [], []
        for c in CONCEPTS:
            if c in o_weights.keys() and l in o_weights[c].keys():
                r_o_weights[l].append(copy.deepcopy(o_weights[c][l]))
            if c in s_weights.keys() and l in s_weights[c].keys():
                r_s_weights[l].append(copy.deepcopy(s_weights[c][l]))
            else:
                log_info(f"Lack weight vectors for concepts {c} at layer {l}")
                MPI.Finalize()


