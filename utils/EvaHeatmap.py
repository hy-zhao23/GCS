import os
from utils.plots import cos_sim_heatmap
import numpy as np
from utils.algo import concept_similarity
from utils.settings import *
from utils.ProcessRankLog import *
from utils.ProcessHeatmapData import *
from utils.DistributeTasks import distribute_tasks
import threading
from utils.logging import log_info

SHOW = False # Set to True if show each plot

r_o_weights = new_layer_dict()
r_s_weights = new_layer_dict()

# lock for heatmap drawing
plt_lock = threading.Lock()


def process_sim_mat(l):
    log_info(f"Start processing similarity matrix at layer {l}...")
    o_tmp = concept_similarity(np.array(r_o_weights[l]))
    s_tmp = concept_similarity(np.array(r_s_weights[l]))

    return l, o_tmp, s_tmp

# compute accuracy hist and similarity heatmaps for concepts at different layers 
def process_file_heatmap(comm, o_file, s_file, sigma):
    rank = comm.Get_rank()
    size = comm.Get_size()
    combined_file = os.path.join(PARA_DIR, f"sim-mat-{sigma}-sigma-{OBSERVED_NUM}.pkl")
    desc = 'sim-mat'
    
    # Check if combined file already exists, if combined exist, then the processed have been finished already
    if not os.path.exists(combined_file):
        log_info(f"Combined file {combined_file} does not exist. Start computing similarity matrices at all layers...")
        task_queue = get_heatmap_tasks()

        # Initialing weights variables
        get_layer_weights(r_o_weights, r_s_weights, o_file, s_file)

        distribute_tasks(comm, task_queue, process_sim_mat, True, desc, node_workers=5)
    else:
        log_info(f"Combined file {combined_file} already exists. Skipping processing.")

    if rank == 0:
        # combined sim mat for all layers and draw heatmap
        log_info("All similarity matrices at all layers are successfully computed! Start drawing heatmaps...")
        combined_data = combine_heatmap_rank_data(combined_file, desc, size)
        cos_sim_heatmap(combined_data, desc)
            
    # Synchronize all processes again
    comm.Barrier()
