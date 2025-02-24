import os
from utils.logging import log_info
from queue import Queue
from utils.settings import PARA_DIR, OBSERVED_NUM
from utils.files import read_pkl
from utils.plots import one_hist
from utils.algo import vectors_similarity
from utils.DistributeTasks import distribute_tasks
from utils.ProcessHistData import *
from utils.ProcessRankLog import *
import threading

# Initialize MPI
SHOW = False # Set to True if show each plot
# lock for heatmap drawing
plt_lock = threading.Lock()

observed_weights = None
sampled_weights = None

# put task process func here to avoid passing observed_weights and sampled_weights
def process_hist_task(c, l):
    o_sim = vectors_similarity(observed_weights[c][l])
    s_sim = vectors_similarity(sampled_weights[c][l])
    os_sim = vectors_similarity(observed_weights[c][l], sampled_weights[c][l])
    return (c, l, o_sim, s_sim, os_sim)

# compute accuracy hist and similarity heatmaps for concepts at different layers 
def process_file_hist(observed_file, sampled_file, sigma, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    combined_file = os.path.join(PARA_DIR, f"sim-hist-{sigma}-sigma-{OBSERVED_NUM}.pkl")
    
    # Check if combined file already exists
    if os.path.exists(combined_file):
        log_info(f"Combined file {combined_file} already exists. Skipping processing.")
        return

    # if not combined file not exists, load weights files
    global observed_weights, sampled_weights
    observed_weights = read_pkl(observed_file)
    sampled_weights = read_pkl(sampled_file)

    task_queue = get_file_hist_tasks()
    # label of files for generated hist-related files
    desc = 'sim-hist'
    # distributing tasks to all the nodes
    distribute_tasks(comm, task_queue, process_hist_task, True, desc, 50)
    log_info(f"Finished computing hist tasks!")

    if rank == 0:
        # combine rank files, write combined_file and detele the ranked files
        combine_hist_rank_data(combined_file, desc, size)
            
    # Synchronize all processes again
    comm.Barrier()


# Plot layerwise similarity histogram of observed, sampled, o-s vectors from every concepts
# Plot average similarity of observed, sampled, o-s vectors
def plot_hist(sigma, comm):
    rank = comm.Get_rank()

    avg_sim = new_concept_layer_dict()

    task_queue = Queue()

    s_file = os.path.join(PARA_DIR, f"sim-hist-{sigma}-sigma-{OBSERVED_NUM}.pkl")
    s_data = read_pkl(s_file)

    if rank == 0:
        if s_data:
            get_layerwise_combined_data(s_data)
            average_vector_similarity(s_data, avg_sim)
            plot_avg_sim(avg_sim)
        else:
            log_error(f"{s_file} is empty!")
        
        # Only need to compute task_queue on rank 0, as tasks are distributed on rank 0
        get_one_hist_tasks(task_queue)

    # plot one_hist only using one process each time considering the limitaion of plt
    distribute_tasks(comm, task_queue, one_hist, node_workers=10)