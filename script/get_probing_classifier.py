import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.logging import log_info
# use the whole dataset to compute linear classifier
from utils.settings import *
from utils.DistributeTasks import distribute_tasks
from mpi4py import MPI
from utils.ProcessProbData import *

def train_prob(comm, paths):
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()
    log_info(f"Process rank: {rank}, Size: {size}, Processor name: {name}")

    probing_task_initialization()
    task_queue = get_task_queue(paths)

    distribute_tasks(comm, task_queue, process_task, node_workers=5)

    if rank == 0:
        log_info("Rank 0 starting to merge results from all temporary files")

        merged_weight_vec = merge_probing_files()
    else:
        merged_weight_vec = None

    return merged_weight_vec

if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Check if OBSERVED_LINEAR_FILE exists
    if os.path.exists(OBSERVED_LINEAR_FILE):
        log_info(f"OBSERVED_LINEAR_FILE already exists at {OBSERVED_LINEAR_FILE}. Skipping training.")
        MPI.Finalize()
        exit(0)

    # lhs = find_concept_pkl(HS_DIR)
    lhs = [os.path.join(HS_DIR, f"{concept}.pkl") for concept in CONCEPTS]  
    # if DATASET == "openai":
    #     lhs = [os.path.join(HS_DIR, f"{concept}.pkl") for concept in CONCEPTS] 
    # elif DATASET == "goemo":
    #     # lhs = [os.path.join(HS_DIR, "emotion.pkl")] 
    #     lhs = [os.path.join(HS_DIR, "emotion.pkl")] 
    observed_weight_vec = train_prob(comm, lhs)

    MPI.Finalize()