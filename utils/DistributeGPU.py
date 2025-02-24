import os
import tempfile
import torch as t
import torch.distributed as dist
from utils.logging import log_info, log_error

# Global variable to store distributed information
dist_info = None

def setup_distributed():
    global dist_info
    try:
        rank = int(os.environ.get("SLURM_PROCID", "0"))
        world_size = int(os.environ.get("SLURM_NTASKS", "2"))
        gpus_per_node = t.cuda.device_count()
        local_rank = rank % gpus_per_node
        node_rank = rank // gpus_per_node

        # Create a temporary file for initialization
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        init_file = temp_file.name
        temp_file.close()

        if not dist.is_initialized():
            log_info("Initializing distributed environment...")
            dist.init_process_group(backend="nccl", init_method=f"file://{init_file}",
                                    world_size=world_size, rank=rank)
        else:
            log_info("Distributed environment already initialized.")
    
        t.cuda.set_device(local_rank)
    
        log_info(f"Distributed setup complete. Rank: {rank}, Local Rank: {local_rank}, "
                 f"Node Rank: {node_rank}, World Size: {world_size}, GPUs per Node: {gpus_per_node}")
    
        dist_info = {
            "local_rank": local_rank,
            "node_rank": node_rank,
            "rank": rank,
            "world_size": world_size
        }
    
    except Exception as e:
        log_error(f"Error in setup_distributed: {e}")

def get_dist_info():
    global dist_info
    if dist_info is None:
        raise RuntimeError("Distributed environment not initialized. Call setup_distributed() first.")
    return dist_info

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

setup_distributed()