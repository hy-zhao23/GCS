import torch as t
import gc
from utils.logging import log_error, log_warning, log_info


# Empty cache in GPU to avoid OutofMemory
def clean_cache():
    try:
        t.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        log_error(f"Error in clean_cache: {e}")

def get_gpu_free_memory():
    if t.cuda.is_available():
        num_gb = 1024**3
        gpu_memory = {i: (t.cuda.get_device_properties(i).total_memory - t.cuda.memory_allocated(i)) / num_gb  # Convert to GB
                for i in range(t.cuda.device_count())}
        return gpu_memory

def print_memory_usage(step=''):
    gpu_memory = get_gpu_free_memory()

    for gpu_id, free_memory in gpu_memory.items():
        log_info(f"{step} - GPU {gpu_id} free memory: {free_memory:.2f} GB")