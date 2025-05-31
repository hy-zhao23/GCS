import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.files import read_pkl, write_pkl
from utils.settings import SEED_FILE, OBSERVED_NUM, tmp_dir, DATASET_SIZE, SAMPLE_SIZE
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import random
from utils.logging import log_info

# generate {num} unique seeds for dataset split. If the file exists then reuse it.
def generate_unique_random_seeds(min_value=1, max_value=1000000):
    if os.path.exists(SEED_FILE):
        log_info(f'Seed file {SEED_FILE} exists!')
        return read_pkl(SEED_FILE)

    if OBSERVED_NUM > (max_value - min_value + 1):
        raise ValueError(f"Cannot generate {OBSERVED_NUM} unique integers in range {min_value} to {max_value}")
    
    unique_seeds = set()
    while len(unique_seeds) < OBSERVED_NUM:
        unique_seeds.add(random.randint(min_value, max_value))
    
    seeds = list(unique_seeds)
    write_pkl(seeds, SEED_FILE)
    return seeds

def generate_sample_indices(seed):
    rng = np.random.default_rng(seed)
    indices = set(rng.choice(SAMPLE_SIZE, DATASET_SIZE, replace=False))
    n_indices = set(range(SAMPLE_SIZE)) - indices
    return list(indices), list(n_indices) 

def get_all_indices(seeds):
    all_indices, all_n_indices = [], []
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(generate_sample_indices, seed) for seed in seeds[:OBSERVED_NUM]]
        for future in as_completed(futures):
            indices, n_indices = future.result()
            all_indices.append(indices)
            all_n_indices.append(n_indices)

    file = os.path.join(tmp_dir, f'{OBSERVED_NUM}-{DATASET_SIZE}-indices.pkl')
    write_pkl(all_indices, file)
    file = os.path.join(tmp_dir, f'{OBSERVED_NUM}-{DATASET_SIZE}-indices_opposite.pkl') 
    write_pkl(all_n_indices, file)


if __name__ == "__main__":
    seeds = generate_unique_random_seeds()
    get_all_indices(seeds)
