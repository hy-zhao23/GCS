import os
import sys
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
from utils.settings import * 
from utils.files import write_pkl, read_pkl
import threading
from utils.logging import log_info
from concurrent.futures import ThreadPoolExecutor, as_completed

cl_pairs = [(c, l) for c in CONCEPTS for l in LAYERS]

# Get Gaussian distribution of concept vector and sample from it
def concept_distribution(concept_vec: dict):
    '''
    Compute mean and variance for each concept using threading with concept and layer pairs
    '''
    try:
        gau_mean_vec, gau_std_vec = {}, {}
        lock = threading.Lock()

        def process_concept_layer(c, l):
            try:
                data = concept_vec[c][l]
                mean = np.mean(data, axis=0).astype(float).tolist()
                std = np.std(data, axis=0).astype(float).tolist()
                with lock:
                    if c not in gau_mean_vec:
                        gau_mean_vec[c] = {}
                        gau_std_vec[c] = {}
                    gau_mean_vec[c][l] = [mean]
                    gau_std_vec[c][l] = [std]
            except Exception as e:
                log_info(f"Error in process_concept_layer: {e}")
                raise

        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(process_concept_layer, c, l) for c, l in cl_pairs]
            for future in as_completed(futures):
                future.result()

        write_pkl((gau_mean_vec, gau_std_vec), GAUSSIAN_LINEAR_FILE)
        return gau_mean_vec, gau_std_vec
    except Exception as e:
        log_info(f"Error in concept_distribution: {e}")
        raise

# Randomly sample vectors form 1-sigma area
def get_std_concept_vec(gau_mean, gau_std, desc, num):
    try:
        sampled_concept_vec = {}
        lock = threading.Lock()

        def process_concept_layer(c, l):
            mean = gau_mean[c][l][0]
            std_var = gau_std[c][l][0]
            dim = len(mean)
            samples = np.empty((SAMPLED_NUM, dim))
            for i in range(dim):
                scale = num * std_var[i]
                samples[:, i] = np.random.uniform(low=mean[i] - scale, high=mean[i] + scale, size=SAMPLED_NUM)

            prev_upper = [x + (num-1)*y for x, y in zip(mean, std_var)]
            prev_lower = [x - (num-1)*y for x, y in zip(mean, std_var)]
            current_upper = [x + num*y for x, y in zip(mean, std_var)]
            current_lower = [x - num*y for x, y in zip(mean, std_var)]

            clipped_samples = np.clip(samples, current_lower, current_upper)
            mask = np.all((clipped_samples >= prev_lower) & (clipped_samples <= prev_upper), axis=1)
        
            while np.any(mask):
                resampled = np.random.uniform(low=current_lower, high=current_upper, size=(np.sum(mask), dim))
                clipped_samples[mask] = resampled
                mask = np.all((clipped_samples >= prev_lower) & (clipped_samples <= prev_upper), axis=1)
        
            with lock:
                if c not in sampled_concept_vec:
                    sampled_concept_vec[c] = {}
                sampled_concept_vec[c][l] = clipped_samples
    
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(process_concept_layer, c, l) for c, l in cl_pairs]
            for future in as_completed(futures):
                future.result()

        file = os.path.join(PARA_DIR, f'sampled-{desc}-{num}-sigma-{SAMPLED_NUM}.pkl')
        write_pkl(sampled_concept_vec, file)
    except Exception as e:
        log_info(f"Error in get_std_concept_vec: {e}")
        raise



if __name__ == "__main__":
    log_info("Reading OBSERVED_LINEAR_FILE...")
    observed_weights = read_pkl(OBSERVED_LINEAR_FILE)

    log_info("Getting Gaussian mean and std vectors...")
    gau_mean_vec, gau_std_vec = concept_distribution(observed_weights)

    log_info("Generating sampled vectors...")
    for i in range(1, 6):  # Generate for 1 to 5 sigma
        log_info(f"Generating {i}-sigma vectors...")
        get_std_concept_vec(gau_mean_vec, gau_std_vec, 'linear', i)

    log_info("Process completed successfully.")

