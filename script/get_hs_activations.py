import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.settings import TEXT_DIR, CONCEPTS, MODEL_SHORT, TMP_DATA_DIR, DATASET
from utils.files import find_concept_pkl, preprocess_files
from tqdm import tqdm
from utils.logging import log_info, log_error
from utils.ProcessHSData import *
from utils.ProcessRankLog import combine_hs_rank_data
from utils.DistributeGPU import *

def process_concept(c, samples, loop_size, batch_size):
    """
    For LLMs, we always assign one task to one node, so here we use rank and node_rank mistakenly.
    """
    dist_info = get_dist_info()
    try:
        if dist_info['rank'] == 0:
            log_info(f"Processing concept {c} on {dist_info['world_size']} total GPUs")
        
        task_initializad(dist_info['rank'])
        # Process positive samples
        positive_samples = samples.get("positive", [])
        log_info(f"Positive samples for {c}: {len(positive_samples)}")
        node_positive_samples = positive_samples[dist_info['rank']::dist_info['world_size']]
        get_hs_distributed(node_positive_samples, c, loop_size, batch_size, label=1)
            
        # Process negative samples
        negative_samples = samples.get("negative", [])
        log_info(f"Negative samples for {c}: {len(negative_samples)}")
        node_negative_samples = negative_samples[dist_info['rank']::dist_info['world_size']]
        get_hs_distributed(node_negative_samples, c, loop_size, batch_size, label=0)

        task_completed(c, dist_info['rank'])
        
        # Combine results (only on rank 0)
        if dist_info['rank'] == 0:
            # Implement logic to combine results from all nodes
            combine_hs_rank_data(c, dist_info['world_size'])
    
    except Exception as e:
        log_error(f"Error processing concept {c} on rank {dist_info['rank']}: {e}")


def run_processing(samples):
    try:
        loop_size, batch_size = 50, 10
        if MODEL_SHORT == "Llama-2-13b-chat-hf":
            log_info(f"Running with model {MODEL_SHORT}")
            loop_size, batch_size = 200, 100
    
        for c in tqdm(CONCEPTS):
            log_info(f"Getting hidden representation for concept {c}:")
        
            output = os.path.join(TMP_DATA_DIR, f'{c}.pkl')
            if check_hs_exists(output):
                continue
            process_concept(c, samples[c], loop_size, batch_size)
    except Exception as e:
        log_error(f"Error in run_processing: {e}")

def main():
    # setup_distributed()
    try:
        if DATASET == "openai":
            files = find_concept_pkl(TEXT_DIR)
            files = [item for item in files if "concept.pkl" not in item]
            samples = preprocess_files(files)
            run_processing(samples)
            log_info(f"All hidden states for {DATASET} have been successfully saved!")
        elif DATASET == "goemo":
            for c in CONCEPTS:
                file = os.path.join(TEXT_DIR, f"{c}.pkl")
                if not os.path.exists(file):
                    log_error(f"File {file} does not exist")
                    continue
                samples = read_pkl(file)
                log_info(f"Processing concept {c} with {file}")
                process_concept(c, samples[c], 200, 50)
            log_info(f"All hidden states for {DATASET} have been successfully saved!")
    except Exception as e:
        log_error(f"Error in main execution: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    main()