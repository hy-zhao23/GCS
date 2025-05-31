import os
import torch as t
from utils.settings import tmp_dir, LAYERS, PARA_DIR, OBSERVED_NUM, HS_DIR, CONCEPTS
from utils.files import read_pkl, write_pkl
from utils.logging import log_info, log_error, log_warning
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import copy

def wait_pending_files(pending_files):
    while True:
        if check_files_availability(pending_files) > 0:
            time.sleep(10)
        else:
            break

def new_concept_layer_dict():
    try:
        data = {}
        for c in CONCEPTS:
            data[c] = new_layer_dict()
        return data
    except Exception as error:
        log_error(f"Error generating concept-level dictionary: {error}")

def new_layer_dict():
    try:
        return {l: None for l in LAYERS}
    except Exception as error:
        log_error(f"Error generating layer-level dictionary: {error}") 

        
def get_used_ranks_files(desc, size):
    rank_files = []
    for i in range(1, size):
        file = os.path.join(tmp_dir, f"{desc}-rank{i}.pkl")
        rank_files.append(file)
    return rank_files
    
# check how many rank logs are still not ready
def check_files_availability(files):
    cnt = 0
    for file in files:
       if not os.path.exists(file): 
            cnt += 1    
    log_info(f"There are {cnt} files missing!")
    return cnt

def get_hist_rank_file_data(file, combined_data):
    data = read_pkl(file)
    if data:
        for c, l, o_sim, s_sim, os_sim in data:
            if o_sim is None or s_sim is None or os_sim is None:
                log_error(f"{c}-{l} sim has None!")
            else:
                combined_data[c][l] = (o_sim, s_sim, os_sim)
        log_info(f"Successfully read {file}")
    else:
        log_info(f"{file} is empty")

def delete_rank_files(files):
    try:
        for file in files:
            if os.path.exists(file):
                os.remove(file)
        log_info("Successfully deleted all files")
    except Exception as error:
        log_error(f"Error occurred while deleting rank files: {error}")

# combine data from rank file and write to the output file
def combine_hist_rank_data(output_file, desc, size):
    try:
        if os.path.exists(output_file):
            log_info(f"Combined file {output_file} already exists. Skipping combining.")
            return 

        pending_files = get_used_ranks_files(desc, size)
        # Wait util all files are ready
        wait_pending_files(pending_files)
        combined_data = new_concept_layer_dict()
        for f in pending_files:
            get_hist_rank_file_data(f, combined_data)
        write_pkl(combined_data, output_file)
        delete_rank_files(pending_files)
        log_info(f"Successfully combined hist rank data and saved to {output_file}")
        return combined_data
    except Exception as e:
        log_error(f"Error in combining hist rank data: {e}")
        raise

def get_heatmap_rank_file_data(file, combined_data):
    data = read_pkl(file)
    if data:
        for l, o_tmp, s_tmp in data:
            combined_data[l] = (o_tmp, s_tmp)
        log_info(f"Successfully read {file}")
    else:
        log_info(f"{file} is empty")

# combine data from rank file and write to the output file
def combine_heatmap_rank_data(output_file, desc, size):
    try:
        if os.path.exists(output_file):
            log_info(f"Combined file {output_file} already exists. Skipping combining.")
            return read_pkl(output_file)

        log_info(f"Combined file {output_file} does not exist. Start combining...")
        pending_files = get_used_ranks_files(desc, size)
        # Wait util all files are ready
        wait_pending_files(pending_files)
        combined_data = new_layer_dict()
        for f in pending_files:
            get_heatmap_rank_file_data(f, combined_data)
        write_pkl(combined_data, output_file)
        delete_rank_files(pending_files)
        log_info(f"Successfully combined heatmap rank data and saved to {output_file}")
        return combined_data
    except Exception as e:
        log_error(f"Error in combining heatmap rank data: {e}")
        raise

def assign_data(data_chunk, combined_data, lock):
    try:    
        for sigma, c, l, acc in data_chunk:
            with lock:
                combined_data[c][l][sigma] = acc
            if not acc:
                log_warning(f"{c}-{l}-{sigma} is {acc}")
    except Exception as e:
        log_error(f"Error in assigning data: {e}")
        raise

def process_file(file, combined_data, global_lock):
    try:
        log_info(f'Loading {file}...')
        data = read_pkl(file)
        if not data:
            log_warning(f"{file} is empty")
            return

        # Split data into chunks for threading
        chunk_size = 60  # Adjust based on your data and system
        data_chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    
        with ThreadPoolExecutor(max_workers=60) as executor:
            futures = [executor.submit(assign_data, chunk, combined_data, global_lock) 
                       for chunk in data_chunks]
            for future in as_completed(futures):
                future.result()  # This will raise an exception if the task failed
    
        log_info(f"Successfully read {file}")
    except Exception as e:
        log_error(f"Error in processing file: {e}")
        raise

def combine_acc_rank_data(desc, size):
    try:
        output_file = os.path.join(PARA_DIR, f"{desc}-{OBSERVED_NUM}.pkl")
        if os.path.exists(output_file):
            log_info(f"Combined file {output_file} already exists. Skipping combining.")
            return read_pkl(output_file)

        pending_files = get_used_ranks_files(desc, size)
        wait_pending_files(pending_files)

        combined_data = {}
        for c in CONCEPTS:
            combined_data[c] = {}
            for l in LAYERS:
                combined_data[c][l] = [None] * 6

        global_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(process_file, file, combined_data, global_lock) for file in pending_files]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    log_warning(f'A file generated an exception: {exc}')

        write_pkl(combined_data, output_file)
        # delete_rank_files(pending_files)
        log_info(f"Successfully combined accuracy rank data and saved to {output_file}")
        return combined_data
    except Exception as e:
        log_error(f"Error in combining accuracy rank data: {e}")
        raise

def wait_hs_task_completion(concept, world_size):
    try:
        log_info(f"Waiting for all ranks to complete processing for concept {concept}")
        all_ranks_complete = False
        while not all_ranks_complete:
            all_ranks_complete = True
            for rank in range(world_size):
                rank_flag_file = os.path.join(tmp_dir, f"{concept}_rank_{rank}_complete")
                if not os.path.exists(rank_flag_file):
                    all_ranks_complete = False
                    break
            if not all_ranks_complete:
                time.sleep(5)  # Wait for 5 seconds before checking again
        log_info(f"All ranks have completed their tasks for concept {concept}")
        
        # Clean up flag files
        for rank in range(world_size):
            flag_file = os.path.join(tmp_dir, f"{concept}_rank_{rank}_complete")
            if os.path.exists(flag_file):
                os.remove(flag_file)
        
    except Exception as e:
        log_error(f"Error while waiting for task completion: {e}")
        raise


def combine_hs_rank_data(concept, world_size):
    try:
        output_file = os.path.join(HS_DIR, f"{concept}.pkl")
        if os.path.exists(output_file):
            log_info(f"Combined file {output_file} already exists. Skipping combining.")
        files = [os.path.join(tmp_dir, f"{concept}-rank{rank}.pkl") for rank in range(world_size)]
        wait_hs_task_completion(concept, world_size)

        combined_data = []
        for rank in range(world_size):
            file = os.path.join(tmp_dir, f"{concept}-rank{rank}.pkl")
            if file in files:
                data = read_pkl(file)
                for i, d in enumerate(data):
                    if i >= len(combined_data):
                        combined_data.append(copy.deepcopy(d))
                    else:
                        combined_data[i][0].extend(copy.deepcopy(d[0]))
                        combined_data[i][1] = t.cat((combined_data[i][1], copy.deepcopy(d[1])), dim=0)
                    log_info(f"combined_data[{i}][0]: {len(combined_data[i][0])} \t combined_data[{i}][1]: {combined_data[i][1].shape}")
            else:
                log_error(f"Expected rank file not found: {file}")

        write_pkl(combined_data, output_file)
        
        delete_rank_files(files)
    except Exception as e:
        log_error(f"Error in combining HS rank data: {e}")
        raise

    
    