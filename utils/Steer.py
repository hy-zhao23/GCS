from utils.files import read_pkl, write_pkl
from utils.settings import *
import os
import numpy as np
import time
from utils.load_data import split_data
from utils.logging import log_info
from utils.ProcessProbData import process_layer_data, load_checkpoint
from utils.DistributeGPU import get_dist_info
from utils.steering_layer import SteeringLayer
import copy
from pathlib import Path
from utils.ProcessRankLog import wait_pending_files
import pandas as pd

def get_all_vectors():
    '''
        Get steering vectors for mean shift, n sigma, one linear
    '''
    try:
        all_vec = {}

        all_vec['mean shift'] = get_all_mean_vectors()
        
        sampled_vec = get_avg_linear_vector()
        for i in range(1, 6):
            all_vec[f'{i} sigma'] = sampled_vec[i] 
        
        all_vec['one linear'] = get_one_linear_vector()
        
        return all_vec

    except Exception as e:
        log_info(f"Error in get_all_vectors: {e}")
        raise

def get_avg_linear_vector():
    try:
        avg_linear_vec = {}
        for i in range(1,6):
            avg_linear_vec[i] = {}
            sampled_vec = read_pkl(os.path.join(PARA_DIR, f"sampled-linear-{i}-sigma-{SAMPLED_NUM}.pkl"))
            for c in CONCEPTS:
                avg_linear_vec[i][c] = {}
                for layer in LAYERS:
                    avg_linear_vec[i][c][layer] = np.mean(sampled_vec[c][layer], axis=0)
        return avg_linear_vec
    except Exception as e:
        log_info(f"Error in get_avg_linear_vector: {e}")
        raise

# get mean vectors for each concept
def mean_vector(concept):
    try:
        file = os.path.join(HS_DIR, f"{concept}.pkl")
        hs = read_pkl(file)
        m_vec = {}

        for l in LAYERS:
            m_vec[l] = []
            v = hs[l][1]
            p_data, n_data = split_data(v)
            diff = (np.mean(p_data, axis=0) - np.mean(n_data, axis=0))[:-1]
            m_vec[l].append(diff)

        return m_vec
    except Exception as e:
        log_info(f"Error in mean_vector: {e}")
        raise

# get mean vectors for all concepts
def get_all_mean_vectors():
    try:
        mean_vec = {}
        for c in CONCEPTS:
            mean_vec[c] = mean_vector(c)

        return mean_vec
    except Exception as e:
        log_info(f"Error in get_mean_vectors: {e}")
        raise

def combine_one_linear_checkpoint(one_linear_vec):
    '''
    Combine one linear checkpoint for all concepts
    meanwhile remove the checkpoint files and flag files
    return all the one linear vectors {concept: {layer: vector}}
    '''
    try:
        for c in CONCEPTS:
            one_linear_vec[c] = {}
            for layer in LAYERS:
                file = os.path.join(TMP_DATA_DIR, f"checkpoint-probing-{c}-{layer}-complete.log")
                while not os.path.exists(file):
                    time.sleep(10)

                checkpoint_file = os.path.join(TMP_DATA_DIR, f"checkpoint-probing-{c}-{layer}.pkl")
                one_linear_vec[c][layer] = read_pkl(checkpoint_file)['logistic']
                os.remove(file)
                os.remove(checkpoint_file)
        return one_linear_vec
    except Exception as e:
        log_info(f"Error in combine_one_linear_checkpoint: {e}")
        raise

def task_initiation(c):
    try:
        for layer in LAYERS:
            file = os.path.join(TMP_DATA_DIR, f"checkpoint-probing-{c}-{layer}-complete.log")
            if os.path.exists(file):
                os.remove(file)
    except Exception as e:
        log_info(f"Error in task_initiation: {e}")
        raise

def get_conept_one_linear_vec(c):
    '''
    Get one linear vector for a concept
    load checkpoint for each layer before processing, so we don't need to process these already processed
    '''
    try:
        data = read_pkl(os.path.join(HS_DIR, f"{c}.pkl"))

        for layer in LAYERS:
            # now each layer has only one data point, we need to put the hidden state in a list to use process_layer_data
            layer_data = [data[layer][1]]
        
            checkpoint_file = os.path.join(TMP_DATA_DIR, f"checkpoint-probing-{c}-{layer}.pkl")
            result = load_checkpoint(checkpoint_file, len(layer_data))
        
            log_info(f'Getting one linear vectors for concept {c} at layer {layer}. data length: {len(layer_data)}')
            process_layer_data(layer_data, result, c, layer, checkpoint_file)
    except Exception as e:
        log_info(f"Error in get_conept_one_linear_vec: {e}")
        raise

def check_file_exist():
    try:
        if os.path.exists(os.path.join(TMP_DATA_DIR, f"one-linear-vec-{DATASET}-complete.log")):
            if os.path.exists(os.path.join(PARA_DIR, f"one-linear-vec-{DATASET}.pkl")):
                return True
            else:
                os.remove(os.path.join(TMP_DATA_DIR, f"one-linear-vec-{DATASET}-complete.log"))
                return False
        else:
            return False
    except Exception as e:
        log_info(f"Error in check_file_exist: {e}")
        raise

def get_one_linear_vector():
    try:
        rank = get_dist_info()['rank']
        if rank == 0:
            one_linear_vec = {}
            if check_file_exist():
                one_linear_vec = read_pkl(os.path.join(PARA_DIR, f"one-linear-vec-{DATASET}.pkl"))
            else:
                for c in CONCEPTS:
                    get_conept_one_linear_vec(c)
                    one_linear_vec = combine_one_linear_checkpoint(one_linear_vec)

                write_pkl(one_linear_vec, os.path.join(PARA_DIR, f"one-linear-vec-{DATASET}.pkl"))
                write_pkl("complete", os.path.join(TMP_DATA_DIR, f"one-linear-vec-{DATASET}-complete.log"))
      
        else:
            if os.path.exists(os.path.join(TMP_DATA_DIR, f"one-linear-vec-{DATASET}-complete.log")):
                if os.path.exists(os.path.join(PARA_DIR, f"one-linear-vec-{DATASET}.pkl")):
                    return read_pkl(os.path.join(PARA_DIR, f"one-linear-vec-{DATASET}.pkl"))
                else:
                    os.remove(os.path.join(TMP_DATA_DIR, f"one-linear-vec-{DATASET}-complete.log"))
            
            while not os.path.exists(os.path.join(TMP_DATA_DIR, f"one-linear-vec-{DATASET}-complete.log")):
                time.sleep(10)
        return read_pkl(os.path.join(PARA_DIR, f"one-linear-vec-{DATASET}.pkl")) 
    
    except Exception as e:
        log_info(f"Error in get_one_linear_vector: {e}")
        raise

def load_sentences():
    """Load the questions"""
    # get correct path
    path_to_sentences = os.path.join(PROMPT_DIR, f"{DATASET}.txt")

    sentences = []
    with open(path_to_sentences,'r',encoding="utf-8") as tfile:
        sentences = tfile.readlines()
    # strip away \n
    stripped = [s.strip() for s in sentences if s.strip()]    
    
    return stripped

def generate_sentence(inf_model, manner, coef, method, prompts, tokenizer, device_map, csv_dump):
    
    """Generate sentences for the given prompts with modified models"""

    for sentence, prompt in prompts[:50]:
        input_tokens = tokenizer(prompt, return_tensors="pt").to(device_map['model.embed_tokens'])
        gen_tokens = inf_model.generate(input_tokens.input_ids, max_length=150)
        
        log_info(f"Steering \"{sentence}\" from {manner}, coefficient {coef}, method {method}")
        
        output = tokenizer.batch_decode(gen_tokens)[0].replace(prompt,'').replace('\n', ' ').replace(';','-')
        
        log_info(f"Generated sentence: {output}")
        log_info("==============================")

        csv_dump.append([method, coef, manner, sentence, output, None, None])
    
    return csv_dump

def get_steered_model(project_model, steer_layers, steering_vectors, b):
    original_model = copy.deepcopy(project_model.get_model())
    for layer_num in steer_layers:
        original_model.model.layers[layer_num] = SteeringLayer(original_model.model.layers[layer_num], layer_num, steering_vectors[layer_num], b)

    return original_model


def write_csv(csv_dump, rank, c):
    try:
        evaluation_path = os.path.join(TMP_DATA_DIR, f"evaluation/{SETTING}")
        Path(evaluation_path).mkdir(parents=True, exist_ok=True) 
        
        # Convert csv_dump to a pandas DataFrame
        df = pd.DataFrame(csv_dump[1:], columns=csv_dump[0])  # Skip the header row when creating DataFrame
        
        # Ensure all columns are of type string to avoid encoding issues
        df = df.astype(str)
        
        # Save to CSV using pandas to_csv method
        csv_file_path = os.path.join(evaluation_path, f"generate_sentences_{c}_rank{rank}.csv")
        df.to_csv(csv_file_path, sep=';', index=False, encoding='utf-8')
        
        # Log the successful save
        log_info(f"CSV data saved to {csv_file_path}")

        write_pkl("completed", os.path.join(evaluation_path, f"generate_sentences_{c}_rank{rank}_completed.pkl"))
    except Exception as e:
        log_info(f"Error in write_csv: {e}")
        raise

def get_rank_completed_file(world_size, c):
    try:
        files = []
        for i in range(world_size):
            files.append(os.path.join(TMP_DATA_DIR, f"evaluation/{SETTING}/generate_sentences_{c}_rank{i}_completed.pkl"))
        return files
    except Exception as e:
        log_info(f"Error in get_rank_file: {e}")
        raise

def combine_rank_csv(world_size, c):
    try:
        files = get_rank_completed_file(world_size, c)
        wait_pending_files(files)

        rank_files = []
        for i in range(world_size):
            rank_files.append(os.path.join(TMP_DATA_DIR, f"evaluation/{SETTING}/generate_sentences_{c}_rank{i}.csv"))
        
        all_data = None
        for f in rank_files:
            data = pd.read_csv(f, sep=';', encoding='utf-8')
            if all_data is None:
                all_data = copy.deepcopy(data)
            else:
                all_data = pd.concat([all_data, copy.deepcopy(data)], ignore_index=True)
        
        all_data.to_csv(os.path.join(TMP_DATA_DIR, f"evaluation/{SETTING}/generate_sentences_{c}.csv"), sep=';', index=False, encoding='utf-8')
        log_info(f"Combined CSV files into {os.path.join(TMP_DATA_DIR, f'evaluation/{SETTING}/generate_sentences_{c}.csv')}")
    except Exception as e:
        log_info(f"Error in combine_rank_csv: {e}")
        raise