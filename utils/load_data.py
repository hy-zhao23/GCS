from utils.settings import TMP_DATA_DIR, SAMPLE_SIZE, OBSERVED_NUM, DATASET_SIZE
from utils.files import read_pkl
from utils.logging import log_info, log_error 
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Use loop here, it is much faster than threading!!!
def split_data(data):
    p_data, n_data = [], []

    for row in data:
        if row[-1] == 1:
            p_data.append(row)
        else:
            n_data.append(row)
    
    return p_data, n_data

def process_row(row, p_data, n_data):
    tmp_p = [p_data[k] for k in row]
    tmp_n = [n_data[k] for k in row]
    return tmp_p + tmp_n

def process_layer(l, layer_data, indices, file):
    p_data, n_data = split_data(layer_data)
    if not (len(p_data) >= SAMPLE_SIZE and len(n_data) >= SAMPLE_SIZE):
        log_info(f"Hidden states from {file} at {l}-th layer: Pos: {len(p_data)}, Neg: {len(n_data)}")
        return None
    
    p_data = p_data[:SAMPLE_SIZE]
    n_data = n_data[:SAMPLE_SIZE]
    
    result = []
    
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(process_row, row, p_data, n_data) for row in indices]
        for future in as_completed(futures):
            try:
                result.append(future.result())
            except Exception as exc:
                log_error(f'An error occurred: {exc}')
    
    return l, result


def load_layerwise_training_dataset(file: str, layer, opposite=False):
    indice_file = os.path.join(TMP_DATA_DIR, f"{OBSERVED_NUM}-{DATASET_SIZE}-indices.pkl")
    if opposite:
        indice_file = os.path.join(TMP_DATA_DIR, f"{OBSERVED_NUM}-{DATASET_SIZE}-indices_opposite.pkl")
    
    indices = read_pkl(indice_file)
    data = None

    if indices:
        layer_data = read_pkl(file)[layer][1]
       
        try:
            l, d = process_layer(layer, layer_data, indices, file)
            data = np.array(d)
        except Exception as exc:
            log_error(f'Layer {l} generated an exception: {exc}')

    return data 
