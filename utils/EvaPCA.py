from queue import Queue
from utils.logging import log_info, log_error
from utils.files import read_pkl
from utils.settings import CONCEPTS, LAYERS
from utils.DistributeTasks import distribute_tasks
import numpy as np
from sklearn.decomposition import PCA
from utils.plots import plot_layer_pca
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Lock

# Create a lock for matplotlib
plt_lock = Lock()

def get_pca_tasks(files):
    try:
        task_queue = Queue()
        for file in files:
            task_queue.put((file,))
        return task_queue
    except Exception as e:
        log_error(f"Error in getting pca tasks: {e}")

# get mean concept vectors with 1000 vectors for each concept
def get_mean_vectors(file):
    try:
        data = read_pkl(file)
        label = ""
        if "observed" in file:
            label = "observed"
        else:
            label = "sampled"

        mean_vectors = {}
        for c in CONCEPTS:
            for l in LAYERS:   
                concept_data = data[c][l]
                concept_mean = np.mean(concept_data, axis=0, keepdims=True)
                if l not in mean_vectors.keys():
                    mean_vectors[l] = (label, [concept_mean])
                else:
                    mean_vectors[l][1].append(concept_mean)
        return mean_vectors
    except Exception as e:
        log_error(f"Error in getting mean vectors: {e}")

def get_layer_pca(layer, layer_data, label):
    try:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(np.concatenate(layer_data, axis=0))

        # num = layer_data[0].shape[0]
        num = 16
        with plt_lock:
            plot_layer_pca(layer, reduced_data, num, label)
    except Exception as e:
        log_error(f"Error in getting layer pca: {e}")


def process_file_pca(file):
    try:
        log_info(f"Processing {file}...")
        mean_vectors = get_mean_vectors(file)

        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = []
            for layer, (label, vectors) in mean_vectors.items():
                futures.append(executor.submit(get_layer_pca, layer, vectors, label))
        
        for future in futures:
            future.result()

        log_info(f"PCA plots for all layers have been generated and saved in the 'pca_plots' directory.")
    except Exception as e:
        log_error(f"Error in processing file pca: {e}")
        
def process_pca(comm, files):
    try:
        rank = comm.Get_rank()
        log_info(f"Processing PCA for observed and sampled weights...")

        task_queue = Queue()
        if rank == 0:
            task_queue = get_pca_tasks(files)

        # make sure one rank only process one file here
        distribute_tasks(comm, task_queue, process_file_pca, node_workers=1)

        log_info(f"PCA plots for all layers have been generated and saved in the 'pca_plots' directory.")
    except Exception as e:
        log_error(f"Error in processing pca: {e}")