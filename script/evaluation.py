import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.settings import PARA_DIR
from utils.files import find_all_pkl, get_sampled_sigma
from mpi4py import MPI
from utils.logging import log_info, log_error

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initialize MPI
SHOW = False # Set to True if show each plot

def compute_draw_hist(o_file):
    try:
        s_file = os.path.join(PARA_DIR, 'sampled-linear-1-sigma-1000.pkl')
        log_info(f"Processing files {o_file} and {s_file} for histogram")
        
        sigma = get_sampled_sigma(s_file)
    
        from utils.EvaHist import process_file_hist, plot_hist
        
        log_info(f"Getting vector similarity for historgram...")
        process_file_hist(o_file, s_file, sigma, comm)

        log_info(f"Plotting historgram...")
        plot_hist(sigma, comm)

    except Exception as e:
        log_error(f"Error in computing histogram: {e}")

def compute_draw_heatmap(o_file):
    s_file = os.path.join(PARA_DIR, 'sampled-linear-1-sigma-1000.pkl')
    sigma = get_sampled_sigma(s_file)

    from utils.EvaHeatmap import process_file_heatmap

    log_info(f"Computing concept similarity and plotting heatmap...")
    process_file_heatmap(comm, o_file, s_file, sigma)

def compute_draw_accuracy(o_file, s_files):
    # Make sure dataset is generated first, run generate_training_dataset.py first
    from utils.CAccuracy import compare_accuracy
    log_info(f"Computing accuracies for observed and sampled weights...") 
    compare_accuracy(comm, o_file, s_files)

def compute_PCA(o_file):
    from utils.EvaPCA import process_pca
    log_info(f"Computing PCA for observed and sampled weights...")
    s_file = os.path.join(PARA_DIR, 'sampled-linear-1-sigma-1000.pkl')
    files = [o_file, s_file]
    process_pca(comm, files)

def main():
    log_info(f"Getting all observed and sampled weights files...")
    # number of observed weights can be different
    files = find_all_pkl(PARA_DIR)
    observed_files = [f for f in files if "observed" in f]
    sampled_files = [f for f in files if "sampled" in f]

    for f in observed_files:
        # number of observed_weights per concept per layer
        num = f.split('-')[-1].split('.')[0]

        # using linear vector to represent concepts
        med = 'linear'
        # matched sampled files according to observed file
        s_files = [f for f in sampled_files if med in f and num in f]
        
        log_info(f"Starting computation and drawing...")
        # compute_draw_hist(f)
        compute_draw_heatmap(f)
        # compute_draw_accuracy(f, s_files)
        # compute_PCA(f)

if __name__ == "__main__":
    main()
