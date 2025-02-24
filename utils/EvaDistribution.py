import numpy as np
from utils.settings import LAYERS, CONCEPTS, GAUSSIAN_LINEAR_FILE
from utils.files import read_pkl
from utils.logging import log_info, log_error
from utils.plots import distance_heatmap    


# Load the Gaussian parameters from the file
def load_gaussian_params():
    return read_pkl(GAUSSIAN_LINEAR_FILE)

# Compute the Wasserstein distance between two Gaussian distributions.
def compute_gaussian_distance(mean1, var1, mean2, var2):
    # Ensure inputs are numpy arrays
    mean1, var1, mean2, var2 = map(np.array, [mean1, var1, mean2, var2])
    
    # Compute the distance for each dimension
    squared_diff = (mean1 - mean2)**2
    var_sum = var1 + var2
    var_product = np.sqrt(var1 * var2)
    
    # Sum up the contributions from each dimension
    distance_squared = np.sum(squared_diff + var_sum - 2*var_product)
    
    # Return the square root of the sum
    return np.sqrt(distance_squared)

# Compute the distance matrix for a given layer
def distance_matrix(layer: int, mean_vec: dict, sqrt_var_vec: dict):
    
    num = len(CONCEPTS)
    dis_mat = np.zeros((num, num))

    for i in range(num):
        for j in range(i + 1, num):
            c1, c2 = CONCEPTS[i], CONCEPTS[j]
            m_1, m_2 = mean_vec[c1][layer], mean_vec[c2][layer]
            v_1, v_2 = sqrt_var_vec[c1][layer], sqrt_var_vec[c2][layer]
            
            distance = compute_gaussian_distance(m_1, v_1, m_2, v_2)
            dis_mat[i, j] = distance
            dis_mat[j, i] = distance

    return dis_mat

def compute_concept_distances(epsilon=1e-8):
    """
    Compute the distances between pairs of observed/sampled weights for each concept and layer.
    """
    try:
        mean_vec, var_vec = load_gaussian_params()

        distances = {l: None for l in LAYERS}

        # Pre-compute square roots of variances
        sqrt_var_vec = {c: {l: list(np.sqrt(np.array(var_vec[c][l]) + epsilon)) for l in LAYERS} for c in CONCEPTS}

        for layer in LAYERS:
            log_info(f"Computing distances for layer {layer}")

            distances[layer] = distance_matrix(layer, mean_vec, sqrt_var_vec)

        return distances
    except Exception as e:
        log_error(f"Error in computing concept distances: {e}")
        raise

def process_distribution_distance():
    desc = 'distance'
    distances = compute_concept_distances()
    distance_heatmap(distances, desc) 