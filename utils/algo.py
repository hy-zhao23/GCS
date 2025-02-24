import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed

# Normalized matrix
def cos_sim_matrix(array):
    norm_array = array / np.linalg.norm(array, axis=1, keepdims=True)
    cosine_sim_matrix = np.dot(norm_array, norm_array.T)
    return cosine_sim_matrix

# the probability of data fall in 1-sigma area
def one_std_estimation(data, mean, std_var):
    mean = np.array(mean)
    std_var = np.array(std_var)
    return np.mean((data>=(mean - std_var)) & (data<=(mean + std_var)))  


# get all cosine similarity values, to draw hist
# compute the similarities of weights within a concept
def vectors_similarity(a: list, b=None):
    flattened = None
    if b is None:
        similarity_mat = cosine_similarity(a)
        up_tri_indices = np.triu_indices_from(similarity_mat, k=1) # Upper half of similarity matrix not including diagnoal
        flattened = similarity_mat[up_tri_indices]
    else:
        similarity_mat = cosine_similarity(a, b)
        flattened = similarity_mat.flatten()

    return flattened

# average all cosine similarity values, to draw similarity heatmap
def process_pair(args):
    i, j, mvs_i, mvs_j = args
    tmp = vectors_similarity(mvs_i, mvs_j)
    return i, j, sum(tmp) / len(tmp)

def concept_similarity(mvs: list):
    cnt = len(mvs)
    sim_mat = np.zeros((cnt, cnt))
    
    # Prepare arguments for multiprocessing
    args_list = [(i, j, mvs[i], mvs[j]) for i in range(cnt) for j in range(i, cnt)]
    
    # Use multiprocessing to compute similarities
    with ThreadPoolExecutor(max_workers=100) as executor:
        future_to_pair = {executor.submit(process_pair, arg): arg for arg in args_list}
        for future in as_completed(future_to_pair):
            i, j, sim = future.result()
            sim_mat[i][j] = sim_mat[j][i] = sim
    
    return sim_mat
