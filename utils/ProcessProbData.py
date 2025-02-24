import os
from utils.files import read_pkl, write_pkl
from utils.logging import log_info, log_error
from utils.settings import OBSERVED_NUM, TMP_DATA_DIR, CONCEPTS, LAYERS, OBSERVED_LINEAR_FILE 
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.load_data import load_layerwise_training_dataset
from utils.files import get_concept
from utils.ProcessRankLog import wait_pending_files, delete_rank_files
from queue import Queue
import copy

def get_probing_files():
    try:
        files = []
        for concept in CONCEPTS:
            for layer in LAYERS:
                files.append(os.path.join(TMP_DATA_DIR, f"checkpoint-probing-{concept}-{layer}.pkl"))
        return files
    except Exception as e:
        log_error(f"Error getting probing files: {str(e)}")
        raise

def merge_probing_files():
    try:
        files = probing_checkpoint_complete()
        wait_pending_files(files)
        delete_rank_files(files)

        files = get_probing_files()
        merged_weight_vec = {}
        for path in files:
            s = path.split('/')[-1].split('.')[0].split('-')
            concept, layer = s[-2], int(s[-1])
            merged_weight_vec.setdefault(concept, {})[layer] = read_pkl(path)['logistic']

        write_pkl(merged_weight_vec, OBSERVED_LINEAR_FILE)
        log_info(f"Rank 0 finished merging and writing results")
        delete_rank_files(files)
        return merged_weight_vec
    except Exception as e:
        log_error(f"Error merging probing files: {str(e)}")
        raise

def probing_task_initialization():
    try:
        files = probing_checkpoint_complete()
        for file in files:
            if os.path.exists(file):
                os.remove(file)
    except Exception as e:
        log_error(f"Error initializing probing task: {str(e)}")
        raise

# get all probing files that need to be merged
def probing_checkpoint_complete():
    try:
        files = []
        for concept in CONCEPTS:
            for layer in LAYERS:
                files.append(os.path.join(TMP_DATA_DIR, f"checkpoint-probing-{concept}-{layer}-complete.log"))
        return files
    except Exception as e:
        log_error(f"Error getting probing files: {str(e)}")
        raise

# get all concepts and layers that need to be trained
def get_task_queue(paths):
    try:
        task_queue = Queue()
        for path in paths:
            for layer in LAYERS:
                task_queue.put((path, layer))
        return task_queue
    except Exception as e:
        log_error(f"Error getting task queue: {str(e)}")
        raise

# load checkpoint file
def load_checkpoint(checkpoint_file, num):
    if os.path.exists(checkpoint_file):
        return read_pkl(checkpoint_file)
    # 'processed' related to training vectors
    return {'logistic': [None] * num, 'processed': 0}

# save checkpoint file
def save_checkpoint(result, checkpoint_file):
    write_pkl(result, checkpoint_file)

# Check if the temporary file already exists and is not broken
def check_tmp_file(tmp_file, concept, layer):
    if os.path.exists(tmp_file):
        try:
            result = read_pkl(tmp_file)
            if result is not None and 'logistic' in result:
                if len(result['logistic']) == OBSERVED_NUM and all(v is not None for v in result['logistic']):
                    log_info(f"Task for concept {concept} at layer {layer} already completed.")
                    return True
        except Exception as e:
            log_error(f"Error checking temporary file for concept {concept} at layer {layer}: {e}")

    log_info(f"Temporary file for concept {concept} at layer {layer} is broken or does not exist. Reprocessing...")
    return False    

# train logistic regression model
def logistic_vector(data, i, concept, layer, verbose=False, max_iter=60, loops=10):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    features, targets = data[i][:,:-1], data[i][:,-1]
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.3, shuffle=True, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.66, shuffle=True, random_state=42)

    best_val_acc, best_train_acc, best_test_acc = 0, 0, 0
    n_iter = 0
    best_coef = None
    converged = None

    for _ in range(loops):
        lr_model = LogisticRegression(solver='lbfgs', max_iter=max_iter, fit_intercept=False)
        lr_model.fit(X_train, y_train)
        
        converged = lr_model.n_iter_ <= max_iter
        train_pred = lr_model.predict(X_train)
        test_pred = lr_model.predict(X_test)
        val_pred = lr_model.predict(X_val)
                
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)

        if converged or val_acc > 0.9:
            best_coef = copy.deepcopy(lr_model.coef_[0])
            best_train_acc = train_acc
            best_test_acc = test_acc
            n_iter = lr_model.n_iter_
            break
        elif val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_acc = train_acc
            best_test_acc = test_acc
            best_coef = lr_model.coef_[0]
            n_iter = lr_model.n_iter_
    if verbose or not converged: 
        log_info(f'Concept {concept} Layer {layer} {i}-th classifier. Training Accuracy: {best_train_acc:.4f}. Test Accuracy: {best_test_acc:.4f}. Number of iterations: {n_iter}. Converged: {"Yes" if converged else "No"}')

    return i, best_coef

# distribute training tasks at the concept-layer level to different threads, if the model is trained and saved in checkpoint file, skip it
def process_layer_data(data, result, concept, layer, checkpoint_file):
    try:
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = []
            for i in range(len(data)):
                if result['logistic'][i] is None:
                    futures.append(executor.submit(logistic_vector, data, i, concept, layer))
                else:
                    result['processed'] += 1

            # Wait for all futures to complete
            for future in as_completed(futures):
                i, coef = future.result()  # This ensures any exceptions are raised
                result['logistic'][i] = coef
                result['processed'] += 1
                if result['processed'] % len(data) == 0:
                    save_checkpoint(result, checkpoint_file)
                    write_pkl("complete", os.path.join(TMP_DATA_DIR, f"checkpoint-probing-{concept}-{layer}-complete.log"))
        
    except Exception as e:
        log_error(f"Error processing layer data: {str(e)}")
        save_checkpoint(result, checkpoint_file)
        raise

# process each concept and layer, if their is checkpoint file, read it
def process_task(file, layer):
    try:
        concept = get_concept(file)
        
        log_info(f"Loading training datasets for concept {concept} at layer {layer}...")
        data = load_layerwise_training_dataset(file, layer)
        checkpoint_file = os.path.join(TMP_DATA_DIR, f"checkpoint-probing-{concept}-{layer}.pkl")
        result = load_checkpoint(checkpoint_file, len(data))
    
        log_info(f'Getting vectors for concept {concept} at layer {layer}.')
        log_info(f"data length: {len(data)}")
        process_layer_data(data, result, concept, layer, checkpoint_file)

    except Exception as e:
        log_error(f"Error processing task: {str(e)}")
        raise