from utils.settings import *
from utils.files import read_pkl
from utils.logging import log_info, log_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from queue import Queue
import numpy as np

def get_all_weights():
    all_weights = [read_pkl(OBSERVED_LINEAR_FILE), 
            read_pkl(os.path.join(PARA_DIR, f'sampled-linear-1-sigma-1000.pkl')),
            read_pkl(os.path.join(PARA_DIR, f'sampled-linear-2-sigma-1000.pkl')),
            read_pkl(os.path.join(PARA_DIR, f'sampled-linear-3-sigma-1000.pkl')),
            read_pkl(os.path.join(PARA_DIR, f'sampled-linear-4-sigma-1000.pkl')),
            read_pkl(os.path.join(PARA_DIR, f'sampled-linear-5-sigma-1000.pkl'))
        ]
    log_info("Finished reading weights files!")
    return all_weights


def get_acc_tasks(f, sigmas):
    task_queue = Queue()
    for c in CONCEPTS:
        for l in LAYERS:
            task_queue.put((c, l, f, True))
            for s in sigmas:
                task_queue.put((c, l, s, False))

    return task_queue

def check_dim(X):
    if MODEL_SHORT == 'Llama-2-7b-chat-hf':
        return X.shape[1] != 4096
    elif MODEL_SHORT == 'Llama-2-13b-chat-hf':
        return X.shape[1] != 5120
    elif MODEL_SHORT == 'gemma-7b':
        return X.shape[1] != 3072

# sys.stdout.write(f"Computing accuracy for {desc}: {len(data)} samples...\n")
# sys.stdout.flush()
def test_lr_accuracy(data, i, weights, observed, inter=False):
    # Create the logistic regression model with fit_intercept=False
    try:
        d = np.array(data[i]) if observed else np.array(data)
        X = d[:, :-1]
        y = d[:, -1]
        model = LogisticRegression(fit_intercept=inter)
        model.coef_ = np.array(weights[i]).reshape(1, -1)
        model.classes_ = np.array([0, 1])

        model.intercept_ = np.zeros(1) if inter else np.array([0.0])

        if check_dim(X):
            raise ValueError(f"Unexpected number of features in X: {X.shape[1]}. Expected 4096")

        y_pred = model.predict(X)
        return accuracy_score(y, y_pred)
    except Exception as e:
        log_error(f"Error in test_logistic_regression_accuracy: {e}")
        return None