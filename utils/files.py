import os 
import sys
import pickle
import json
from utils.settings import CONCEPTS
from glob import glob
from utils.logging import log_info, log_warning, log_error

# Only find all related files under current folder
def find_all_pkl(path):
    file_type = '.pkl'
    files = glob(os.path.join(path, f'*{file_type}'))
    return files 

def get_sampled_sigma(file):
    try:
        sigma = int(file.split('/')[-1].split('-')[2])
        if not sigma:
            sys.stderr.write(f"Error: {file} is not named in the form of sampled files!\n")
            sys.stderr.flush()
        return sigma
    except Exception as e:
        sys.stderr.write(f"Error extracting sigma from {file}: {e}\n")
        sys.stderr.flush()
        return None


# Find all pkl files only related to concepts we are studying
def find_concept_pkl(path):
    files = find_all_pkl(path)
    filtered_files = []
    for t in CONCEPTS:
        t = f'{path}/{t}.pkl'
        if t in files:
            filtered_files.append(t)
        else:
            log_warning(f"{t} is missing!")
    
    log_info(f"Find {len(filtered_files)} concept-related .pkl files under {path}")
    return filtered_files


def preprocess_files(files):
    try:
        result = {}
        for file in files:
            with open(file, 'rb') as f:
                data = pickle.load(f)        
                for key in data.keys():
                    if key in result.keys():
                        result[key] = {**data[key], **result[key]}
                    else:
                        result[key] = data[key]
    except Exception as e:
        log_error(f"Error in preprocess_files: {e}")
        return None
    return result

# Get concept from files path
def get_concept(path):
    l = path.split('/')[-1]
    l = l.split('.')[0]
    return l

def write_pkl(data, filename, verbose=True):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        if verbose:
            log_info(f"Data successfully written to {filename}")
        return True
    except Exception as e:
        log_error(f"Unexpected error while writing to {filename}: {e}")
        return False

def read_pkl(filename):
    try:
        if os.path.getsize(filename) == 0:
            log_warning(f"{filename} is empty.")
            return None
        
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        log_warning(f"File {filename} not found.")
        return None
    except pickle.UnpicklingError as e:
        log_error(f"Error unpickling {filename}: {e}")
        return None
    except Exception as e:
        log_error(f"An unexpected error occurred while reading {filename}: {e}")
        return None


def read_json(filename):
    try:
        if os.path.getsize(filename) == 0:
            log_warning(f"{filename} is empty.")
            return None
        
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        log_warning(f"File {filename} does not exist.")
        return None
    except json.JSONDecodeError as e:
        log_error(f"Error decoding JSON in {filename}: {e}")
        return None
    except Exception as e:
        log_error(f"An unexpected error occurred while reading {filename}: {e}")
        return None
    
def write_json(data, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        log_info(f"Generated concepts have been successfully written to {filename}")
        return True
    except Exception as e:
        log_error(f"Unexpected error while writing to {filename}: {e}")
        return False
    
