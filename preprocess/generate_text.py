#!/usr/bin/env python
# coding: utf-8
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import re
import openai
from dotenv import load_dotenv
from utils.files import read_pkl, write_pkl, read_json, write_json 
from utils.settings import TEXT_DIR, C_FILE, OPENAI_KEY
import pickle
import argparse
import threading
from queue import Queue
from utils.logging import log_info, log_error

load_dotenv()

# total = SAMPLE_SIZE
total = 5000

# read concepts that need generating sequence from json files
def read_concepts(file_path):
    if file_path.endswith(".json"):
        return read_json(file_path)
    if file_path.endswith(".pkl"):
        return read_pkl(file_path)

# parse sentence generated from gpt4
def extract(str):
    str = str.split('\n')
    pattern = r'^\d+\.\s*'
    l =[]
    for s in str:
        if re.search(pattern, s):
            s = re.sub(pattern, '', s)
            l.append(s)
    return l

# if the file exists, read existing data and calculate how many sentences are in need
def read_data(file: str, total: int, c: str):
    data = {
        c: {
            "positive":[],
            "negative":[]
        }
    }
    if not os.path.exists(file):
        return data, total, total
    else:
        p, n = 0, 0
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                if c in data.keys():
                    if "positive" in data[c].keys():
                        tmp = data[c]["positive"]
                        if isinstance(tmp, list):
                            p = len(tmp)
                    if "negative" in data[c].keys():
                        tmp = data[c]["negative"]
                        if isinstance(tmp, list):
                            n = len(tmp)
                log_info(f'Concept {c}, Pos: {p}, Neg: {n}')
        except EOFError:
            log_error("The pickle file might be corrupted, will be regenerated")
            
        return data, total - p, total -n

def get_prompt(c, prompts, positive=True):
    if positive:
        return prompts[c]['positive']
    else:
        return prompts[c]['negative']    

def process_prompt(c, num, nli_result, rest, prompts, positive=True):
    prompt = get_prompt(c, prompts, positive)
    client = openai.OpenAI(api_key = OPENAI_KEY)
    
    if num > 0: 
        if num < 5:
            num = 5

        nli_p_response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            # model="gpt-4-1106-preview",
            model="gpt-4o",
        )
        result = nli_p_response.choices[0].message.content
        res = extract(result)
        if len(res) >= 5:
            if positive:    
                nli_result[c]["positive"].extend(res)
            else:
                nli_result[c]["negative"].extend(res)
            rest -= len(res)
    
    return nli_result, rest

# Generate data with openai API
def generate_sample(k, c):

    file = f'{TEXT_DIR}/{c}.pkl'

    # Test if the file exits. If so, how many samples are there
    nli_result, rest_p, rest_n = read_data(file, total, c)
    if rest_p == 0 and rest_n == 0:
        log_info(f'Concept {c} is completed!')
        return
    
    prompts = read_json(f'preprocess/prompt.json')

    while(rest_p > 0 or rest_n > 0):
        # Generate 50 pos/neg samples each time
        log_info(f'\tThere are {rest_p} positive samples and {rest_n} negative samples left for concept {c};')
        num_p = 100 if rest_p >= 100 else rest_p
        num_n = 100 if rest_n >= 100 else rest_n

        # Create a chat completion request for NLI, and make sure generate more than 5 samples
        nli_result, rest_p = process_prompt(c, num_p, nli_result, rest_p, prompts, positive=True)
        nli_result, rest_n = process_prompt(c, num_n, nli_result, rest_n, prompts, positive=False)
    
        write_pkl(nli_result, file)
        
    if rest_p == 0 and rest_n == 0:
        log_info(f'Concept {c} is completed!')

def worker(queue):
    while True:
        task = queue.get()
        try:
            if task is None:
                break
            k, v = task
            generate_sample(k, v)
        finally:
            queue.task_done()

def generate_contexts_threaded(data, num_threads):
    queue = Queue()

    for key, value in data.items():
        if isinstance(value, list):
            for item in value:
                queue.put((key, item))
        else:
            queue.put((key, value))

    # Start worker threads
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(queue,))
        t.daemon = True  # Set threads as daemon
        t.start()
        threads.append(t)

    # import pdb; pdb.set_trace()
    # Add items to the queue
    

    # Add None to the queue to signal threads to exit
    for _ in range(num_threads):
        queue.put(None)

    # Wait for all tasks to be completed
    queue.join()

    # No need to explicitly wait for threads to finish
    log_info("All tasks completed.")

def main(args):
    # Define the concepts for the NLI task
    if args.file:
        concepts = read_concepts(C_FILE)
    else:
        concepts = {"Moives":["Action","Comedy", "Horror", "Science Fiction", "Animation"],
                "Mammals":["Primates", "Chiroptera", "Cetacea", "Lagomorpha", "Marsupialia"],
                "Natural Event":["Earthquake", "Solar eclipse", "Storm surge", "Pandemic"],
                "Sports Event":["Cycling competition", "Footbal match", "Tennis tournament", "Motor race"],
                "Biomolecule":["Enzyme", "Gene", "Hormone", "Protein"],
                "Chemical":["Chemical compound", "Chemical elements", "Vaccine", "Mineral"],
                "Natural Place":["Planet", "Glacier", "Beach", "Mountain"],
                "Populated Place":["Town", "Island", "City district", "village"],
                "Animal":["Bird", "Fish", "Insect", "Cat"],
                "Plant":["Conifer", "Fern", "Moss", "Flower"]
                }
        write_json(concepts, C_FILE)

    concepts = {'':['Joyful tweets']}
    
    num_threads = int(args.nthreads) if args.nthreads else 20
    generate_contexts_threaded(concepts, num_threads)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate NLI samples')
    parser.add_argument('--file', type=bool, default=True, help='Path to the file containing concepts')
    parser.add_argument('--nthreads', type=str, help='Number of thread, default 10')
    args = parser.parse_args()
    main(args)
