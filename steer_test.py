from utils.settings import *

import os
from utils.files import read_pkl
from model.model import project_model
import numpy as np
from utils.Steer import *
from utils.memory import clean_cache
from utils.logging import log_info


concept = CONCEPTS[0]
data = read_pkl(OBSERVED_LINEAR_FILE)[concept]
# layer 30
vecs = {i:data[i][0] for i in range(1, 31)}

prompts = [(f"Write an tweet-like review from users about AirPods Pro in an angry manner. The angry review should be within 10 words.", 
                    f"Below is an instruction that describes a task. \n"
                    f"Write a response that appropriately completes the request.\r\n\r\n"
                    f"### Instruction:\r\nWrite an tweet-like review from users about AirPods Pro in an angry manner. The angry review should be within 10 words.\r\n\r\n### Response:")]

tokenizer = project_model.get_tokenizer()
device_map = project_model.get_device_map()

clean_cache()
left = 0.15; right = 0.20
csv_dump = [['steering_method', 'lambda', 'manner', 'prompt', 'gen_text', 'joyful', 'angry']]

log_info(f"Steering layers: {STEER_LAYERS}")
log_info(f"################### Original Steering ###################")
# log_info(f"################### Test Steering ###################")

for coef in np.linspace(left, right, 50):
    inf_model = get_steered_model(project_model, STEER_LAYERS, vecs, coef)                        
    generate_sentence(inf_model, 'angry', coef, 'random', prompts, tokenizer, device_map, csv_dump)

    del inf_model
    clean_cache()