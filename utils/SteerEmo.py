import numpy as np
from utils.logging import log_info
from model.model import project_model
from utils.Steer import *
from utils.settings import STEER_LAYERS, CONCEPTS
from utils.memory import clean_cache
from utils.DistributeGPU import get_dist_info

def get_evaluation_prompts(evaluation_prompts, manner):
    try:
        prompts = []
        if CONCEPTS[0] == 'emotion':
            for num_sentence, sentence in enumerate(evaluation_prompts):
                sentence = f"Write an review in an {manner} manner for the following movie <{sentence}>. The {manner} review should be within 10 words."
                prompts.append((sentence,
                    f"Below is an instruction that describes a task. \n"
                    f"Write a response that appropriately completes the request.\r\n\r\n"
                    f"### Instruction:\r\n{sentence}\r\n\r\n### Response:"))
        if CONCEPTS[0] == 'Joyful tweets': 
            prompts = [(f"Write an tweet-like review from users about AirPods Pro in an {manner} manner. The {manner} review should be within 10 words.", 
                    f"Below is an instruction that describes a task. \n"
                    f"Write a response that appropriately completes the request.\r\n\r\n"
                    f"### Instruction:\r\nWrite an tweet-like review from users about AirPods Pro in an {manner} manner. The {manner} review should be within 10 words.\r\n\r\n### Response:")]
        return prompts
    except Exception as e:
        log_info(f"Error in get_evaluation_prompts: {e}")
        raise

# get the lambda steering strength for the given manner, angry with positive direction, joyful with negative direction
def get_lambda_area(manner):
    if manner == 'angry':
        return 0.01, 0.10
    else:
        return -0.085, -0.04


def generate_all_prompts_all_manners(evaluation_prompts, manners):
    """Evaluate all prompts in the specified emotional manners for all lambda steering strengths."""

    dist_info = get_dist_info()
    rank = dist_info['rank']
    world_size = dist_info['world_size']

    try:
        tokenizer = project_model.get_tokenizer() 
        device_map = project_model.get_device_map()
        all_vectors = get_all_vectors()
        log_info(f"Loading all vectors...")
    except Exception as e:
        log_info(f"Error in loading all vectors in generate_all_prompts_all_manners: {e}")
        raise
    
    try:
        for c in CONCEPTS:
            file = os.path.join(TMP_DATA_DIR, f"evaluation/{SETTING}/generate_sentences_{c}_rank{rank}_completed.pkl")
            if not os.path.exists(file):
                # iterate over all manners
                local_vectors = dict(list(all_vectors.items())[rank::world_size])
                csv_dump = [['steering_method', 'lambda', 'manner', 'prompt', 'gen_text', 'joyful', 'angry']]

                # iterate over all methods
                for method, v in local_vectors.items():
                    # iterate over all manners, including angry and joyful
                    for manner in manners:
                        left, right = get_lambda_area(manner)
                        left = 0.06; right = 0.09
                        for coef in np.linspace(left, right, 100):
                            # get joy output?
                            prompts = get_evaluation_prompts(evaluation_prompts, manner)

                            inf_model = get_steered_model(project_model, STEER_LAYERS, v[c], coef)                        
                            generate_sentence(inf_model, manner, coef, method, prompts, tokenizer, device_map, csv_dump)

                            del inf_model
                            clean_cache()

                write_csv(csv_dump, rank, c)

            if rank == 0:
                combine_rank_csv(world_size, c)

    except Exception as e:
        log_info(f"Error in generate_all_prompts_all_manners: {e}")
        raise
        
    