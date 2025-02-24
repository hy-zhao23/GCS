import os
import sys
from dotenv import load_dotenv
import torch as t
import json

load_dotenv()

def parse_layers(layers_str):
    layers = []
    for item in layers_str.split(','):
        item = item.strip()
        if '-' in item:
            start, end = map(int, item.split('-'))
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(item))
    return sorted(set(layers)) 

def get_all_concepts(file):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            concepts = json.load(f)
            if concepts:
                c_values = [item for _, v in concepts.items() for item in v]
                return c_values
            else:
                sys.stderr.write(f"{file} is empty!")
                sys.stderr.flush()
                return
    except Exception as e:
        sys.stderr.write(f"Error reading {file}: {e}")
        sys.stderr.flush() 

# Model configuration
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE").strip())
DATASET_SIZE = int(os.getenv("DATASET_SIZE").strip())
OBSERVED_NUM = int(os.getenv("OBSERVED_NUM").strip())
SAMPLED_NUM = int(os.getenv("SAMPLED_NUM").strip())

DEVICES = 'cuda' if t.cuda.is_available() else 'cpu'

MODEL_NAME = os.getenv("MODEL")
DATASET = os.getenv("DATASET")
HF_KEY = os.getenv("HF_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

MODEL_SHORT = MODEL_NAME.split('/')[-1]
SETTING = os.path.join(MODEL_SHORT, DATASET)
LAYERS = parse_layers(os.getenv("LAYERS"))
STEER_LAYER_LLAMA7B = parse_layers(os.getenv("STEER_LAYER_LLAMA7B"))
STEER_LAYER_LLAMA13B = parse_layers(os.getenv("STEER_LAYER_LLAMA13B"))
STEER_LAYER_GEMMA7B = parse_layers(os.getenv("STEER_LAYER_GEMMA7B"))

# proj = os.getenv("PROJ")
proj_dir = "/project/md748/hz54/kav/KAV"
res_dir = os.path.join(proj_dir, "results")
# res_dir = os.path.join(proj_dir, "rebuttal")
# res_dir = proj_dir

TMP_DATA_DIR = os.path.join(res_dir, "tmp") # concept file path
SEED_DIR = os.path.join(res_dir, "seeds") # concept file path
WEIGHT_DIR = os.path.join(proj_dir, os.getenv("WEIGHT_DIR"), MODEL_SHORT)
HS_DIR = os.path.join(res_dir, "hidden_state", SETTING)
PROMPT_DIR = os.path.join(proj_dir, os.getenv("TEXT_DIR"), 'evaluation_prompts')
TEXT_DIR = os.path.join(proj_dir, os.getenv("TEXT_DIR"), DATASET)
RAW_DIR = os.path.join(proj_dir, os.getenv("RAW_DIR"))
C_FILE_DIR = os.path.join(proj_dir, os.getenv("C_FILE_DIR")) # concept file path
CHECKPOINT_DIR = os.path.join(res_dir, "checkpoints", SETTING)
PARA_DIR = os.path.join(res_dir, "para", SETTING)

# FIG_DIR = os.path.join(proj_dir, "fig", "rebuttal", SETTING)
FIG_DIR = os.path.join(proj_dir, "fig", SETTING)

os.makedirs(PARA_DIR, exist_ok=True)
os.makedirs(TMP_DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(C_FILE_DIR, exist_ok=True)
os.makedirs(SEED_DIR, exist_ok=True)
os.makedirs(WEIGHT_DIR, exist_ok=True)
os.makedirs(HS_DIR, exist_ok=True)
os.makedirs(PROMPT_DIR, exist_ok=True)


C_FILE = os.path.join(C_FILE_DIR, 'concept_gen.json')
SEED_FILE = os.path.join(SEED_DIR, f'{OBSERVED_NUM}.pkl')
OBSERVED_LINEAR_FILE = os.path.join(PARA_DIR, f'observed-linear-{OBSERVED_NUM}.pkl') 
SAMPLED_LINEAR_FILE = os.path.join(PARA_DIR, f'sampled-linear-1-sigma-{SAMPLED_NUM}.pkl') 
GAUSSIAN_LINEAR_FILE = os.path.join(PARA_DIR, f'gaussian-linear-{OBSERVED_NUM}.pkl') 

CONCEPTS = []

if DATASET == "openai":
    CONCEPTS = get_all_concepts(C_FILE)
    # CONCEPTS = ['Bird in Village']
elif DATASET == "goemo":
    # CONCEPTS = ['Joyful tweets']
    CONCEPTS = ['emotion']

STEER_LAYERS = None

if MODEL_NAME == "google/gemma-7b":
    STEER_LAYERS = STEER_LAYER_GEMMA7B
elif MODEL_NAME == "meta-llama/Llama-2-13b-chat-hf":
    STEER_LAYERS = STEER_LAYER_LLAMA13B
elif MODEL_NAME == "meta-llama/Llama-2-7b-chat-hf":
    STEER_LAYERS = STEER_LAYER_LLAMA7B