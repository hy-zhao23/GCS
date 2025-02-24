import os
import sys
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.SteerEmo import *
from utils.Steer import load_sentences

# load all prompts we want to use for the evaluation
evaluation_prompts = load_sentences()

# manners = ['angry', 'joyful']
manners = ['angry']
generate_all_prompts_all_manners(evaluation_prompts, manners)
