import os
from utils.files import read_pkl, write_pkl
from utils.memory import clean_cache, print_memory_usage
from utils.settings import SAMPLE_SIZE, TMP_DATA_DIR
from utils.logging import log_info, log_error
from torch.utils.data import DataLoader, TensorDataset
from model.model import project_model
from utils.DistributeGPU import get_dist_info
import copy
import torch as t
def prepare_model_and_tokenizer():
    dist_info = get_dist_info()
    model, tokenizer = None, None
    try:
        with project_model as pm:
            model = pm.get_model()
            tokenizer = pm.get_tokenizer()

        if model is None or tokenizer is None:
            raise ValueError("Model or tokenizer is None after initialization")
        
        log_info(f"Model and tokenizer prepared for rank {dist_info['rank']} out of {dist_info['world_size']}")
        return model, tokenizer
    except Exception as e:
        log_error(f"Error in prepare_model_and_tokenizer: {e}")

def prepare_dataloader(batch_v, tokenizer, batch_size):
    try:
        tokens = tokenizer(batch_v, padding=True, return_tensors="pt")
        dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'])
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    except Exception as e:
        log_error(f"Error in prepare_dataloader: {e}")

def process_batch(model, input_ids, attention_mask):
    try:
        with t.no_grad():
            outputs = model(input_ids, attention_mask, output_hidden_states=True)
            print_memory_usage("After model forward pass")
        
        # The hidden states are already distributed across GPUs
        # We need to gather them on the CPU
        hidden_states = [hs.to('cpu') for hs in outputs.hidden_states]
        del outputs   

        return hidden_states
    except Exception as e:
        log_error(f"Error in process_batch: {e}")
        return None

def task_initializad(rank):
    flag_file = os.path.join(TMP_DATA_DIR, f"prob_rank_{rank}_complete")
    if os.path.exists(flag_file):
        os.remove(flag_file)


def task_completed(concept, rank):
    flag_file = os.path.join(TMP_DATA_DIR, f"prob_rank_{rank}_complete")
    with open(flag_file, 'w') as f:
        f.write(f"Rank {rank} completed processing for concept {concept}")
    log_info(f"All tasks for {concept} have been completed on rank {rank}")

def get_hs_distributed(v: list, concept: str, loop_size: int, batch_size: int, label: int):
    t.set_default_dtype(t.float32)
    clean_cache()
    try:
        model, tokenizer = prepare_model_and_tokenizer()

        text_states = []
        # be aware of loopsize, seems result is not appended to text_states
        for i in range(0, len(v), loop_size):
            end_i = min(i + loop_size, len(v))
            batch_v = v[i:end_i]
            loader = prepare_dataloader(batch_v, tokenizer, batch_size)

            for j, (input_ids, attention_mask) in enumerate(loader):
                input_ids = input_ids.to(model.device)
                attention_mask = attention_mask.to(model.device)
                hs = process_batch(model, input_ids, attention_mask)

                start_idx = j * batch_size
                end_idx = min((j + 1) * batch_size, len(batch_v))
                batch_text = batch_v[start_idx:end_idx]

                if hs is not None:
                    text_states = get_batch_hs_data(text_states, hs, batch_text, len(hs), label)
                else:
                    log_error(f"hs is None for batch {i}")
                del input_ids, attention_mask, hs
                clean_cache()

            del loader, batch_v  # Delete local variables

        save_hs(text_states, concept, label)
        del text_states  # Delete local variables
        clean_cache()

    except Exception as e:
        log_error(f"Error in get_hs_distributed: {e}")
    finally:
        del model, tokenizer
        clean_cache()
        
def get_batch_hs_data(text_states, hs, texts, layers, label):
    try:
        for i in range(layers):
            # send tensor to cpu
            last_hs = hs[i][:, -1, :].cpu()
            _size = last_hs.shape[0]
            labels = t.full((_size, 1), label, device=last_hs.device)
            last_hs = t.cat((last_hs, labels), dim=1)
            # if layers is not full, append new layer, this is for the first round to append hidden states
            if len(text_states) != layers:
                text_states.append([copy.deepcopy(texts), last_hs])
            # if layers is full, concat last_hs to existing layer
            else:
                text_states[i][0] += copy.deepcopy(texts)
                text_states[i][1] = t.cat((text_states[i][1], copy.deepcopy(last_hs)), dim=0)
        log_info(f"Hidden states for {len(texts)} texts have been processed. Total texts: {len(text_states[0][0])}")
        return text_states
    except Exception as e:
        log_error(f"Error in get_batch_hs_data: {e}")

def save_hs(text_states, concept, label):
    try:
        dist_info = get_dist_info()
        # if label == 1, save positive hidden states, usually positive hidden states are saved first
        f_path = os.path.join(TMP_DATA_DIR, f"{concept}-rank{dist_info['rank']}.pkl")
        if label == 1:
            write_pkl(text_states, f_path)
            log_info(f"{len(text_states[-1][0])} positive hidden states have been written to {f_path}!")
        else:
            # if label == 0, save negative hidden states, usually negative hidden states are saved second,
            # so we need to add positive hidden states to negative hidden states
            existing_data = read_pkl(f_path)
            layers = len(text_states)
            for i in range(layers):
                text_states[i][0] += copy.deepcopy(existing_data[i][0])
                text_states[i][1] = t.cat((text_states[i][1], copy.deepcopy(existing_data[i][1])), dim=0)  
            write_pkl(text_states, f_path)
            log_info(f"{len(text_states[-1][0])} hidden states have been written to {f_path}!")
    except Exception as e:
        log_error(f"Error in save_hs: {e}")

# check if file are stored with layer * 2 * SAMPLE_SIZE, if so, skip processing
def check_hs_exists(f_path: str):
    try:
        if os.path.exists(f_path) and os.path.getsize(f_path) > 0:
            data = read_pkl(f_path)
            if len(data) > 0 and len(data[0]) == 2 and len(data[0][0]) >= 2 * SAMPLE_SIZE:
                log_info(f"Hidden state file {f_path} already exists. Skipping processing.")
                return True
    except Exception as e:
        log_error(f"Error in check_hs_exists: {e}")
    return False


Share


You said:
生成10句话
ChatGPT said:
ChatGPT
好的，你需要生成什么样的10句话呢？是日常对话、幽默句子、励志名言，还是其他主题？









