from utils.settings import TMP_DATA_DIR, OPENAI_KEY, PARA_DIR, SETTING
import os
import copy
import pandas as pd
from utils.logging import log_info, log_error
import openai
import re


def preprocess_file(file):
    try:
        with open(file, 'r') as f:
            content = f.read()

        content = re.sub(r'"*\s*<s>\s*"*', '', content)
        content = re.sub(r'"*\s*</s>\s*"*', '', content)

        with open(file, 'w') as f:
            f.write(content)

    except Exception as e:
        log_error(f"An error occurred in preprocess_text: {e}")
        return None

def get_evaluation_prompt(manner, generated_sentence):
    try:
        opp = "joyful" if manner == "angry" else "angry"
        # prompt for gpt
        begin_prompt = f"Below is an instruction that describes a task. \n Write a response that appropriately completes the request.\r\n\r\n ### Instruction:\r\n"
        end_prompt = f"\r\n\r\n### Response:"

        emo_prompt = f"Rate the following text from 0 to 10, with 0 being not {opp} at all and 10 being completely {opp}. And review if the context is repetitive or chaotic from 0 to 10. Please answer in the following format: {opp} Score: x out of 10; Repetitive or chaotic: y out of 10. Text: {generated_sentence}"

        if manner == "angry" or "joyful":
            return begin_prompt + emo_prompt + end_prompt
        
        opp = "modern" if manner == "Shakespear" else "Shakespear"
        
        style_prompt = f"Review if the following text is written in the style of modern language or the style of Shakespear's language. With 0 being not {opp} at all and 10 being completely {opp}. And review if the context is repetitive or chaotic from 0 to 10. Please answer in the following format: {opp} Score: x out of 10; Repetitive or chaotic: y out of 10. Text: {generated_sentence}"

        if manner == "modern" or "Shakespear":
            return begin_prompt + style_prompt + end_prompt
    except Exception as e:
        log_error(f"An error occurred in get_evaluation_prompt: {e}")
        return None
    
def get_steered_text(manners):
    try:
        file = os.path.join(TMP_DATA_DIR, 'evaluation', f'{SETTING}', "generate_sentences.csv")
        preprocess_file(file)
        steered_text = pd.read_csv(file, delimiter=';')

        file = os.path.join(PARA_DIR, "score_steered_text.csv")
        new_steered_text = copy.deepcopy(steered_text)
        new_steered_text['hallucination'] = None 

        if os.path.exists(file):
            data = pd.read_csv(file, delimiter=';')
            if not data:
                new_steered_text = copy.deepcopy(data)
            

        # Filter steered_text to only keep rows where manner is 'angry'
        new_steered_text = new_steered_text[new_steered_text['manner'].isin(manners)]

        log_info(f"Loaded {len(new_steered_text)} steered texts for {manners}...")

        # Count the number of data points for each method
        method_counts = new_steered_text['steering_method'].value_counts()
        
        log_info("Data count for each method:")
        for method, count in method_counts.items():
            log_info(f"{method}: {count}")

        # Calculate the total number of data points
        total_count = len(new_steered_text)
        log_info(f"Total number of data points: {total_count}")

        return new_steered_text
    except Exception as e:
        log_error(f"An error occurred in get_steered_text: {e}")
        return None

# send prompt to gpt and get response
def prompt_gpt(manner, gen_sentence, method, coef):
    try:
        prompt = get_evaluation_prompt(manner, gen_sentence)
        client = openai.OpenAI(api_key = OPENAI_KEY)
        nli_p_response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="gpt-4o",
        )
        result = nli_p_response.choices[0].message.content
        manner_score, chaotic_score = extract(result)

        while not manner_score or not chaotic_score:
            result = prompt_gpt(manner, gen_sentence)
            manner_score, chaotic_score = extract(result)
        
        log_info(f"{method}-{coef}: {gen_sentence} got {manner_score} for the opposite of {manner}, {chaotic_score} for hallucination")
        return int(manner_score), int(chaotic_score)
    except Exception as e:
        log_error(f"An error occurred in prompt_gpt: {e}")
        return None, None

def extract(result):
    try:
        result = result.split(";")
        if len(result) == 2:
            manner_score = result[0].split(" out of")[0].strip()[-1]
            chaotic_score = result[1].split(" out of")[0].strip()[-1]
            # Check if manner_score and chaotic_score can be converted to integers between 0 and 10
            try:
                manner_score_int = int(manner_score)
                chaotic_score_int = int(chaotic_score)
                if not (0 <= manner_score_int <= 10 and 0 <= chaotic_score_int <= 10):
                    return None, None
            except ValueError:
                return None, None
            return manner_score, chaotic_score
        else:
            return None, None
    except Exception as e:
        log_error(f"An error occurred in extract: {e}")
        return None, None

def get_emotion(manner):
    if manner == "joyful":
        return "angry"
    elif manner == "angry":
        return "joyful"
    elif manner == "modern":
        return "Shakespear"
    elif manner == "Shakespear":
        return "modern"

def process_chunk(chunk):
    try:
        chunk_result = chunk.copy()
        # import pdb; pdb.set_trace()
        for index, row in chunk.iterrows():
            manner = row["manner"]
            emotion = get_emotion(manner)

            if pd.isna(chunk_result.loc[index, f'{emotion}']) or pd.isna(chunk_result.loc[index, 'hallucination']):
                manner_score, chaotic_score = prompt_gpt(manner, row["gen_text"], row["steering_method"], row["lambda"]) 
                chunk_result.loc[index, f'{emotion}'] = manner_score
                chunk_result.loc[index, 'hallucination'] = chaotic_score

        return chunk_result
    except Exception as e:
        log_error(f"An error occurred in process_chunk: {e}")
        return None