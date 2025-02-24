from utils.files import write_pkl
from utils.settings import RAW_DIR
import os

def goemo_get_only_ekman(df):
    """
    For GoEmotions we only want the 6 base emotions and only samples that can be unambiguously
    assigned to a single base emotion, even if they have multiple labels overall.
    The function returns the index of the base emotion in the base_emotions list.
    """
    base_emotions = ['joy', 'anger']
    dataset_emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
        'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
        'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
        'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
        'remorse', 'sadness', 'surprise', 'neutral'
    ]
    base_emotion_indices = [dataset_emotions.index(emotion) for emotion in base_emotions]
    
    def get_base_emotion(labels):
        base_labels = [dataset_emotions[label] for label in labels if label in base_emotion_indices]
        return base_labels[0] if len(base_labels) == 1 else None

    filtered_df = df[df['labels'].apply(lambda x: sum(1 for l in x if l in base_emotion_indices) == 1)].copy()
    filtered_df['base_emotion'] = filtered_df['labels'].apply(get_base_emotion)
    
    temp_dict = {
        'text': filtered_df['text'].tolist(),
        'index': filtered_df.index.tolist(),
        'labels': filtered_df['base_emotion'].tolist()
    }

    raw_file = os.path.join(RAW_DIR, "goemo_joy_anger.pkl")
    write_pkl(temp_dict, raw_file)
    
    print("The GoEmo dataset has been filtered to contain samples with a single base emotion!")
    temp_dict = {
        'emotion': {
            'positive': filtered_df[filtered_df['base_emotion'] == 'joy']['text'].tolist(),
            'negative': filtered_df[filtered_df['base_emotion'] == 'anger']['text'].tolist(),
        } 
    }
    
    return temp_dict