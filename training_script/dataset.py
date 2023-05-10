import re
import pandas as pd
from random import randint

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EN_SocialMediaDS(Dataset):
    def __init__(self, data_path, 
                 sheet_name, 
                 model_ckpt, 
                 max_token_length=50):
        super().__init__()
        self.max_token_length = max_token_length

        # label encodings
        self.emotion_labels = [
            'ANGER',
            'DISGUST',
            'FEAR',
            'JOY',
            'NEUTRAL',
            'SADNESS',
            'SURPRISE'
        ]
        self.stance_labels = [
            'AGAINST',
            'FAVOR',
            'NONE'
        ]

        # load excel
        self.data = pd.read_excel(data_path, sheet_name=sheet_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[ENTITY]"]})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx, :]

        entity = row['entity']
        entity = entity.strip()

        comment = row['text']
        comment = comment.replace(entity, "[ENTITY]")
        comment = comment.split()

        # long comments
        if len(comment) > self.max_token_length:
            # get index of entity in comment
            # some entity is companied with puncatuations i.e. boarders,
            for i, word in enumerate(comment):
                if "[ENTITY]" in word:
                    entity_ind = i
                    break
            start = max(entity_ind - int(self.max_token_length/2), 0)
            end = start + self.max_token_length
            comment = comment[start:end]
            
        comment = ' '.join(comment)
        comment = comment.replace("\\", "")

        emotion = row['emotion']
        emotion = self.emotion_labels.index(emotion)

        stance = row['stance']
        stance = self.stance_labels.index(stance)

        return f"{comment} [SEP] [ENTITY]", emotion, stance

    def choose(self):
        return self[randint(0, len(self)-1)]

    def get_tokenizer_size(self):
        return len(self.tokenizer)

    def decode(self, input_id):
        return self.tokenizer.decode(input_id)

    def collate_fn(self, data):
        comments, emotions, stances = zip(*data)
        comments = self.tokenizer(comments,
                                  padding=True,
                                  return_tensors='pt')
        comments = {k:v.to(device) for k, v in comments.items()}
        emotions = torch.tensor(emotions).long().to(device)
        stances = torch.tensor(stances).long().to(device)
        return comments, emotions, stances
    
class ZH_SocialMediaDS(Dataset):
    def __init__(self, data_path, 
                 sheet_name, 
                 model_ckpt, 
                 max_token_length=50):
        super().__init__()
        self.max_token_length = max_token_length

        # label encodings
        self.emotion_labels = [
            'ANGER',
            'DISGUST',
            'FEAR',
            'JOY',
            'NEUTRAL',
            'SADNESS',
            'SURPRISE'
        ]
        self.stance_labels = [
            'AGAINST',
            'FAVOR',
            'NONE'
        ]

        # load excel
        self.data = pd.read_excel(data_path, sheet_name=sheet_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[ENTITY]"]})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx, :]

        entity = row['entity']
        entity = entity.strip()

        comment = row['text']
        comment = comment.replace(entity, "[ENTITY]")

        # Define regular expressions for entity token, Chinese characters, English words, punctuation, emoticons, and emojis
        entity_pattern = r'\[ENTITY\]'
        chinese_pattern = r'[\u4e00-\u9fff]'
        punctuation_pattern = r'[^\w\s]'
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "]+", flags=re.UNICODE)
        
        # Split the sentence into tokens using regular expressions
        tokens = re.findall(rf'{entity_pattern}|{chinese_pattern}|{punctuation_pattern}|{emoji_pattern.pattern}', comment)

        # long comments
        if len(tokens) > self.max_token_length:
            # get index of entity in comment
            # some entity is companied with puncatuations i.e. boarders
            entity_ind = None
            for i, word in enumerate(tokens):
                if "[ENTITY]" in word:
                    entity_ind = i
                    break
            assert entity_ind == None, "{row['entity']} does not exist in {row['text']}"
            start = max(entity_ind - int(self.max_token_length/2), 0)
            end = start + self.max_token_length
            comment = tokens[start:end]

        comment = ''.join(comment)
        comment = comment.replace("\\", "")

        emotion = row['emotion']
        emotion = self.emotion_labels.index(emotion)

        stance = row['stance']
        stance = self.stance_labels.index(stance)

        return f"{comment} [SEP] [ENTITY]", emotion, stance

    def choose(self):
        return self[randint(0, len(self)-1)]

    def get_tokenizer_size(self):
        return len(self.tokenizer)

    def decode(self, input_id):
        return self.tokenizer.decode(input_id)

    def collate_fn(self, data):
        comments, emotions, stances = zip(*data)
        comments = self.tokenizer(comments,
                                  padding=True,
                                  return_tensors='pt')
        comments = {k:v.to(device) for k, v in comments.items()}
        emotions = torch.tensor(emotions).long().to(device)
        stances = torch.tensor(stances).long().to(device)
        return comments, emotions, stances
