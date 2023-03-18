import pandas as pd
from random import randint

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SocialMediaDS(Dataset):
    def __init__(self, dataset, model_ckpt, max_token_length=50):
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
        # self.data = pd.read_excel(data_path, sheet_name=sheet_name)
        self.data = dataset
        # self.data = pd.read_csv(data_path)
        # self.data = data.reset_index()

        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[ENTITY]"]})

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
        return row['text'], row['entity'], f"{comment} [SEP] [ENTITY]"

    def choose(self):
        return self[randint(0, len(self)-1)]

    def get_tokenizer_size(self):
        return len(self.tokenizer)

    def decode(self, input_id):
        return self.tokenizer.decode(input_id)

    def collate_fn(self, data):
        original_comments, entities, comments = zip(*data)
        comments = self.tokenizer(comments,
                                  padding=True,
                                  return_tensors='pt')
        comments = {k: v.to(device) for k, v in comments.items()}
        return original_comments, entities, comments
