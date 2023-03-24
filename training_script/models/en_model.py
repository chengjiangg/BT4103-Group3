
from torch import nn
from transformers import AutoModel

class EN_DoubleClassifier(nn.Module):
    def __init__(self, model_ckpt, 
                 emotion_nlabels=1, 
                 stance_nlabels=1, 
                 tokenizer_size=30523,
                 dropout_prob=0.35):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_ckpt)
        self.encoder.resize_token_embeddings(tokenizer_size)
        encoder_config = self.encoder.config
        self.emotion_classifier = nn.Sequential(
            nn.BatchNorm1d(encoder_config.hidden_size),
            nn.Dropout(dropout_prob),
            nn.Linear(encoder_config.hidden_size, emotion_nlabels)
        ) 
        self.stance_classifier = nn.Sequential(
            nn.BatchNorm1d(encoder_config.hidden_size),
            nn.Dropout(dropout_prob),
            nn.Linear(encoder_config.hidden_size, stance_nlabels)
        ) 

        self.emotion_classifier.apply(self.init_weights)
        self.stance_classifier.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)

    def get_summary(self):
        print(self)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        x = self.encoder(**x)
        x = x.last_hidden_state[:, 0] # [cls] emb
        emotion_outputs = self.emotion_classifier(x)
        stance_outputs = self.stance_classifier(x)
        return emotion_outputs, stance_outputs