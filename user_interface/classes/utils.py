import torch
from torch.utils.data import DataLoader

from .dataset import SocialMediaDS
from .model import DoubleClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#data_path = "./Data/Mappings/Generated/combine_data.xlsx"
#sheet_name = "Sheet1"


def get_dataloader(dataset, model_ckpt="zanelim/singbert", batch_size=32):
    ds = SocialMediaDS(dataset, model_ckpt)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=ds.collate_fn)
    return ds, dl


def get_model(model_weight_path, model_ckpt="zanelim/singbert"):
    model = DoubleClassifier(
        model_ckpt, emotion_nlabels=7, stance_nlabels=3).to(device)
    model_weight = torch.load(
        model_weight_path, map_location=torch.device(device))
    model.load_state_dict(model_weight)
    return model.to(device)


def map_emotion(emotion_labels, prediction):
    return emotion_labels[prediction]


def map_stance(stance_labels, prediction):
    return stance_labels[prediction]


def run_prediction(dataset, model_weight_path,
                   batch_size=32, model_ckpt="zanelim/singbert"):
    ds, dl = get_dataloader(dataset, batch_size=batch_size,
                            model_ckpt=model_ckpt)
    model = get_model(model_weight_path, model_ckpt=model_ckpt)
    model.eval()

    text, entities, emotions, stances = [], [], [], []
    with torch.no_grad():
        for data in dl:
            original_comments, target_entities, comments = data
            emo_outputs, stance_outputs = model(comments)

            text.extend(original_comments)
            entities.extend(target_entities)
            emotions.extend(emo_outputs.argmax(-1).detach().cpu().numpy())
            stances.extend(stance_outputs.argmax(-1).detach().cpu().numpy())
    emotions = [map_emotion(ds.emotion_labels, x) for x in emotions]
    stances = [map_stance(ds.stance_labels, x) for x in stances]

    return text, entities, emotions, stances
