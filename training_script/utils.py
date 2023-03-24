import logging
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn.functional as F

def compute_metrics(targets, preds):
    targets = targets.cpu().detach()
    preds = preds.cpu().detach()
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='weighted')
    return {'acc': acc, 'f1': f1, 'preds': preds, 'targets':targets}

def focal_loss(gamma=2):
    def compute_loss(preds, targets):
        ce_loss = F.cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-ce_loss) 
        return ((1-pt)**gamma * ce_loss).mean()
    return compute_loss

def train(data, model, optimizer, emo_loss_fn, stance_loss_fn):
    model.train()
    comments, emotions, stances = data
    emo_outputs, stance_outputs = model(comments)
    emo_loss = emo_loss_fn(emo_outputs, emotions)
    stance_loss = stance_loss_fn(stance_outputs, stances)
    loss = emo_loss + stance_loss
    model.zero_grad()
    loss.backward()
    optimizer.step()

    emo_preds = emo_outputs.argmax(-1)
    stance_preds = stance_outputs.argmax(-1)
    emo_metrics = compute_metrics(emotions, emo_preds)
    stance_metrics = compute_metrics(stances, stance_preds)
    return loss, emo_metrics, stance_metrics

@torch.no_grad()
def validate(data, model, emo_loss_fn, stance_loss_fn):
    model.eval()
    comments, emotions, stances = data
    emo_outputs, stance_outputs = model(comments)
    emo_loss = emo_loss_fn(emo_outputs, emotions)
    stance_loss = stance_loss_fn(stance_outputs, stances)
    loss = emo_loss + stance_loss

    emo_preds = emo_outputs.argmax(-1)
    stance_preds = stance_outputs.argmax(-1)
    emo_metrics = compute_metrics(emotions, emo_preds)
    stance_metrics = compute_metrics(stances, stance_preds)
    return loss, emo_metrics, stance_metrics

def init_logger(filename, verbose=True):
    logging.basicConfig(filename=filename,
                        format='%(asctime)s: %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        filemode='w')
    logger = logging.getLogger()

    def log(msg):
        if verbose:
            print(msg)
        logger.info(msg)
        
    return log