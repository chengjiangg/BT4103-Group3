import os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import optim
from torch.utils.data import DataLoader

from utils import *
from model import DoubleClassifier
from dataset import EN_SocialMediaDS, ZH_SocialMediaDS

device = 'cuda' if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser(description= "Multi-Lingual Text Classifier")

# training paths
parser.add_argument('--log_filename', type=str, default="training.log")
parser.add_argument('--excel_filename', type=str, default="")
parser.add_argument('--sheet_name', type=str, default="")
parser.add_argument('--saved_model_name', type=str, default="text_classifier")

# training settings
parser.add_argument('--verbose', action="store_true")
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--focal_loss_gamma', type=float, default=2)
parser.add_argument('--test_size', type=float, default=0.1)
parser.add_argument('--not_stratify_split', action="store_true", help="split dataset without stratification")
parser.add_argument('--stratify_on', type=str, default="emotion", help="column to stratify dataset on")

# model settings
parser.add_argument('--freeze_encoder', action='store_true')
parser.add_argument('--classifier_type', type=str, default="", help='The type of classifier to initialize based on language. OPTION: [en, zh]')

# initialization
args = parser.parse_args()
log = init_logger(os.path.join('./logs', args.log_filename), verbose=args.verbose)

assert args.excel_filename != "", "data path to excel file cannot be empty"
assert args.sheet_name != "", "sheet name of excel file cannot be empty"
assert "xlsx" in args.excel_filename.split("."), "only xlsx file format is supported"
assert args.stratify_on in ['emotion', 'stance'], "can only stratify dataset on emotion or stance"
assert args.classifier_type in ['en', 'zh'], "languages supported for classifier are only en, and zh"

data_path = os.path.join("./data", args.excel_filename)
train_data_path = "./data/temp/train.xlsx"
val_data_path = "./data/temp/val.xlsx"
model_weight_dir = os.path.join("./model_weights", args.saved_model_name)

# create dir
if not os.path.exists('./logs'):
    os.makedirs('./logs')
if not os.path.exists('./model_weights'):
    os.makedirs('./model_weights')
if not os.path.exists(model_weight_dir):
    os.makedirs(model_weight_dir)

df = pd.read_excel(data_path, sheet_name=args.sheet_name)

if args.not_stratify_split:
    train_df, val_df = train_test_split(df, test_size=args.test_size)
else:
    train_df, val_df = train_test_split(df, stratify=df.loc[:, args.stratify_on], test_size=args.test_size)

train_df.to_excel(train_data_path, index=False)
val_df.to_excel(val_data_path, index=False)
log(f'Size of training data: {len(train_df)}')
log(f'Size of validation data: {len(val_df)}')

if args.classifier_type == "en":
    encoder_ckpt = "bert-base-uncased"
    SocialMediaDS = EN_SocialMediaDS
elif args.classifier_type == "zh":
    encoder_ckpt = "hfl/chinese-bert-wwm-ext"
    SocialMediaDS = ZH_SocialMediaDS
    
log(f"Initialized {args.classifier_type} with {encoder_ckpt} encoder")
train_ds = SocialMediaDS(train_data_path, args.sheet_name, encoder_ckpt)
train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                      drop_last=True, collate_fn=train_ds.collate_fn)
val_ds = SocialMediaDS(val_data_path, args.sheet_name, encoder_ckpt)
val_dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=val_ds.collate_fn)

model = DoubleClassifier(encoder_ckpt, emotion_nlabels=7, stance_nlabels=3).to(device)
if not args.freeze_encoder:
    log("Unfreezed model encoder")
    model.unfreeze_encoder()

n_epoch = 10
loss_fn = focal_loss(args.focal_loss_gamma)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

mem = {
    'train_loss': [],
    'train_acc': [],
    'train_f1': [],
    'val_loss': [],
    'val_acc': [],
    'val_f1': []
}

cur_best_f1 = 0

for epoch in tqdm(range(args.epochs), desc='Training'):

    n_batch = len(train_dl)
    train_losses = []
    train_accs = []
    train_f1s = []

    for i, data in enumerate(train_dl):
        train_loss, train_metrics, _ = train(data, model, optimizer, loss_fn, loss_fn)
        pos = epoch + ((i+1)/n_batch)
        train_losses.append(train_loss.cpu().detach())
        train_accs.append(train_metrics['acc'])
        train_f1s.append(train_metrics['f1'])
        msg = f"epoch: {pos:.3f}\ttrain loss: {train_loss:.3f}\ttrain_acc: {train_metrics['acc']:.3f}\ttrain_f1: {train_metrics['f1']:.3f}"
        if args.verbose:
            print('\r', msg, end='')
    
    mem['train_loss'].append(np.mean(train_losses))
    mem['train_acc'].append(np.mean(train_accs))
    mem['train_f1'].append(np.mean(train_f1s))

    n_batch = len(val_dl)
    val_losses = []
    val_accs = []
    val_f1s = []

    for i, data in enumerate(val_dl):
        val_loss, val_metrics, _ = validate(data, model, loss_fn, loss_fn)
        pos = epoch + ((i+1)/n_batch)
        val_losses.append(val_loss.cpu().detach())
        val_accs.append(val_metrics['acc'])
        val_f1s.append(val_metrics['f1'])
        msg = f"epoch: {pos:.3f}\tval loss: {val_loss:.3f}\tval_acc: {val_metrics['acc']:.3f}\tval_f1: {val_metrics['f1']:.3f}"
        if args.verbose:
            print('\r', msg, end='')
    
    if args.verbose:
        print('\r', end='')
    mem['val_loss'].append(np.mean(val_losses))
    mem['val_acc'].append(np.mean(val_accs))
    mem['val_f1'].append(np.mean(val_f1s))

    msg = f"epoch: {epoch+1}\ntrain loss: {mem['train_loss'][-1]:.3f}\ttrain_acc: {mem['train_acc'][-1]:.3f}\ttrain_f1: {mem['train_f1'][-1]:.3f}"
    msg = msg + f"\nval loss: {mem['val_loss'][-1]:.3f}\tval_acc: {mem['val_acc'][-1]:.3f}\tval_f1: {mem['val_f1'][-1]:.3f}\n"
    log(msg)
    scheduler.step()
    model_weight_path = os.path.join(model_weight_dir, f"{args.saved_model_name}_{epoch+1}.pth")
    torch.save(model.state_dict(), model_weight_path)