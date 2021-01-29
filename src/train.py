from math import inf

from torch import dtype
from src.utils import drop_feature_cols, drop_target_cols
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import *
from models.nets import MoANet

def loss_fn(p, t):
    p = p.view(-1)
    t = t.view(-1)
    w = (t * (ONE_W - 1)) + 1
    bce = nn.BCELoss(weight=w)
    loss = bce(p, t)
    return loss


def train(f, t, FOLD, net, optimizer, device):
    train_t = t[f.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_t = t[f.kfold == FOLD]
    train_f = f[f.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_f = f[f.kfold == FOLD]
    train_f = drop_feature_cols(train_f).to_numpy()
    valid_f = drop_feature_cols(valid_f).to_numpy()
    train_t = drop_target_cols(train_t).to_numpy()
    valid_t = drop_target_cols(valid_t).to_numpy()

    train_dataloader = DataLoader(dataset=list(zip(train_f, train_t)), batch_size=BATCH_SIZE)
    valid_dataloader = DataLoader(dataset=list(zip(valid_f, valid_t)), batch_size=BATCH_SIZE)
    
    train_loss = 0.0
    batches_done = 0

    net.train()
    print()

    for i, batch in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()

        batches_done += 1
        f, t = batch
        f = torch.tensor(f, dtype=torch.float32).to(device)
        t = torch.tensor(t, dtype=torch.float32).to(device)
        p = net(f)

        loss = loss_fn(p, t)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss /= batches_done

    valid_loss = 0.0
    batches_done = 0

    net.eval()

    for i, batch in enumerate(tqdm(valid_dataloader)):
        batches_done += 1
        f, t = batch
        f = torch.tensor(f, dtype=torch.float32).to(device)
        t = torch.tensor(t, dtype=torch.float32).to(device)
        p = net(f)

        loss = loss_fn(p, t)
        valid_loss += loss.item()

    valid_loss /= batches_done

    return train_loss, valid_loss

def trainer(f, t, test_f, test_t):
    t = f.merge(t, on="sig_id").iloc[:, -(len(t.columns)-1):]
    test_t = test_f.merge(test_t, on="sig_id").iloc[:, -(len(test_t.columns)-1):]

    net = MoANet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    net = net.to(device)
    optimizer = optim.RMSprop(params=net.parameters(), lr=LEARNING_RATE)

    last_improvement = 0
    last_train_loss = inf
    last_valid_loss = inf

    for epoch in range(N_EPOCHS):
        train_loss, valid_loss = train(f, t, epoch % N_FOLDS, net, optimizer, device)
        print("({:<3d} / {:>3d}) | Training Loss: {:.12f} | Validation Loss: {:.12f} | Test Loss: {:.12f}".format(
            epoch, N_EPOCHS, train_loss, valid_loss, evaluate(test_f, test_t, net, optimizer, device)))

        torch.save(net.state_dict(), f"models/weights/net_{epoch}_{TIME}.pth")

        if valid_loss < last_valid_loss:
            last_improvement = 0
        else:
            last_improvement += 1
        
        if last_improvement == EARLY_STOPPING:
            print(f"No improvement in {EARLY_STOPPING} epochs. Training Stopped.")
            return

def evaluate(f, t, net, optimizer, device):
    f = drop_feature_cols(f).to_numpy()
    t = drop_target_cols(t).to_numpy()

    test_dataloader = DataLoader(dataset=list(zip(f, t)), batch_size=BATCH_SIZE)

    test_loss = 0.0
    batches_done = 0

    net.eval()

    for i, batch in enumerate(tqdm(test_dataloader)):
        batches_done += 1
        f, t = batch
        f = torch.tensor(f, dtype=torch.float32).to(device)
        t = torch.tensor(t, dtype=torch.float32).to(device)
        p = net(f)

        loss = loss_fn(p, t)
        test_loss += loss.item()

    test_loss /= batches_done

    return test_loss