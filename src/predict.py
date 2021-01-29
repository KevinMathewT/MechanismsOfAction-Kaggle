from src.config import TEST_FEATURES
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def encodeLabelColumns(f):
    return MultiColumnLabelEncoder(["cp_type", "cp_dose"]).fit_transform(f)

def reorderFeatures(f):
    return pd.concat([getInfo(f), getOrderedGenes(f), getOrderedCells(f)], axis=1)

def getInfo(f):
    return f.loc[:, ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]

def getOrderedGenes(f, scaler="standard"):
    if scaler == "minmax":
        g = f.loc[:, sorted(f.loc[0, [cols for cols in f.columns if cols.startswith("g-")]].index.to_numpy())]
        g = (g - g.min()) / (g.max() - g.min())
    
    elif scaler == "standard":
        g = f.loc[:, sorted(f.loc[0, [cols for cols in f.columns if cols.startswith("g-")]].index.to_numpy())]
        g = (g - g.mean()) / g.std()
        
    return g


def getOrderedCells(f, scaler="standard"):
    if scaler == "minmax":
        c = f.loc[:, sorted(f.loc[0, [cols for cols in f.columns if cols.startswith("c-")]].index.to_numpy())]
        c = (c - c.min()) / (c.max() - c.min())
    
    elif scaler == "standard":
        c = f.loc[:, sorted(f.loc[0, [cols for cols in f.columns if cols.startswith("c-")]].index.to_numpy())]
        c = (c - c.mean()) / c.std()
        
    return c


class MultiColumnLabelEncoder:
    def __init__(self, columns = None):
        self.columns = columns # array of column names to encode

    def fit(self, X, y=None):
        return self # not relevant here

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

def drop_feature_cols(X):
    X = X.drop([col for col in ["sig_id", "kfold"] if col in X.columns], axis=1)
    return X

def drop_target_cols(X):
    return X

import torch
import torch.nn as nn
from src.config import *

class GeneRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = nn.LSTM(
            input_size=GENE_RNN_INPUT_SIZE,
            hidden_size=GENE_RNN_HIDDEN_SIZE,
            num_layers=GENE_RNN_LAYERS,
            bidirectional=GENE_BIDIRECTIONAL,
            dropout=GENE_RNN_DROPOUT
        )
        self.reduce = nn.Linear(
            in_features=(1 + GENE_BIDIRECTIONAL) * GENE_RNN_HIDDEN_SIZE,
            out_features=GENE_RNN_OUTPUT_SIZE
        )
        self.project = nn.Linear(
            in_features=GENE_TIME_STEPS,
            out_features=GENE_OUTPUT_SIZE
        )

    def forward(self, x):
        x = x.reshape(GENE_TIME_STEPS, -1, GENE_RNN_INPUT_SIZE) # x should be GENE_BATCH_SIZE x GENE_TIME_STEPS
        output, _ = self.recurrent(x)
        t, b, h = output.size()
        output = output.view(t * b, h)

        reduced = self.reduce(output).view(t, b, GENE_RNN_OUTPUT_SIZE).squeeze(2).permute(1, 0) # b x t, GENE_OUTPUT_SIZE = 1
        ret = self.project(reduced)

        return ret

class CellRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = nn.LSTM(
            input_size=CELL_RNN_INPUT_SIZE,
            hidden_size=CELL_RNN_HIDDEN_SIZE,
            num_layers=CELL_RNN_LAYERS,
            bidirectional=CELL_BIDIRECTIONAL,
            dropout=CELL_RNN_DROPOUT
        )
        self.reduce = nn.Linear(
            in_features=(1 + CELL_BIDIRECTIONAL) * CELL_RNN_HIDDEN_SIZE,
            out_features=CELL_RNN_OUTPUT_SIZE
        )
        self.project = nn.Linear(
            in_features=CELL_TIME_STEPS,
            out_features=CELL_OUTPUT_SIZE
        )

    def forward(self, x):
        x = x.reshape(CELL_TIME_STEPS, -1, CELL_RNN_INPUT_SIZE) # x should be CELL_BATCH_SIZE x CELL_TIME_STEPS
        output, _ = self.recurrent(x)
        t, b, h = output.size()
        output = output.view(t * b, h)

        reduced = self.reduce(output).view(t, b, CELL_RNN_OUTPUT_SIZE).squeeze(2).permute(1, 0) # b x t, CELL_OUTPUT_SIZE = 1
        ret = self.project(reduced)

        return ret

class MoANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.GeneNet = GeneRNN()
        self.CellNet = CellRNN()
        self.output = nn.Linear(
            in_features=GENE_OUTPUT_SIZE + CELL_OUTPUT_SIZE + 3,
            out_features=N_OUTPUTS
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, l = x.size()
        meta = x[:, 0:3]
        gene = x[:, 3:3+772]
        cell = x[:, 3+772:]

        gene = self.GeneNet(gene)
        cell = self.CellNet(cell)

        x = torch.cat([meta, gene, cell], dim=1)
        x = self.output(x)
        x = self.sigmoid(x)

        return x

f = pd.read_csv(TRAIN_FEATURES)
ids = f["sig_id"].to_numpy()
f = reorderFeatures(f)
f = encodeLabelColumns(f)
net = MoANet()
net.load_state_dict(torch.load(MODEL_WEIGHT_PATH))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
net.eval()

f = drop_feature_cols(f).to_numpy()
f = torch.tensor(f, dtype=torch.float32).to(device)
p = net(f)

submission = pd.read_csv(SAMPLE_SUBMISSION)
# print(ids.shape, p.cpu().detach().numpy().shape)
submission = pd.DataFrame(data=np.c_[ids, p.cpu().detach().numpy()], columns=submission.columns)
# submission[:, 1:] = p.cpu().detach().numpy()

submission.to_csv("submission.csv", index=False)
