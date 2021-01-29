import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from src.config import *

def create_folds(X, y):
    X["kfold"] = -1
    # X = X.sample(frac=1.0).reset_index(drop=True)

    kf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=False, random_state=1337)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=X, y=y)):
        print(len(train_idx), len(val_idx))
        X.loc[val_idx, "kfold"] = fold

    return X
    