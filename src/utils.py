import pandas as pd
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