from src.config import TRAIN_TEST_SPLIT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from src import config, create_folds, utils, train

f = pd.read_csv(config.TRAIN_FEATURES)
t = pd.read_csv(config.TRAIN_TARGETS)
f = utils.reorderFeatures(f)
f = utils.encodeLabelColumns(f)
train_f, train_t, test_f, test_t = f[:int(TRAIN_TEST_SPLIT * len(t))], t[:int(TRAIN_TEST_SPLIT * len(t))], f[int(TRAIN_TEST_SPLIT * len(t)):], t[int(TRAIN_TEST_SPLIT * len(t)):]
train_f = create_folds.create_folds(train_f, train_t)

train.trainer(train_f, train_t, test_f, test_t)

# print(train_f.head(10))
# print(len(train_f))