import os

DATA                = "data"
TRAIN_FEATURES      = os.path.join(DATA, "train_features.csv")
TEST_FEATURES       = os.path.join(DATA, "test_features.csv")
TRAIN_TARGETS       = os.path.join(DATA, "train_targets_scored.csv")
SAMPLE_SUBMISSION   = os.path.join(DATA, "sample_submission.csv")
TRAINING_DATA       = os.path.join(DATA, "train_folds.csv")
MODEL_WEIGHT_PATH   = 'models/weights/net_41_1.pth'
TRAIN_TEST_SPLIT    = 0.8
TIME                = 1

N_FOLDS             = 5
FOLD_MAPPING        = {fold: fold_mapping for fold, fold_mapping in enumerate([[x for x in range(N_FOLDS) if x != i] for i in range(N_FOLDS)])}

BATCH_SIZE          = 32
N_EPOCHS            = 100
EARLY_STOPPING      = 5
LEARNING_RATE       = 0.001
ONE_W               = 300
ZERO_W              = 1

# GENE RNN
GENE_TIME_STEPS     = 772
GENE_RNN_INPUT_SIZE = 1
GENE_RNN_HIDDEN_SIZE= 1
GENE_RNN_LAYERS     = 1
GENE_BIDIRECTIONAL  = True
GENE_RNN_DROPOUT    = 0.3
GENE_RNN_OUTPUT_SIZE= 1
GENE_OUTPUT_SIZE    = 256
GENE_BATCH_SIZE     = BATCH_SIZE

# CELL RNN
CELL_TIME_STEPS     = 100
CELL_RNN_INPUT_SIZE = 1
CELL_RNN_HIDDEN_SIZE= 1
CELL_RNN_LAYERS     = 1
CELL_BIDIRECTIONAL  = True
CELL_RNN_DROPOUT    = 0.3
CELL_RNN_OUTPUT_SIZE= 1
CELL_OUTPUT_SIZE    = 64
CELL_BATCH_SIZE     = BATCH_SIZE

# Output
N_OUTPUTS           = 206