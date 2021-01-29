import sklearn.metrics as metrics
threshold = 0.5

def getPrecisionScore(p, t):
    p = (p > 0.5).astype("int32")
    t = t.astype("int32")
    ret_score = 0
    for a, b in zip(p, t):
        ret_score += metrics.precision_score(b, a)
    return ret_score / len(p)

def getAccuracyScore(p, t):
    p = (p > 0.5).astype("int32")
    t = t.astype("int32")
    ret_score = 0
    for a, b in zip(p, t):
        ret_score += metrics.accuracy_score(b, a)
    return ret_score / len(p)

def getRecallScore(p, t):
    p = (p > 0.5).astype("int32")
    t = t.astype("int32")
    ret_score = 0
    for a, b in zip(p, t):
        ret_score += metrics.recall_score(b, a)
    return ret_score / len(p)

def getAUCScore(p, t):
    ret_score = 0
    for a, b in zip(p, t):
        ret_score += metrics.auc(b, a)
    return ret_score / len(p)

def getLogLoss(p, t):
    ret_score = 0
    for a, b in zip(p, t):
        ret_score += metrics.log_loss(b, a)
    return ret_score / len(p)

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(n_quantiles=2, random_state=0)