import lightgbm as lgb
import numpy as np

def train_meta_learner(meta_features, y_train):
    meta_model = lgb.LGBMClassifier()
    meta_model.fit(meta_features, y_train)
    return meta_model

def build_meta_features(branch_preds):
    # branch_preds: list of numpy arrays from all branches
    return np.hstack(branch_preds)