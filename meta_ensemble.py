import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

def train_meta_learner(meta_features, y_train):
    """
    Trains a stronger LightGBM meta-learner with tuned hyperparameters.
    """
    # Split small validation for early stopping
    X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
        meta_features, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    meta_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=64,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1
    )

    # Use fit without early_stopping_rounds for compatibility
    meta_model.fit(X_meta_train, y_meta_train)

    return meta_model


def build_meta_features(branch_preds):
    """
    Stack all branch predictions as meta-features.
    branch_preds: list of numpy arrays (probabilities)
    """
    return np.hstack(branch_preds)
