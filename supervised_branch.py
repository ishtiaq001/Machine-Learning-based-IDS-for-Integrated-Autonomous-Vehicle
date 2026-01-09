import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

def train_xgb(X_train, y_train):
    """
    Train XGBoost with improved hyperparameters for higher accuracy.
    Improvements:
    - More estimators for better learning
    - Lower learning rate for stability
    - Use tree_method='hist' for speed on large datasets
    - Set subsample and colsample_bytree for regularization
    """
    model = xgb.XGBClassifier(
        n_estimators=500,          # increased trees
        max_depth=6,               # slightly deeper trees
        learning_rate=0.05,        # lower learning rate
        subsample=0.8,             # prevent overfitting
        colsample_bytree=0.8,      # feature subsampling
        tree_method='hist',        # faster on large datasets
        random_state=42,
        eval_metric='mlogloss'     # multi-class loss
    )
    model.fit(X_train, y_train)
    return model


def build_resnet(input_dim, num_classes):
    """
    Build a deeper, more regularized fully-connected network.
    Improvements:
    - Added Dropout to prevent overfitting
    - BatchNormalization for stable training
    - More neurons per layer for higher representation power
    """
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
