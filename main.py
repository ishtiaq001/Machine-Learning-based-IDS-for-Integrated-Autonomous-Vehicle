import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

from data_preprocessing import load_and_merge_nsl_cse
from supervised_branch import train_xgb, build_resnet
from unsupervised_branch import train_autoencoder, train_gmm
from semi_supervised_branch import create_graph_data, train_gnn
from rl_branch import train_rl
from meta_ensemble import build_meta_features, train_meta_learner
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --------------------------
# Step 0: Paths to datasets
# --------------------------
cwd = os.getcwd()
nsl_train_file = os.path.join(cwd, "NSL_KDD-master", "KDDTrain+.txt")
nsl_test_file  = os.path.join(cwd, "NSL_KDD-master", "KDDTest+.txt")
cse_folder = os.path.join(cwd, "CSE-CIC-IDS2018")  # folder with CSE CSVs

categorical_cols = ['protocol_type', 'service', 'flag']

# --------------------------
# Step 1: Load & merge datasets
# --------------------------
X_train, X_test, y_train, y_test = load_and_merge_nsl_cse(
    nsl_path="NSL_KDD-master/KDDTrain+.txt",
    cse_folder="CSE-CIC-IDS2018",
    categorical_cols=['protocol_type','service','flag'],
    label_col='label'
)

num_classes = len(np.unique(y_train))

# --------------------------
# Step 2: Supervised branch
# --------------------------
xgb_model = train_xgb(X_train, y_train)

resnet_model = build_resnet(X_train.shape[1], num_classes)
resnet_model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1, verbose=1)

# --------------------------
# Step 3: Unsupervised branch
# --------------------------
autoencoder, encoder_model = train_autoencoder(X_train, encoding_dim=32)
X_encoded_train = encoder_model.predict(X_train)

gmm_model = train_gmm(X_encoded_train, n_components=num_classes)

# --------------------------
# Step 4: Semi-supervised branch
# --------------------------
graph_data = create_graph_data(X_train, y_train)
gnn_model = train_gnn(graph_data, X_train.shape[1], num_classes)

# --------------------------
# Step 5: RL branch
# --------------------------
rl_model = train_rl(X_train, y_train)

# --------------------------
# Step 6: Build meta features (train)
# --------------------------
xgb_preds = xgb_model.predict_proba(X_train)
resnet_preds = resnet_model.predict(X_train)
gmm_preds = gmm_model.predict_proba(X_encoded_train)

gnn_model.eval()
graph_data.eval = None  # avoid torch warning
with torch.no_grad():
    gnn_preds = torch.exp(gnn_model(graph_data)).numpy()

rl_preds = np.zeros_like(xgb_preds)
for i in range(len(X_train)):
    rl_preds[i, rl_model.predict(X_train[i].reshape(1, -1))[0]] = 1

meta_features = build_meta_features([xgb_preds, resnet_preds, gmm_preds, gnn_preds, rl_preds])
meta_learner = train_meta_learner(meta_features, y_train)

# --------------------------
# Step 7: Evaluate (test)
# --------------------------
xgb_preds_test = xgb_model.predict_proba(X_test)
resnet_preds_test = resnet_model.predict(X_test)
X_encoded_test = encoder_model.predict(X_test)
gmm_preds_test = gmm_model.predict_proba(X_encoded_test)

graph_data_test = Data(x=torch.tensor(X_test, dtype=torch.float), edge_index=graph_data.edge_index)
gnn_model.eval()
with torch.no_grad():
    gnn_preds_test = torch.exp(gnn_model(graph_data_test)).numpy()

rl_preds_test = np.zeros_like(xgb_preds_test)
for i in range(len(X_test)):
    rl_preds_test[i, rl_model.predict(X_test[i].reshape(1, -1))[0]] = 1

meta_features_test = build_meta_features([xgb_preds_test, resnet_preds_test, gmm_preds_test, gnn_preds_test, rl_preds_test])
final_preds = meta_learner.predict(meta_features_test)

# --------------------------
# Step 8: Print evaluation
# --------------------------
print("Accuracy:", accuracy_score(y_test, final_preds))
print("Precision:", precision_score(y_test, final_preds, average='weighted'))
print("Recall:", recall_score(y_test, final_preds, average='weighted'))
print("F1-score:", f1_score(y_test, final_preds, average='weighted'))
