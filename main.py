import os
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_preprocessing import load_nsl, load_cse
from supervised_branch import train_xgb, build_resnet
from unsupervised_branch import train_autoencoder, train_gmm
from semi_supervised_branch import create_graph_data, train_gnn
from rl_branch import train_rl
from meta_ensemble import build_meta_features, train_meta_learner

# --------------------------
# Paths
# --------------------------
cwd = os.getcwd()
nsl_train_file = os.path.join(cwd, "NSL_KDD-master", "KDDTrain+.txt")
cse_folder = os.path.join(cwd, "CSE-CIC-IDS2018")

categorical_cols = ['protocol_type', 'service', 'flag']

# --------------------------
# Utility: Encode labels safely
# --------------------------
def encode_labels(y_train, y_test):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc, le

# --------------------------
# Function to run full pipeline
# --------------------------
def run_pipeline(X_train, X_test, y_train, y_test, dataset_name="Dataset"):
    print(f"\n===== Running pipeline for {dataset_name} =====")

    # Encode labels
    y_train, y_test, label_encoder = encode_labels(y_train, y_test)
    num_classes = len(np.unique(y_train))

    # ----------------------
    # Supervised branch
    # ----------------------
    xgb_model = train_xgb(X_train, y_train)

    resnet_model = build_resnet(X_train.shape[1], num_classes)
    resnet_model.fit(
        X_train,
        y_train,
        epochs=1000,           # enough for convergence
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # ----------------------
    # Unsupervised branch
    # ----------------------
    autoencoder, encoder_model, scaler = train_autoencoder(
        X_train,
        encoding_dim=64,
        epochs=1000,
        batch_size=64
    )

    # Scale train and test for latent encoding
    X_encoded_train = encoder_model.predict(scaler.transform(X_train), verbose=0)
    X_encoded_test  = encoder_model.predict(scaler.transform(X_test), verbose=0)

    gmm_model = train_gmm(
        X_encoded_train,
        n_components=num_classes
    )

    # ----------------------
    # Semi-supervised branch (GNN)
    # ----------------------
    graph_data = create_graph_data(X_encoded_train, y_train)
    gnn_model = train_gnn(
        graph_data,
        input_dim=X_encoded_train.shape[1],
        num_classes=num_classes,
        epochs=1000,
        lr=0.005
    )

    # ----------------------
    # RL branch
    # ----------------------
    rl_model = train_rl(X_train, y_train, timesteps=20000)

    # ----------------------
    # Meta-feature training
    # ----------------------
    xgb_preds = xgb_model.predict_proba(X_train)
    resnet_preds = resnet_model.predict(X_train, verbose=0)
    gmm_preds = gmm_model.predict_proba(X_encoded_train)

    gnn_model.eval()
    with torch.no_grad():
        gnn_preds = torch.exp(gnn_model(graph_data)).numpy()

    rl_preds = np.zeros_like(xgb_preds)
    for i in range(len(X_train)):
        rl_preds[i, rl_model.predict(X_train[i].reshape(1, -1))[0]] = 1

    meta_features = build_meta_features([xgb_preds, resnet_preds, gmm_preds, gnn_preds, rl_preds])
    meta_learner = train_meta_learner(meta_features, y_train)

    # ----------------------
    # Evaluation
    # ----------------------
    xgb_preds_test = xgb_model.predict_proba(X_test)
    resnet_preds_test = resnet_model.predict(X_test, verbose=0)
    gmm_preds_test = gmm_model.predict_proba(X_encoded_test)

    graph_data_test = create_graph_data(X_encoded_test, y_test)
    with torch.no_grad():
        gnn_preds_test = torch.exp(gnn_model(graph_data_test)).numpy()

    rl_preds_test = np.zeros_like(xgb_preds_test)
    for i in range(len(X_test)):
        rl_preds_test[i, rl_model.predict(X_test[i].reshape(1, -1))[0]] = 1

    meta_features_test = build_meta_features([
        xgb_preds_test,
        resnet_preds_test,
        gmm_preds_test,
        gnn_preds_test,
        rl_preds_test
    ])

    final_preds = meta_learner.predict(meta_features_test)

    # ----------------------
    # Metrics
    # ----------------------
    print(f"\nResults for {dataset_name}:")
    print("Accuracy :", accuracy_score(y_test, final_preds))
    print("Precision:", precision_score(y_test, final_preds, average='weighted'))
    print("Recall   :", recall_score(y_test, final_preds, average='weighted'))
    print("F1-score :", f1_score(y_test, final_preds, average='weighted'))


# ==========================================================
# Run NSL-KDD
# ==========================================================
X_train_nsl, X_test_nsl, y_train_nsl, y_test_nsl = load_nsl(
    nsl_train_file,
    categorical_cols=categorical_cols
)
run_pipeline(X_train_nsl, X_test_nsl, y_train_nsl, y_test_nsl, dataset_name="NSL-KDD")

# ==========================================================
# Run CSE-CIC-IDS2018
# ==========================================================
X_train_cse, X_test_cse, y_train_cse, y_test_cse = load_cse(
    cse_folder,
    categorical_cols=categorical_cols
)
run_pipeline(X_train_cse, X_test_cse, y_train_cse, y_test_cse, dataset_name="CSE-CIC-IDS2018")

