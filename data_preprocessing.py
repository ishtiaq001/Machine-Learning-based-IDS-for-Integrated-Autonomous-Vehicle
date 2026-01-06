import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# --------------------------------------------------
# Helper: encode categorical columns safely
# --------------------------------------------------
def encode_categorical(df, categorical_cols):
    df = df.copy()
    df = pd.get_dummies(df, columns=categorical_cols)
    return df


# ==================================================
# NSL-KDD Loader (FIXED)
# ==================================================
def load_nsl(file_path, categorical_cols=None):
    col_names = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes',
        'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
        'num_compromised','root_shell','su_attempted','num_root',
        'num_file_creations','num_shells','num_access_files',
        'num_outbound_cmds','is_host_login','is_guest_login','count',
        'srv_count','serror_rate','srv_serror_rate','rerror_rate',
        'srv_rerror_rate','same_srv_rate','diff_srv_rate',
        'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
        'dst_host_same_srv_rate','dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
        'dst_host_serror_rate','dst_host_srv_serror_rate',
        'dst_host_rerror_rate','dst_host_srv_rerror_rate',
        'label'
    ]

    df = pd.read_csv(file_path, names=col_names)
    print("NSL-KDD loaded:", df.shape)

    y = df['label']
    X = df.drop(columns=['label'])

    # encode categorical columns BEFORE float conversion
    if categorical_cols:
        X = encode_categorical(X, categorical_cols)

    # Convert everything to float safely
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    X = X.values.astype(np.float32)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ==================================================
# CSE-CIC-IDS2018 Loader (FIXED)
# ==================================================
def load_cse(folder_path, categorical_cols=None):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    df_list = []
    for f in csv_files:
        df_list.append(pd.read_csv(os.path.join(folder_path, f)))

    df = pd.concat(df_list, ignore_index=True)
    print("CSE-CIC-IDS2018 loaded:", df.shape)

    y = df['Label']
    X = df.drop(columns=['Label'])

    # Handle categorical columns if any
    if categorical_cols:
        existing = [c for c in categorical_cols if c in X.columns]
        X = encode_categorical(X, existing)

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    X = X.values.astype(np.float32)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
