import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_merge_nsl_cse(nsl_path, cse_folder, label_col='label', categorical_cols=None, test_size=0.3):
    """
    Load NSL-KDD and CSE-CIC-IDS2018 datasets and merge them safely using chunked reading for CSE.
    """
    # -------------------------
    # Load NSL-KDD
    # -------------------------
    nsl_data = pd.read_csv(nsl_path)

    # -------------------------
    # Load CSE-CIC-IDS2018 in chunks
    # -------------------------
    cse_files = [os.path.join(cse_folder, f) for f in os.listdir(cse_folder) if f.endswith(".csv")]
    cse_chunks = []
    for file in cse_files:
        print(f"Processing CSE file: {file}")
        for chunk in pd.read_csv(file, chunksize=50000):  # load 50k rows at a time
            cse_chunks.append(chunk)
    cse_data = pd.concat(cse_chunks, ignore_index=True)
    del cse_chunks  # free memory

    # -------------------------
    # Merge NSL + CSE
    # -------------------------
    print("Merging NSL-KDD and CSE-CIC-IDS2018...")
    full_data = pd.concat([nsl_data, cse_data], ignore_index=True)
    del nsl_data, cse_data  # free memory

    # -------------------------
    # Encode categorical columns
    # -------------------------
    if categorical_cols:
        for col in categorical_cols:
            if col in full_data.columns:
                le = LabelEncoder()
                full_data[col] = le.fit_transform(full_data[col].astype(str))

    # -------------------------
    # Split features and labels
    # -------------------------
    X = full_data.drop(label_col, axis=1).values.astype(np.float32)
    y = full_data[label_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # -------------------------
    # Standardize
    # -------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
