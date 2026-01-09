from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_autoencoder(X_train, encoding_dim=64, epochs=50, batch_size=128):
    """
    Train a fully-connected autoencoder and return encoder and scaler.
    Improvements:
    - Scale input for stability
    - Larger encoding_dim for richer latent representation
    - BatchNormalization & Dropout to prevent NaNs and overfitting
    """
    # Scale input
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))

    # Encoder
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.1)(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)

    # Decoder
    decoded = Dense(128, activation='relu')(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dropout(0.1)(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(
        X_train_scaled,
        X_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    # Encoder model returns scaled latent representation
    encoder_model = Model(inputs=input_layer, outputs=encoded)

    return autoencoder, encoder_model, scaler


def train_gmm(X_encoded, n_components):
    """
    Fit Gaussian Mixture Model directly on latent space
    - Ensure float64 for numerical stability
    - Increased regularization for stability
    """
    X_encoded = X_encoded.astype(np.float64)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=42,
        reg_covar=1e-3,
        max_iter=500,
        init_params='kmeans'
    )

    gmm.fit(X_encoded)
    return gmm
