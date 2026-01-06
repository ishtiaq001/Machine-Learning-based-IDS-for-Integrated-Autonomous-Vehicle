from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import numpy as np

def train_autoencoder(X_train, encoding_dim=32, epochs=20, batch_size=64):
    """
    Train a simple fully-connected autoencoder and return encoder
    """
    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))

    # Encoder
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    # Decoder
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)

    # Autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train autoencoder
    autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    # Encoder model to get compressed features
    encoder_model = Model(inputs=input_layer, outputs=encoded)

    return autoencoder, encoder_model


def train_gmm(X_encoded, n_components, reduce_dim=True, encoding_threshold=20):
    """
    Fit a Gaussian Mixture Model on encoded features
    X_encoded: numpy array from autoencoder encoder
    n_components: number of clusters (classes)
    reduce_dim: optionally reduce high-dimensional encodings via PCA
    encoding_threshold: max dimension before applying PCA
    """
    # Ensure float64 for numerical stability
    X_encoded = X_encoded.astype(np.float64)

    # Reduce dimensionality if too high (stabilizes GMM)
    if reduce_dim and X_encoded.shape[1] > encoding_threshold:
        pca = PCA(n_components=0.95, random_state=42)
        X_encoded = pca.fit_transform(X_encoded)

    # Fit GMM with small regularization to avoid singular covariance
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=42,
        reg_covar=1e-4
    )

    gmm.fit(X_encoded)
    return gmm
