from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from scipy.linalg import eigh
import numpy as np


class PCACleaner(TransformerMixin, BaseEstimator):
    pass


class PCAImputer(TransformerMixin, BaseEstimator):
    pass


class WPCA(TransformerMixin, BaseEstimator):

    def __init__(self, n_components: int = None):
        self.n_components = n_components
        self.centers_ = None
        self.explained_variance_ = None
        self.components_ = None

    def fit(self, X: np.ndarray, y: np.ndarray = None, sample_weights: np.ndarray = None):
        """Calculate sample-weighted EOFs.

        Args:
            X (np.ndarray): Array of data with records along the first dimension, features
                along the second. 
            y (np.ndarray, optional): y variable, which isn't used. Defaults to None.
            sample_weights (np.ndarray, optional): Weights to use for each value in X. Must
                have the same shape as X. Defaults to None.

        Raises:
            ValueError: if sample_weights is specified and is not an np.ndarray with 
                the same dimensions as X. 

        Returns:
            self: 
        """

        # Assign n_components if not already assigned. 
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        # Assign sample weights if not already assigned. 
        if not (sample_weights is None or isinstance(sample_weights, np.ndarray)):
            raise ValueError("sample_weights must be an array with the same shape as X.")
        elif isinstance(sample_weights, np.ndarray) and X.shape != sample_weights.shape:
            raise ValueError("sample_weights must have same shape as X.")

        if sample_weights is None:
            sample_weights = np.ones(X.shape)

        # Compute weighted covariance matrix. 
        self.centers_ = ((X * sample_weights).sum(axis=0) / sample_weights.sum(axis=0))
        Xc = X - self.centers_
        covar = (Xc * sample_weights).T @ (Xc * sample_weights) / (sample_weights.T @ sample_weights)
        covar[np.isnan(covar)] = 0.  # Fill entries where w.T @ w is 0.

        # Calculate eigenvalues and eigenvectors, cumulative explained variance. 
        eigval_idx = (X.shape[1] - n_components, X.shape[1] - 1)
        eigval, eigvec = eigh(covar, eigvals=eigval_idx)
        self.explained_variance_ = eigval[::-1][:n_components]
        self.components_ = eigvec[:, :n_components][:, ::-1]

        return self

    def transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Project DataFrame along EOFs. 

        Args:
            X (np.ndarray): Raw, unprojected array. 
            y (np.ndarray, optional): y variable. Defaults to None.

        Returns:
            np.ndarray: Array of data projected along EOFs. 
        """
        return (X - self.centers_) @ self.components_

    def inverse_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Reproject projected data along original dimensions. 

        Args:
            X (np.ndarray): Projected ata.
            y (np.ndarray, optional): y variable. Defaults to None.

        Returns:
            np.ndarray: Array along original dimensions. 
        """
        return X @ self.components_.T + self.centers_


class WPCACleaner(TransformerMixin, BaseEstimator):
    # Process might be 
    # (1) k fold fit/impute to get predictions for everything
    # (2) Assign error thresholds
    # (3) Remove values with errors > thresholds
    pass


class WPCAImputer(TransformerMixin, BaseEstimator):
    pass


# TODO: Add regularization to WPCA, write PCA-based cleaner, write PCA-based imputer


if __name__ == '__main__':

    X = np.random.normal(0, 1, size=(20, 10))
    p = PCA(n_components=5).fit(X)
    xt = p.transform(X)
    Xr = p.inverse_transform(xt)

    wp = WPCA(n_components=5).fit(X)
    wxt = wp.transform(X)

    wp2 = WPCA(n_components=5).fit(X, sample_weights=np.random.uniform(size=X.shape))
    wxt2 = wp2.transform(X)

    pass


