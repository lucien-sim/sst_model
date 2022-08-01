from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from scipy.linalg import eigh
import numpy as np


def censored_lstsq(A, B, M):
    """Solves least squares problem (AX = B) subject to missing data 
    (mask M with 0 if data missing, 1 if present). 

    Reference: http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/

    Args:
        A (np.ndarray): m x r matrix
        B (np.ndarray): m x n matrix
        M (np.ndarray): m x n binary matrix (zeros indicate missing values)

    Returns:
        np.ndarray: r x n matrix that minimizes norm(M*(AX - B))
    """

    # if B is a vector, simply drop out corresponding rows in A
    if B.ndim == 1 or B.shape[1] == 1:
        return np.linalg.leastsq(A[M], B[M])[0]

    # else solve via tensor representation
    rhs = np.dot(A.T, M * B).T[:, :, None] # n x r x 1 tensor
    T = np.matmul(np.expand_dims(A.T, axis=0), M.T[:, :, None] * np.expand_dims(A, axis=0)) # n x r x r tensor
    return np.squeeze(np.linalg.solve(T, rhs)).T # transpose to get r x n


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
        if X.shape != sample_weights.shape:
            raise ValueError("sample_weights must have same shape as X.")

        # Does not work with NaNs
        if np.sum(np.isnan(X)) > 0: 
            raise ValueError("Input X cannot have NaNs.")

        if sample_weights is None:
            sample_weights = np.ones(X.shape)

        # Compute weighted covariance matrix. 
        self.centers_ = ((X * sample_weights).sum(axis=0) / sample_weights.sum(axis=0))
        Xc = X - self.centers_
        covar = (Xc * sample_weights).T @ (Xc * sample_weights) / (sample_weights.T @ sample_weights)
        covar[np.isnan(covar)] = 0.  # Fill entries where w.T @ w is 0.

        # Calculate eigenvalues and eigenvectors, cumulative explained variance. 
        index_subset = (X.shape[1] - n_components, X.shape[1] - 1)
        eigval, eigvec = eigh(covar, subset_by_index=index_subset)
        self.explained_variance_ = eigval[::-1][:n_components]
        self.components_ = eigvec[:, -n_components:][:, ::-1].T

        return self

    def transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Project DataFrame along EOFs. 

        Args:
            X (np.ndarray): Raw, unprojected array. 
            y (np.ndarray, optional): y variable. Defaults to None.

        Returns:
            np.ndarray: Array of data projected along EOFs. 
        """
        return (X - self.centers_) @ self.components_.T

    def inverse_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Reproject projected data along original dimensions. 

        Args:
            X (np.ndarray): Projected ata.
            y (np.ndarray, optional): y variable. Defaults to None.

        Returns:
            np.ndarray: Array along original dimensions. 
        """
        return X @ self.components_ + self.centers_


class WPCAImputer(TransformerMixin, BaseEstimator): 

    def __init__(self, missing_values: float = np.nan, n_components: int = 30, copy: bool = True):
        """Transformer for imputing with PCA. 
        
        Procedure: 
        1. Use weighted PCA to calculate principal components from incomplete dataset. 
        2. Use least squares to calculate weights that minimize norm(M*(E @ W - X)), where M 
        is a mask of missing data, E is an array of EOFs, W is an array of weights 
        (one per EOF/record), and X is the incomplete dataset.  
        3. Then multiply E @ W to predict all values for each record, based on 
        values that are available. 
        4. Use product E @ W to fill missing data. 

        Args:
            missing_values (float, optional): The placeholder for the missing values. 
                All occurrences of missing_values will be imputed. Defaults to np.nan.
            n_components (int, optional): Number of components to use for PCA. Defaults to 30.
            copy (bool, optional): If True, a copy of X will be created. 
                If False, imputation will be done in-place. Defaults to True.
        """
        self.missing_values = missing_values
        self.n_components = n_components
        self.copy = copy
        self.wpca = None

    def _missing_mask(self, X: np.ndarray) -> np.ndarray: 
        """Generate mask of missing values given X

        Args:
            X (np.ndarray): input array X. 

        Returns:
            np.ndarray: Mask with True where missing value is present, else False. 
        """

        if np.isnan(self.missing_values): 
            missing = np.isnan(X)
        elif pd.isna(self.missing_values): 
            missing = pd.isna(X)
        else: 
            missing = (X == self.missing_values)

        return missing

    def pca_predict(self, X: np.ndarray, missing_mask: np.ndarray = None) -> np.ndarray: 
        """Use PCA to predict all values for each record. These values 
        will be used to fill missing data during imputation. 

        Args:
            X (np.ndarray): input array X. 
            X (np.ndarray, optional): Missing data mask, defaults to None. 

        Returns:
            np.ndarray: array of pca-based predictions.  
        """
        
        # Calcualte PCA weights at each timestep
        if missing_mask is None: 
            missing_mask = self._missing_mask(X)
        w = censored_lstsq(
            self.wpca.components_.T, 
            np.asarray(np.where(missing_mask, 0, X).T), 
            np.asarray((~missing_mask)).T
        )

        return (self.wpca.components_.T @ w).T

    def fit(self, X: np.ndarray, y: np.ndarray = None): 
        """Calculate EOFs for later use. 

        Args:
            X (np.ndarray): Array with missing data. 
            y (np.ndarray, optional): y variable, isn't used. Defaults to None.

        Returns:
            self: 
        """
        
        # Use weighted PCA to calculate EOFs with missing data
        missing = self._missing_mask(X)
        self.wpca = WPCA(n_components=self.n_components).fit(
            np.where(missing, 0, X), y=None, sample_weights=~missing)

        return self
        
    def transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray: 
        """Fill missing data. 

        Args:
            X (np.ndarray): Array with missing data
            y (np.ndarray, optional): y variable, isn't used. Defaults to None.

        Returns:
            np.ndarray: Array with missing data imputed. 
        """

        if self.copy: 
            X = np.copy(X)
        
        missing = self._missing_mask(X)
        preds = self.pca_predict(X, missing_mask=missing)
        X[missing] = preds

        return X
