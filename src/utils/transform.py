import numpy as np
import xarray as xr

from scipy.fftpack import rfft, irfft
from scipy import sparse


class FFTSmoother:

    def __init__(self, axis: int = 0, fft_keep_frac: float = 0.03): 
        """Class for creating an fft-smoothed version of an N-dimensional real 
        numpy array, where the smoothing only occurs along one dimension.  

        Args:
            axis (int, optional): Axis along which to smooth. Defaults to 0.
            fft_keep_frac (float, optional): Top fraction of FFT coefficients 
                to keep. Defaults to 0.03.
        """
        self.axis = axis
        self.fft_keep_frac = fft_keep_frac

        self.sparse_coefs = None
        self.orig_shape = None
        self.new_shape = None
        
    def fit(self, x: np.ndarray): 
        """Find top FFT coefficients for generating smoothed array. 

        Args:
            x (np.ndarray): array to be smoothed.

        Returns:
            self: 
        """

        # Perform real number FFT
        f = rfft(x, axis=self.axis)

        # Sort coefficients along axis of FFT, pick thresholds that separate 
        # bottom [1 - fft_keep_frac] coefficients from top [fft_keep_frac]. 
        f_sort = np.sort(np.abs(f), axis=self.axis)
        thresholds = np.min(f_sort[int((1 - self.fft_keep_frac) * f_sort.shape[0]):, :, :], axis=self.axis)

        # Set all lower coefficients to zero. This is the smoothing. 
        f[np.abs(f) < thresholds] = 0

        # Cache sparse version (scipy can only handle 2d so we need to reshape)
        f_2d = f.reshape(f.shape[0], -1)
        self.orig_shape = f.shape
        self.new_shape = f_2d.shape
        self.sparse_coefs = sparse.coo_matrix(f_2d)

        return self

    def get_smoothed(self) -> np.ndarray: 
        """Generate smoothed array.

        Returns:
            np.ndarray: array of smoothed values.
        """
        return irfft(self.sparse_coefs.toarray().reshape(self.orig_shape), axis=self.axis)


class SeasonalityRemoverFFT: 
    
    def __init__(self, div_std: bool = False, fft_keep_frac: float = 0.03, time_dim: str = 'time'):
        """Removes seasonality from geophysical data with a time dimension. 

        Args:
            div_std (bool, optional): Whether to divide by seasonal 
                STD in normalization. Defaults to False.
            fft_keep_frac (float, optional): Fraction of fourier coefficients 
                to keep in smoothing of mean/std. Defaults to 0.03.
            time_dim (str, optional): Name of time dimension in DataArray. 
                Defaults to 'time'.
        """
        self.div_std = div_std
        self.keep_frac = fft_keep_frac
        self.time_dim = time_dim

        self.mean_fft = None
        self.std_fft = None
        self.seasonal_dims = None
        self.seasonal_coords = None
        
    def fit(self, X: xr.DataArray): 
        """Calculates means and stds needed to remove seasonality.

        Args:
            X (xr.DataArray): DataArray of data that needs seasonality removed.

        Returns:
            self: 
        """
        
        doy = X.coords[self.time_dim].dt.dayofyear
        
        # Calculate seasonal mean
        means = X.groupby(doy).mean()
        doy_axis = means.dims.index('dayofyear')
        self.mean_fft = FFTSmoother(axis=doy_axis, fft_keep_frac=0.03).fit(means.values)
        self.seasonal_dims = means.dims
        self.seasonal_coords = means.coords
        
        if self.div_std: 

            # Calculate seasonal std
            stds = X.groupby(doy).std()
            self.std_fft = FFTSmoother(axis=doy_axis, fft_keep_frac=0.03).fit(stds.values)
        
        return self

    def transform(self, X: xr.DataArray) -> xr.DataArray: 
        """Removes seasonality.

        Args:
            X (xr.DataArray): DataArray of data that needs sesonality removed.

        Returns:
            xr.DataArray: DataArray of data with seasonality removed. 
        """
        
        doy = X.coords[self.time_dim].dt.dayofyear
        
        # Subtract the mean
        tfm = X - self.get_means().sel(dayofyear=doy).ffill(dim=self.time_dim)

        if self.div_std: 
            # Divide by std
            tfm = tfm / self.get_stds().sel(dayofyear=doy).ffill(dim=self.time_dim)
        
        return tfm
    
    def inverse_transform(self, X: xr.DataArray) -> xr.DataArray: 
        """Restores seasonality.

        Args:
            X (xr.DataArray): DaraArray of data with seasonality removed. 

        Returns:
            xr.DataArray: DataArray of data with seasonality restored. 
        """
        
        doy = X.coords[self.time_dim].dt.dayofyear

        if self.div_std: 
            # Multiply by std
            tfm = X * self.get_stds().sel(dayofyear=doy).ffill(dim=self.time_dim)

        else: 
            tfm = X.copy()

        # Add the mean
        tfm = tfm + self.get_means().sel(dayofyear=doy).ffill(dim=self.time_dim)

        return tfm

    def get_means(self) -> xr.DataArray:
        """Get DataArray of seasonal means.

        Returns:
            xr.DataArray: DataArray of seasonal means. 
        """

        means = xr.DataArray(
            data=self.mean_fft.get_smoothed(), 
            coords=self.seasonal_coords, 
            dims=self.seasonal_dims
        )
        
        return means

    def get_stds(self) -> xr.DataArray:
        """Get DataArray of seasonal stds.

        Returns:
            xr.DataArray: DataArray of seasonal stds. 
        """

        if self.div_std: 
            stds = xr.DataArray(
                data=self.std_fft.get_smoothed(), 
                coords=self.seasonal_coords, 
                dims=self.seasonal_dims
            )
            
        else: 
            raise ValueError("Method only valid when div_std=True.")

        return stds


class TrendRemover: 
    
    def __init__(self, deg: int = 1, time_dim: str = 'time'): 
        """Use polynomials to remove the time trend in a DataArray. 
        Will work for DataArrays with 1, 2, or 3 non-time dimensions. 

        Args:
            deg (int, optional): _description_. Defaults to 1.
            time_dim (str, optional): Name of time dimension in DataArray. 
                Defaults to 'time'.
        """
        self.deg = deg
        self.time_dim = time_dim
        self.poly = None
        
    def fit(self, X: xr.DataArray): 
        """Fit trend polynomials.

        Args:
            X (xr.DataArray): DataArray of data to be detrended. 

        Returns:
            self:
        """
        self.poly = X.polyfit(dim=self.time_dim, deg=self.deg, skipna=True)
        return self

    def get_trend(self, X: xr.DataArray) -> xr.DataArray: 
        """Calculate trend

        Args:
            X (xr.DataArray): Array along which to evaluate trend. 

        Returns:
            xr.DataArray: _description_
        """
        return xr.polyval(X.coords[self.time_dim], self.poly.polyfit_coefficients)
        
    def transform(self, X: xr.DataArray) -> xr.DataArray: 
        """Remove trend. 

        Args:
            X (xr.DataArray): DataArray with trend in tact. 

        Returns:
            xr.DataArray: DataArray with trend removed. 
        """
        return X - self.get_trend(X)
    
    def inverse_transform(self, X: xr.DataArray) -> xr.DataArray: 
        """Restore trend. 

        Args:
            X (xr.DataArray): DataArray with trend removed. 

        Returns:
            xr.DataArray: DataArray with trend restored. 
        """
        return X + self.get_trend(X)              

