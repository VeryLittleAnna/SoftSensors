import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

class MyStandardScaler():
    def __init__(self, dims=1):
        self.dims = dims
        self.eps = 1e-15
        
    def fit(self, data):
        assert(len(data.shape) == self.dims)
        self.mean = np.mean(data, axis=-1)
        self.std = np.std(data, axis=-1)

    def fit_transform(self, data):
        assert(len(data.shape) == self.dims)
        self.mean = np.mean(data, axis=-1)
        self.std = np.std(data, axis=-1)
        if self.dims == 1:
            res = (data - self.mean) / np.maximum(self.eps, self.std)
        elif self.dims == 2:
            res = (data - self.mean[:, None]) / np.maximum(self.eps, self.std[:, None])
        return res
    def inverse_transform(self, data):
        assert(len(data.shape) == self.dims)
        if self.dims == 1:
            return data * self.std + self.mean
        else:
            return data * self.std[:, None] + self.mean[:, None]

class Dataset:
    
    def __init__(self, data, soft_data, delay=0, K=60, diff=False, periods=None):
        self.K = K
        data = data.groupby(data.index // K).mean().to_numpy() #усреднение
        mask = ~(soft_data.isna())
        self.mask_inds = np.where(mask)[0]
        self.mask_inds_values = (self.mask_inds - delay) // K
        self.mask_inds_values = self.mask_inds_values[self.mask_inds_values >= 0]
        self.sc_x = StandardScaler()
        if diff:
            data = np.diff(data, axis=0)
        self.dataset_x = self.sc_x.fit_transform(data)
#         self.dataset_x = data
        self.soft_data = soft_data[self.mask_inds].to_numpy()
        if periods is not None:
            with open(periods, "rb") as f:
                self.periods = np.load(f)
        else:
            self.periods = np.ones(self.dataset_x.shape[0], dtype=bool)
               
      
    def window_view(self, W=5, scale_target=True):
#         mask_inds_values = self.mask_inds_values[self.mask_inds_values < self.dataset_x.shape[0] - W + 1]
        mask_inds_values = self.mask_inds_values - W + 1
        mask_inds_values = mask_inds_values[mask_inds_values >= 0]
        periods = self.periods[self.mask_inds_values]
        periods = periods[periods >= W - 1]
        dataset_x = sliding_window_view(self.dataset_x, (W, self.dataset_x.shape[-1]))[:, 0][mask_inds_values]
        dataset_y = self.soft_data.copy()
        sc_y = None
        if scale_target:
            sc_y = MyStandardScaler(dims=1)
            dataset_y = sc_y.fit_transform(dataset_y)
        dataset_y = dataset_y[-dataset_x.shape[0]:][periods]
        dataset_x = dataset_x[periods]
        print(dataset_x.shape, dataset_y.shape)
        return dataset_x, dataset_y, sc_y

    def inverse_transform(self, X=None, y=None):
        if X is not None:
            return self.sc_x.inverse_transform(X)
        if y is not None:
            if self.sc_y is not None:
                return self.sc_y.inverse_tranform(y)
            else:
                return y