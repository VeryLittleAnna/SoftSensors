import torch.nn as nn
import torch
# import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit



class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, output, target):
        if not torch.is_tensor(output):
            output = torch.tensor(output).float()
        if not torch.is_tensor(target):
            target = torch.tensor(target).float()
        return torch.sqrt(self.mse(output, target))

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, start_from_epoch=0, path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.path = path
        self.min_loss_epoch_number = -1
        self.counter_not_better = 0
        self.start_from_epoch = start_from_epoch

    def early_stop(self, validation_loss, model=None):
        self.counter += 1   
        if self.counter <= self.start_from_epoch:
            return False
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.min_loss_epoch_number = self.counter
            self.counter_not_better = 0
            if self.path is not None and model is not None:
                torch.save(model.state_dict(), self.path)
        elif validation_loss > (self.min_validation_loss + self.min_delta) and self.counter > self.start_from_epoch:
            self.counter_not_better += 1
            if self.counter_not_better > self.patience:
                return True
        return False
    
    def restore_model(self, model):
        if self.path is None:
            raise("Best model haven't saved, no path")
        model.load_state_dict(torch.load(self.path))
        return model
    
def MAPE(target, output):
    target = np.maximum(np.abs(np.array(target)), 1e-15)
    output = np.array(output)
    mape = np.mean(np.abs(target - output) / target, axis=-1)
    return mape

def global_seed(seed: int) -> None:
    """
    Set global seed for reproducibility.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
class TestFixedSplitter(): 
    def __init__(self, n_splits=5, max_train_size=None, test_start_index=None, test_end_index=None, train_size_in_folds=None):
            self.max_train_size = max_train_size
            self.test_start_index = test_start_index
            self.test_end_index = test_end_index
            self.n_splits = n_splits
    def split(self, X, y=None):
        n_folds = self.n_splits + 1
        n_samples = X.shape[0]
        fold_size = n_samples // n_folds
        if self.test_start_index is not None:
             test_start_index = self.test_start_index
             test_indices = np.arange(self.test_start_index, self.test_end_index)
             train_indices = np.arange(0, self.test_start_index)
        else:
            test_start_index =  n_samples - n_samples // n_folds
            test_indices = np.arange(test_start_index, n_samples)
            train_indices = np.arange(0, test_start_index)
        print(test_start_index, fold_size)
        train_ends = [test_start_index - i * fold_size for i in range(self.n_splits)]
        for i, train_end in enumerate(train_ends):
             if i == len(train_ends) - 1:
                  yield (
                       train_indices[0:test_start_index], 
                       test_indices
                  )
             else:
               yield (
                  train_indices[max(0, train_end - fold_size):test_start_index],
                  test_indices
             )
       
       
        
class MyTimeSeriesSplitter():
    def __init__(self, n_splits=5, train_size_in_folds=1):
          self.train_size_in_folds = train_size_in_folds
          self.n_splits = n_splits
          
    def split(self, X, y=None):  
        splitter = TimeSeriesSplit(n_splits=self.n_splits,  max_train_size=int(self.train_size_in_folds/(self.n_splits + 1) * X.shape[0]))
        splitter = iter(splitter.split(X))
        for i in range(self.train_size_in_folds - 1):
            next(splitter); 
        for (train_inds, test_inds) in splitter:
             yield (train_inds, test_inds)