import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
import pickle


class RMSELoss(nn.Module):
    """
    Class for RMSE loss as pytorch losses
    """
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
    """
    Class for early stopping model during training.
    Args:
        patience (int) : number of epochs with no improvement after which training will be stopped
        min_delta (float) : change to qualify as improvement
        start_from_epoch (int) : number of epoches not to monitore loss
        path (string) : path for saving best model if not None
    """
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
    """
    MAPE metric (mean absolute percentage error)
    """
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
    """
    Cross-validation splitter with fixed test fold. Train sample is several folds just before test.
    Args:
        n_splits (int) : number of splits (n_folds - 1)
        max_train_size_in_folds (int) : minimum number of folds in train
        test_start_index (int) : index in dataset for start test fold. If None - test fold is the last one
        test_end_index (int) : index in dataset for end test fold.
    """
    def __init__(self, n_splits=5, test_start_index=None, test_end_index=None, train_size_in_folds=None):
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
    """
    Ordinary cross-validation scheme for time series. Fixed sizes of train and test, test fold follows train.
    Args:
        n_splits (int) : number of folds - 1
        train_size_in_folds (int) : fixed number of folds in train
    """
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
                
def write_table(R, dataset_x, dataset_y, soft_index, folder=None, mask_train=None, mask_valid=None, mask_test=None, path=None, y_scaler=None, num_fold=0):
    """
    Writing results table with prediction target on dataset_x.
    Args:
        R : Regressor model
        dataset_x (torch.tensor) : x data
        dataset_y (torch.tensor) : target data
        soft_index (int) : index of soft sensor
        folder (string) : where to save results. Can be None only with path is None.
        mask_train (np.array) : mask with the indices of train folds
        mask_valid (np.array) : mask with the indices of valid folds. Can be None.
        mask_test (np.array) : mask with the indices of test folds
        path (string) : where to save table. If None, doesnt save
        y_scaler (MyStandartScaler) : scaler for target. Can be None

    """
    y = dataset_y.clone().cpu().detach().numpy()
    y_pred = R.apply_model(dataset_x, y_scaler=y_scaler)
    mask = np.zeros(dataset_x.shape[0])
    mask[:] = -1 
    mask[mask_train] = 0 #TRAIN
    if mask_valid is not None:
        mask[mask_valid] = 1 # VALID
    mask[mask_test] = 2 #TEST
    table = pd.DataFrame(np.stack((y, y_pred, mask), axis=1), columns= ['y', 'y_pred', 'mode'])
    if path is not None:
        path = f"{folder}/table_{soft_index}_{num_fold}---{path if path is not None else ''}.csv"
        table.to_csv(path)
    return table

def pack_results(results, soft_index, folder=None, save_to_file=True, path=None, type_of_test="CV", **kwargs):
    """
    Save results from cross-validation in one result pickle table
    Args:
         results : list of results tables
         soft_index (int) : index of sensor
         folder (string) : where to save pickle. Can be None only with save_to_file == False
         save_to_file (bool) : flag if save to file
         path (string) : suffix of file name. If None, current datetime is used.
         type_of_test (string) : type of cross-validation
    Return:
        Pickle file - dict with keys
            name : name of model 
            folds : list of results tables
            params: all parameters for model, cross-validation, data
            type_of_test: in ["CV", "fixed", "MyTimeSeries"] - type of cross-validation
            sensor (int) : index of sensor
    """
    if "splitter" in kwargs:
        kwargs.pop('splitter')
    kwargs['type_of_test'] = type_of_test
    name = ('cGAN' if 'name' not in kwargs else kwargs['name'])
    kwargs['name'] = name
    file = {"name": name, "sensor":soft_index, "params":kwargs, "folds":results, "type_of_test":type_of_test}
    print(file['folds'][-1].mean())
    if save_to_file:
        if path is  None:
            path = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        with open(f"{folder}/{soft_index}/{soft_index}_{type_of_test}_{path}.pickle", "wb") as f:
            pickle.dump(file, f)
            print("File saved")
    return file