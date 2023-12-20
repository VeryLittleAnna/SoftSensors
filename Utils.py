import torch.nn as nn
import torch
# import pandas as pd
import numpy as np


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
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.min_loss_epoch_number = self.counter
            self.counter_not_better = 0
            if self.path is not None and model is not None:
                torch.save(model.state_dict(), self.path)
        elif validation_loss > (self.min_validation_loss + self.min_delta) and self.counter > self.start_from_epoch:
            self.counter_not_better += 1
            if self.counter_not_better >= self.patience:
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