import pandas as pd
import torch
import numpy as np
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_shape=1, output_shape=2):
        super(Generator, self).__init__()
        flatten_shape = np.prod(np.array(output_shape))
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, flatten_shape),
        )
        if not isinstance(output_shape, int):
            self.layers.add_module("unflatten", nn.Unflatten(-1, output_shape))
    
    def forward(self, z, y=None):
        # z - noise
        # print(z.shape, y.shape, z.dtype, y.dtype)
        if y is not None:
            z = torch.stack((z, y), dim=1)
        x = self.layers(z)
        return x
        
class Discriminator(nn.Module):
    def __init__(self, input_shape=2, conditional=True):
        super(Discriminator, self).__init__()
        flatten_shape = np.prod(np.array(input_shape)) + (1 if conditional else False)
        self.layers = nn.Sequential(
            nn.Linear(flatten_shape, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )
        
    def forward(self, x, y=None):
        if y is not None:
            x = torch.cat((x.flatten(1, -1), y[:, None]), dim=1)
        x = self.layers(x).flatten()
        return x
        

class Regressor(nn.Module):
    def __init__(self, input_shape=1):
        super(Regressor, self).__init__()
        flatten_shape = np.prod(np.array(input_shape))
        self.layers = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1), # (batch_size, N, Q) or (batch, N, LAG, Q)
            nn.Linear(flatten_shape, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        x = self.layers(x).flatten()
        return x