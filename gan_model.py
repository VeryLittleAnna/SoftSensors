import pandas as pd
import torch
import numpy as np
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_shape=1, output_shape=2):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_shape),
        )
    
    def forward(self, z, y=None):
        # z - noise
        # print(z.shape, y.shape, z.dtype, y.dtype)
        if y is not None:
            z = torch.stack((z, y), dim=1)
        x = self.layers(z)
        return x
        
class Discriminator(nn.Module):
    def __init__(self, input_shape=2):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 256),
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
            x = torch.cat((x, y[:, None]), dim=1)
        x = self.layers(x).flatten()
        return x
        

class Regressor(nn.Module):
    def __init__(self, input_shape=1):
        super(Regressor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 512),
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