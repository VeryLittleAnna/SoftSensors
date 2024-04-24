import torch
import numpy as np
import torch.nn as nn

class GeneratorBaseClass(nn.Module):
    """ 
    Base class for Generator models. 
    Real labels is defined as 1.
    Args:
        z-shape (int or tuple) : shape of noize
    """
    def __init__(self, z_shape=1):
        super().__init__()
        self.z_shape = (tuple([z_shape]) if isinstance(z_shape, int) else z_shape)
        
    def train_(self, z, y, D, G_loss, G_opt, labels_true):
        """
        Generator is trained to minimize the discriminator's error, aiming for the discriminator to classify its outputs as real.
        """
        G_opt.zero_grad()
        x_g = self.forward(z, y)
        D_pred_fake = D(x_g, y)
        loss_G = G_loss(D_pred_fake, labels_true)
        loss_G.backward()
        G_opt.step()
        return loss_G.cpu().detach().numpy()
    
    def eval_(self, z, y, D, G_loss, labels_true):
        """
        Generator validation is checking that the discriminator classifies the generator's outputs as real.
        """
        with torch.no_grad():
            x_g = self.forward(z, y)
            D_pred_fake = D(x_g, y=y)
            loss_G = G_loss(D_pred_fake, labels_true)
            return loss_G.cpu().detach().numpy()
    def get_noise(self, n):
        """
        Getting normal noize with defined shape.
        Args: 
            n (int) - size of batch or dataset
        """
        sz = tuple([n]) + self.z_shape
        return torch.normal(0, 1, size=sz, dtype=torch.float32)
    @property
    def device(self):
        return next(self.parameters()).device
        
class DiscriminatorBaseClass(nn.Module):
    """
    Base class for conditional discrimimator models.
    Real labels is defined as 1.
    """
    def __init__(self):
        super().__init__()

    def train_(self, z, x, y, G, R, D_loss, D_opt, labels_true, labels_fake):
        """
        Discriminator is trained to distinguish real data from fake data. It use condition - target value.
        """
        D_opt.zero_grad()
        x_g = G(z, y)
        y_g = R(x_g).flatten()
        D_pred = self.forward(torch.cat((x, x_g), dim=0), y=torch.cat((y, y_g), dim=0))
        labels = torch.cat((labels_true, labels_fake), dim=0)
        loss_D = D_loss(D_pred, labels)
        loss_D.backward()
        D_opt.step()
        return loss_D.cpu().detach().numpy()
    
    def eval_(self, z, x, y, G, R, D_loss, labels_true, labels_fake):
        """
        Discriminator validation is checking that is classifies the generator's outputs as fake and real data as real.
        """
        with torch.no_grad():
            x_g = G(z, y)
            y_g = R(x_g).flatten()
            D_pred = self.forward(torch.cat((x, x_g), dim=0), y=torch.cat((y, y_g), dim=0))
            labels = torch.cat((labels_true, labels_fake), dim=0)
            loss_D = D_loss(D_pred, labels)
            return loss_D.cpu().detach().numpy()
    @property
    def device(self):
        return next(self.parameters()).device
        
class RegressorBaseClass(nn.Module):
    """
    Base class for regressor models.
    """
    def __init__(self):
        super().__init__()
    
    def train_(self, x, y, R_loss, R_opt):
        """
        Training step for given loss and optimizer.
        """
        R_opt.zero_grad()
        y_pred = self.forward(x)
        loss_R = R_loss(y_pred, y)
        loss_R.backward()
        R_opt.step()
        return loss_R.cpu().detach().numpy()
    
    def eval_(self, x, y, R_loss):
        """
        Validation step for given loss.
        """
        with torch.no_grad():
            y_pred = self.forward(x)
            loss_R = R_loss(y_pred, y)
            return loss_R.cpu().detach().numpy()
        
    def apply_model(self, data_x, y_scaler=None):
        """
        Applying model to data without loss checking. If scaler is not None, it performs inverse target data transforming.
        """
        self.eval()
        with torch.no_grad():
            if not torch.is_tensor(data_x):
                data_x = torch.tensor(data_x)
            self.to(data_x.device)
            inputs = data_x.float()
            outputs = self.forward(inputs).cpu().detach().numpy()
        if y_scaler is not None:
            outputs = y_scaler.inverse_transform(outputs)
        return outputs
    @property
    def device(self):
        return next(self.parameters()).device


class Generator(GeneratorBaseClass):
    """
    MLP-based Generator model (Multilayer perceptron).
    Args:
        input_shape (int or tuple) : shape of noize
        outout_shape (int or tuple) : shape of syntetic data
    """
    def __init__(self, input_shape=1, output_shape=2):
        super().__init__(z_shape=input_shape)
        flatten_shape = np.prod(np.array(output_shape))
        self.layers = nn.Sequential(
            nn.Linear(input_shape + 1, 512),
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
        if y is not None:
            y = y[..., None]
            z = torch.cat((z, y), dim=-1)
            x = self.layers(z)
        return x
    def __str__(self):
        return "MLP-Generator"
        
class Discriminator(DiscriminatorBaseClass):
    """
    MLP-based Discriminator class (Multilayer perceptron).
    Args:
        input_shape (int or tuple) : shape of data samples 
        conditional (bool) : flag if discriminator is conditional and uses target value
    """
    def __init__(self, input_shape=2, conditional=True):
        super().__init__()
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
        

class Regressor(RegressorBaseClass):
    """
    MLP-based Regressor class (Multilayer perceptron).
    Args:
        input_shape (int or tuple) : shape of data samples 
    """
    def __init__(self, input_shape=1):
        super().__init__()
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
    
class LSTM_Regressor(RegressorBaseClass):
    """
    LSTM-based Regressor class (Long Short-Term Memory).
    Args:
        hidden_size (int) : number of hidden units in LSTM
        input_size (int or tuple) : shape of data samples 
        n_layers (int) : number of layers in LSTM
    """
    def __init__(self, hidden_size=50, input_size=1, n_layers=1):
        super().__init__()
        self.layer1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.layer2 = nn.Linear(in_features=hidden_size, out_features=1)
        

    def forward(self, x):
        x, _ = self.layer1(x) #out, cell
        x = x[:, -1, :]
        x = self.layer2(x)
        return x.reshape(-1,)
    
    
    
class RNN_Block(nn.Module):
    """
    Block of multilayer GRU+linear to construct models.
    Args:
        input_size (int) : shape of data samples 
        hidden_size (int) : number of hidden units in GRU
        output_size (int) : shape of output data
        n_layers (int) : number of layers in GRU
    """
    def __init__(self, input_size=57, hidden_size=20, output_size=1, n_layers=1):
        super(RNN_Block, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_out, _ = self.gru(x)
        output = self.linear(h_out)
        return output
    
class RnnGenerator(GeneratorBaseClass):
    """
    RNN-based Generator model (GRU-Block).
    Args:
        input_size (int or tuple) : shape of noize. 
        W (int) : size of time window. If None, then input_size must be tuple.
        hidden_size (int) : number of hidden units in GRU
        output_size (int) : shape of output data
        n_layers (int) : number of layers in GRU
    """
    def __init__(self, input_size=1, W=1, hidden_size=1, output_size=1, n_layers=1):
        super().__init__(z_shape=(W, input_size))
        # Z = (inp)-> (W, hid) -> (W, out)
        if isinstance(input_size, int):
            self.W, self.Q = W, input_size + 1
        else:
            self.W, self.Q = None, input_size[-1] + 1
        self.rnn_layers = RNN_Block(input_size=self.Q,
                                        hidden_size=hidden_size,
                                        output_size=output_size,
                                        n_layers=n_layers)

    def forward(self, z, y=None):
        y = y[..., None].repeat((1, z.shape[1]))[..., None]
        z = torch.cat((z, y), dim=2)
        output = self.rnn_layers(z)
        return output
    
    def __str__(self):
        return "RNN-Generator"

class RnnDiscriminator(DiscriminatorBaseClass):
    """
    RNN-based Discriminator model (GRU-Block).
    Args:
        input_size (int or tuple) : shape of noize. 
        W (int) : size of time window. If None, Discriminator return W values.
        hidden_size (int) : number of hidden units in GRU
        n_layers (int) : number of layers in GRU
    """
    def __init__(self, input_size=1, hidden_size=1, n_layers=1, W=None):
        super().__init__()
        self.rnn_layers = RNN_Block(input_size=input_size + 1,
                                hidden_size=hidden_size,
                                output_size=hidden_size,
                                n_layers=n_layers)
        if W is not None:
            self.out = nn.Sequential(
                        nn.Flatten(start_dim=1, end_dim=-1),
                        nn.Linear(hidden_size * W, 1)
            )
        else:
            self.out = nn.Linear(hidden_size, 1)

    def forward(self, x, y=None):
        y = y[..., None].repeat((1, x.shape[1]))[..., None]
        x = torch.cat((x, y), dim=2)
        x = self.rnn_layers(x)
        output = torch.sigmoid(self.out(x)).flatten()
        return output