import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader


class cGAN_fitter:
    """
    Class for fitting Regressor with GAN.
    """
    def __init__(self, 
                G_opt=torch.optim.Adam,
                G_lr=1e-4,
                D_opt=torch.optim.Adam,
                D_lr=1e-4,
                R_opt=torch.optim.Adam,
                R_lr=1e-4,
                epoches=1, 
                n_gen=1, 
                n_dis=1, 
                n_reg=1, 
                train_on_real=True,
                device='cpu',
                batch_size=4,
                y_scaler=None,
                save_plots_test=False,
                **kwargs
                ):
        self.device = device
        self.train_on_real = train_on_real
        self.batch_size = batch_size
        self.epoches = epoches 
        self.n_gen = n_gen
        self.n_dis = n_dis 
        self.n_reg = n_reg
        self.y_scaler = y_scaler
        self.save_plots_test = save_plots_test
                
        #Generator
        self.G_loss = nn.BCELoss()
        self.G_opt = G_opt
        self.G_opt_params = {'lr':G_lr}
        #Discriminator
        self.D_loss = nn.BCELoss()
        self.D_opt = D_opt
        self.D_opt_params = {'lr':D_lr}
        #Regressor
        self.R_loss = nn.MSELoss()
        self.R_opt = R_opt
        self.R_opt_params = {'lr':R_lr}

    def validate(self, test_x, test_y, D=None, G=None, R=None, losses_log=None, label="valid"):
        """
        Validation:
            - Regressor - loss for prediction target
            - Generator - loss for Discriminator to identify that syntetic data is fake
            - Regressor - loss for Discriminator to identify that syntentic data is fake and real data is real
        Results losses append in dictionary losses_log
        """
        D.eval(); G.eval(); R.eval()
        D, G, R = [x.to(self.device) for x in (D, G, R)]
        with torch.no_grad():
            x = torch.tensor(test_x, dtype=torch.float32).to(self.device)
            y = torch.tensor(test_y, dtype=torch.float32).to(self.device)
            labels_true = torch.ones(x.shape[0], dtype=torch.float32).to(self.device)
            labels_fake = torch.zeros(x.shape[0], dtype=torch.float32).to(self.device)
            z = G.get_noise(x.shape[0]).to(self.device)
            loss_G = G.eval_(z, y, D, self.G_loss, labels_fake)
            loss_D = D.eval_(z, x, y, G, R,self.D_loss, labels_true, labels_fake)
            loss_R = R.eval_(x, y, self.R_loss)

            losses_log['Generator'][label].append(loss_G)
            losses_log['Discriminator'][label].append(loss_D)
            losses_log['Regressor'][label].append(loss_R)
            if self.save_plots_test is not None:
                y_pred = R.apply_model(test_x, scaler=self.y_scaler)
                plt.plot(test_y.cpu().detach().numpy(), color="blue")
                plt.plot(y_pred, color="orange")
                y_min, y_max = float(torch.min(test_y) - torch.std(test_y)), float(torch.max(test_y) + torch.std(test_y))
                plt.ylim((y_min, y_max))
                plt.savefig(save_plots_test + f"{epoch:03d}" + '.png', bbox_inches='tight')
                plt.title(f"{epoch:03d}")
                plt.clf()


    def fit(self, train_x, train_y, test_x, test_y, G=None, D=None, R=None):
        """
        Training process
        """
        G.to(self.device); R.to(self.device); D.to(self.device)
        G_opt = self.G_opt(G.parameters(), **self.G_opt_params)
        D_opt = self.D_opt(D.parameters(), **self.D_opt_params)
        R_opt = self.R_opt(R.parameters(), **self.R_opt_params)
        losses_log = {"Regressor":defaultdict(list), "Discriminator":defaultdict(list), "Generator":defaultdict(list)}
        train_x, train_y, test_x, test_y = [torch.tensor(tmp, dtype=torch.float32).to(self.device) for tmp in \
                                        (train_x, train_y, test_x, test_y)]
        dataloader = DataLoader(np.arange(train_x.shape[0]), batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epoches):
            D_run_loss, G_run_loss, R_run_loss_real, R_run_loss_fake = 0, 0, 0, 0
            D.train(); G.train(); R.train()
            #Discriminator
            for it in range(self.n_dis):
                for ids in dataloader:
                    x, y = train_x[ids], train_y[ids]
                    labels_true = torch.ones(x.shape[0], dtype=torch.float32).to(self.device)
                    labels_fake = torch.zeros(x.shape[0], dtype=torch.float32).to(self.device)
                    z = G.get_noise(x.shape[0]).to(self.device)
                    loss_D = D.train_(z, x, y, G, R, self.D_loss, D_opt, labels_true, labels_fake)
                    D_run_loss += loss_D
            #Generator
            for it in range(self.n_gen):
                for ids in dataloader:
                    x, y = train_x[ids], train_y[ids]
                    labels_true = torch.ones(x.shape[0], dtype=torch.float32).to(self.device)
                    labels_fake = torch.zeros(x.shape[0], dtype=torch.float32).to(self.device)
                    z = G.get_noise(x.shape[0]).to(self.device)
                    loss_G = G.train_(z, y, D, self.G_loss, G_opt, labels_true)
                    G_run_loss += loss_G
            #Regressor
            for it in range(self.n_reg):
                for ids in dataloader:
                    x, y = train_x[ids], train_y[ids]
                    labels_true = torch.ones(x.shape[0], dtype=torch.float32).to(self.device)
                    labels_fake = torch.zeros(x.shape[0], dtype=torch.float32).to(self.device)
                    z = G.get_noise(x.shape[0]).to(self.device)
                    if self.train_on_real:
                        loss_R_real = R.train_(x, y, self.R_loss, R_opt)
                        R_run_loss_real += loss_R_real
                    x_g = G(z, y)
                    loss_R_fake = R.train_(x_g, y, self.R_loss, R_opt)
                    R_run_loss_fake += loss_R_fake
            
            losses_log['Discriminator']['train'].append(D_run_loss / len(dataloader) / self.n_dis)
            losses_log['Generator']['train'].append(G_run_loss / len(dataloader) / self.n_gen)
            losses_log['Regressor']['train_fake'].append(R_run_loss_fake / len(dataloader) / self.n_reg)
            if self.train_on_real:
                losses_log['Regressor']['train_real'].append(R_run_loss_real / len(dataloader) / self.n_reg)

            #validation
            self.validate(test_x, test_y, D=D, G=G, R=R, losses_log=losses_log)
            if epoch % 10 == 0:
                print(f"EPOCH : {epoch}| Train loss: {losses_log['Regressor']['train_real'][-1]} | Valid loss {losses_log['Regressor']['valid'][-1]}")
        return D, G, R, losses_log
