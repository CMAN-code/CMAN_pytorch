from train_helpers.one_class import BaseModel
from models.MEM_mnist import MEMMNIST
from models.loss_functions import LSALoss
from torch import optim
import torch.nn as nn
import torch
from os.path import join
from typing import Tuple

import numpy as np
import torch
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.loss_functions import LSALoss
from utils import novelty_score
from utils import normalize
import os
from torch import optim
import time
from torchvision.utils import save_image
from models.loss_functions.att_loss import AttLoss


def load_model_state_dict(model, state_dict):
    model_state_dict = model.state_dict()
    new_params = model_state_dict.copy()
    for k, v in state_dict.items():
        new_params[k] = v
    model.load_state_dict(new_params)

class MEM_Mnist_Trainer(BaseModel):
    """Mnist_Trainer
 Class
    """

    @property
    def name(self): return 'MEM_Mnist_Trainer'

    def __init__(self, opt, dataset):
        super(MEM_Mnist_Trainer, self).__init__(opt, dataset)

        self.epoch = 0
        self.times = []
        self.total_steps = 0

        self.loss = LSALoss(cpd_channels=100)
        self.w_loss = AttLoss()

        self.netg  = MEMMNIST(input_shape=self.dataset.shape, code_length=64, cpd_channels=100, mem_dim=opt.MemDim, shrink_thres=1.0/opt.MemDim).to(self.device)
        print("model:")
        print(self.netg)

        self.optimizer = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max =self.opt.niter)

        if self.opt.resume != "":
            print("\nLoading pre-trained networks.")
            load_model_state_dict(self.netg, torch.load(os.path.join(opt.resume)))
            self.optimizer = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            print("\tDone.\n")

    def forward(self, x):
        self.x_r, self.z, self.z_dist, self.att = self.netg(x)

    def backward(self, x):
        self.error = self.loss(x, self.x_r, self.z, self.z_dist) + self.opt.alpha * self.w_loss(self.att)
        self.error.backward()

    def optimize_params(self, x):
        self.forward(x)
        self.optimizer.zero_grad()
        self.backward(x)
        self.optimizer.step()

    @torch.no_grad()
    def test(self):
        self.netg.eval()

        min_llk, max_llk, min_rec, max_rec= self.compute_normalizing_coefficients()
        batch_size  = self.opt.batchsize
        self.dataset.test(self.normal_class)
        
        loader = DataLoader(self.dataset)

        sample_llk = np.zeros(shape=(len(loader),))
        sample_rec = np.zeros(shape=(len(loader),))

        sample_y = np.zeros(shape=(len(loader),))

        print(f'Computing scores for {self.dataset}')
        for i, (x, y) in tqdm(enumerate(loader), leave= False, total=len(loader)):
            x = x.to(self.device)
            
            x_r, z, z_dist, att = self.netg(x)

            self.loss(x, x_r, z, z_dist)

            sample_llk[i] = - self.loss.autoregression_loss
            sample_rec[i] = - self.loss.reconstruction_loss

            sample_y[i] = y.item()

        sample_llk = normalize(sample_llk, min_llk, max_llk)
        sample_rec = normalize(sample_rec, min_rec, max_rec)

        sample_ns = novelty_score(sample_llk, sample_rec)

        this_class_metrics = [
            roc_auc_score(sample_y, sample_llk), 
            roc_auc_score(sample_y, sample_rec), 
            roc_auc_score(sample_y, sample_ns) 
        ]
        oc_table = self.now_table
        oc_table.add_row([self.best_auc]+this_class_metrics)
        print(oc_table)
        file_name = os.path.join(self.opt.expr_dir, 'opt.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('\n')
            opt_file.write(f'------------ {self.epoch} -------------')
            opt_file.write('\n')
            opt_file.write(str(oc_table))
        return this_class_metrics[2]

    @torch.no_grad()
    def compute_normalizing_coefficients(self):
        # type: (int) -> Tuple[float, float, float, float]
        """
        Computes normalizing coeffients for the computation of the Novelty score (Eq. 9-10).
        :param cl: the class to be considered normal.
        :return: a tuple of normalizing coefficients in the form (llk_min, llk_max, rec_min, rec_max).
        """
        self.netg.eval()

        self.dataset.val(self.normal_class)
        loader = DataLoader(self.dataset)

        sample_llk = np.zeros(shape=(len(loader),))
        sample_rec = np.zeros(shape=(len(loader),))

        for i, (x, y) in enumerate(loader):
            x = x.to(self.device)
            x_r, z, z_dist, att = self.netg(x)

            self.loss(x, x_r, z, z_dist)
            sample_llk[i] = - self.loss.autoregression_loss
            sample_rec[i] = - self.loss.reconstruction_loss


        return sample_llk.min(), sample_llk.max(), sample_rec.min(), sample_rec.max()

    @property
    def now_table(self):
        # type: () -> PrettyTable
        """
        Sets up a nice ascii-art table to hold results.
        This table is suitable for the one-class setting.
        :return: table to be filled with auroc metrics.
        """
        table = PrettyTable()
        table.field_names = ['Best_AUC', 'AUROC-LLK', 'AUROC-REC', 'AUROC-NS']
        table.float_format = '0.3'
        return table