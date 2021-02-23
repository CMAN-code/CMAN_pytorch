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

class BaseModel():
    """
        BaseModel for autoregression
    """
    def __init__(self, opt, dataset):
        self.opt = opt
        self.device = torch.device(opt.device)
        self.dataset = dataset
        self.normal_class =  opt.normal_class
        self.best_auc = 0

    def save_weights(self, epoch):
        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save(self.netg.state_dict(),
                   '%s/netG_%d.pth' % (weight_dir, epoch+1))

    def train_one_epoch(self):
        self.netg.train()
        self.netg.close_one_step_grad()
        epoch_iter = 0
        self.dataset.train(self.normal_class)
        dataloader = DataLoader(dataset=self.dataset, batch_size=self.opt.batchsize, shuffle=True, drop_last=self.opt.droplast, num_workers=int(self.opt.workers))
        for data in tqdm(dataloader, leave= False, total=len(dataloader)):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            (x, _) = data
            
            self.x = x.to(self.device)
            
            self.optimize_params(self.x)
        
        self.netg.open_grad()
        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, self.opt.niter))
        print(">>>>>>>>>>loss:{}".format(self.error.cpu().item()))

    def train_two_epoch(self):
        self.netg.train()
        self.netg.close_two_step_grad()
        epoch_iter = 0
        self.dataset.train(self.normal_class)
        dataloader = DataLoader(dataset=self.dataset, batch_size=self.opt.batchsize, shuffle=True, drop_last=self.opt.droplast, num_workers=int(self.opt.workers))
        for data in tqdm(dataloader, leave= False, total=len(dataloader)):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            (x, _) = data
            
            self.x = x.to(self.device)
            
            self.optimize_params(self.x)

        self.netg.open_grad()
        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, self.opt.niter))
        print(">>>>>>>>>>loss:{}".format(self.error.cpu().item()))

    def train(self):
        ##
        # TRAIN
        self.total_steps = 0

        # Train for niter epochs.
        print(">> Training model %s." % self.name)
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one step
            self.train_one_epoch()

            # Train for two step
            self.train_two_epoch()

            if self.scheduler is not None:
                self.scheduler.step()

            #to-do

            if (self.epoch + 1) % self.opt.save_image_freq == 0:
                img_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'image')
                if not os.path.exists(img_dir): os.makedirs(img_dir)
                save_image(torch.cat((self.x, self.x_r), dim=0), os.path.join(img_dir, f"epoch_{(self.epoch + 1)}.png"))

            if (self.epoch + 1) % self.opt.print_freq == 0:
                res = self.test()
                if res > self.best_auc:
                    self.best_auc = res
                    self.save_weights(10000)
            if (self.epoch + 1) % self.opt.save_model_freq == 0:
                    self.save_weights(self.epoch)

        print(">> Training model %s.[Done]" % self.name)
        return self.best_auc

    @torch.no_grad()
    def test(self):
        ##
        # Test
        # First we need a run on validation, to compute
        # normalizing coefficient of the Novelty Score (Eq.9)
        self.netg.eval()

        min_llk, max_llk, min_rec, max_rec = self.compute_normalizing_coefficients()
        # 临时解决cuda内存溢出问题
        batch_size  = self.opt.batchsize
        self.dataset.test(self.normal_class)
        
        loader = DataLoader(self.dataset)

        sample_llk = np.zeros(shape=(len(loader),))
        sample_rec = np.zeros(shape=(len(loader),))
        sample_y = np.zeros(shape=(len(loader),))

        print(f'Computing scores for {self.dataset}')
        for i, (x, y) in tqdm(enumerate(loader), leave= False, total=len(loader)):
            x = x.to(self.device)
            
            x_r, z, z_dist = self.netg(x)

            self.loss(x, x_r, z, z_dist)

            sample_llk[i] = - self.loss.autoregression_loss
            sample_rec[i] = - self.loss.reconstruction_loss
            sample_y[i] = y.item()

        # Normalize scores
        sample_llk = normalize(sample_llk, min_llk, max_llk)
        sample_rec = normalize(sample_rec, min_rec, max_rec)
        # Compute the normalized novelty score
        sample_ns = novelty_score(sample_llk, sample_rec)

        # Compute AUROC for this class
        this_class_metrics = [
            roc_auc_score(sample_y, sample_llk),  # likelihood metric
            roc_auc_score(sample_y, sample_rec),  # reconstruction metric
            roc_auc_score(sample_y, sample_ns)    # novelty score
        ]
        oc_table = self.empty_table
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
            x_r, z, z_dist = self.netg(x)

            self.loss(x, x_r, z, z_dist)
            
            sample_llk[i] = - self.loss.autoregression_loss
            sample_rec[i] = - self.loss.reconstruction_loss

        return sample_llk.min(), sample_llk.max(), sample_rec.min(), sample_rec.max()

    @property
    def empty_table(self):
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