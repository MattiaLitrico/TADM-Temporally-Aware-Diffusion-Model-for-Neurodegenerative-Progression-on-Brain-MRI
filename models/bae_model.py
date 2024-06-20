from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from .module_util import default
from utils.sr_utils import SSIM, PerceptualLoss
from utils.hparams import hparams
import pdb

class BAEModel(nn.Module):
    def __init__(self,rrdb_net):
        super().__init__()
        
        self.cond_proj = nn.ConvTranspose2d(hparams['rrdb_num_feat'] * ((hparams['rrdb_num_block'] + 1) // 3),
                                            hparams['hidden_size'], hparams['sr_scale'] * 1, hparams['sr_scale'],
                                            hparams['sr_scale'] // 2)

        # condition net
        self.rrdb = rrdb_net
        self.bae = nn.Sequential(nn.Conv2d(hparams['hidden_size']*2, hparams['hidden_size'], 2, stride=2), nn.ReLU(), \
        nn.Conv2d(hparams['hidden_size'], hparams['hidden_size'], 2, stride=2), nn.ReLU(), \
        nn.Conv2d(hparams['hidden_size'], hparams['hidden_size']//2, 2, stride=2), nn.ReLU(), \
        nn.Conv2d(hparams['hidden_size']//2, hparams['hidden_size']//4, 1, stride=1), nn.ReLU(), \
        nn.Conv2d(hparams['hidden_size']//4, 1, kernel_size=1, stride=1), nn.Dropout(p=0.8), \
        nn.Flatten(), nn.Linear(500, 1), nn.ReLU()
        )

    def forward(self, img_lr, img_hr, diff_ages=None):
        
        self.rrdb.eval()
        with torch.no_grad():
            _, cond_lr = self.rrdb(img_lr, True)
            _, cond_hr = self.rrdb(img_hr, True)

            cond_lr = self.cond_proj(torch.cat(cond_lr[2::3], 1))
            cond_hr = self.cond_proj(torch.cat(cond_hr[2::3], 1))

        time_pred = self.bae(torch.cat((cond_lr, cond_hr), dim=1))
        
        loss = F.l1_loss(time_pred, diff_ages.unsqueeze(1))
        
        return time_pred, loss

    @torch.no_grad()
    def sample(self, img_lr, img_sr, diff_ages=None):
        
        self.rrdb.eval()
        
        _, cond_lr = self.rrdb(img_lr, True)
        _, cond_sr = self.rrdb(img_sr, True)

        cond_lr = self.cond_proj(torch.cat(cond_lr[2::3], 1))
        cond_sr = self.cond_proj(torch.cat(cond_sr[2::3], 1))
        
        time_pred = self.bae(torch.cat((cond_lr, cond_sr), dim=1))
        
        loss = F.l1_loss(time_pred, diff_ages.unsqueeze(1))
        
        return time_pred, loss
