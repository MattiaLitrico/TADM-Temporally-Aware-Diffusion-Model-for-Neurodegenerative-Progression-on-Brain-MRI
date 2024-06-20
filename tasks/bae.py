import os.path
import torch
from models.bae_model import BAEModel
from models.diffsr_modules import RRDBNet
from tasks.trainer import Trainer
from utils.hparams import hparams
from utils.utils import load_ckpt
from tasks.srdiff_oasis import OasisDataSet
import pdb


class BAETrainer(Trainer):
    def build_model(self):        
        if hparams['use_rrdb']:
            rrdb = RRDBNet(3, 3, hparams['rrdb_num_feat'], hparams['rrdb_num_block'],
                           hparams['rrdb_num_feat'] // 2)
            
            if hparams['rrdb_ckpt'] != '' and os.path.exists(hparams['rrdb_ckpt']):
                load_ckpt(rrdb, hparams['rrdb_ckpt'])
        else:
            rrdb = None
        self.model = BAEModel(
            rrdb_net=rrdb
        )
        self.global_step = 0
        return self.model

    def sample_and_test(self, sample, img_sr=None, diff_ages=None):
        ret = {k: 0 for k in self.metric_keys}
        ret['n_samples'] = 0
        
        img_lr = sample['img_lr']

        #Only during pretraining
        if img_sr is None:
            img_sr = sample['img_hr']

        if diff_ages is None:
            diff_ages = sample['diff_ages']
        
        time_pred, loss = self.model.sample(img_lr, img_sr, diff_ages=diff_ages)
        
        return time_pred, loss

    def build_optimizer(self, model):
        params = list(model.named_parameters())
        if not hparams['fix_rrdb']:
            params = [p for p in params if 'rrdb' not in p[0]]
        params = [p[1] for p in params]
        return torch.optim.Adam(params, lr=hparams['lr'])

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)

    def training_step(self, batch):
        img_hr = batch['img_hr']
        img_lr = batch['img_lr']
        
        diff_ages = batch['diff_ages']
        
        time_pred, losses = self.model(img_lr, img_hr, diff_ages=diff_ages)
        total_loss = losses
        return losses, total_loss, {'img_hr':img_hr, 'img_lr':img_lr, 'img_lr_up':img_lr, 'time_pred':time_pred, 'diff_ages':diff_ages}

class BAEOasisTask(BAETrainer):
    def __init__(self):
        super().__init__()
        self.dataset_cls = OasisDataSet