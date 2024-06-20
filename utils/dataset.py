import numpy as np
from PIL import Image
import nibabel as nib
from torch.utils.data import Dataset
from torchvision import transforms

from .hparams import hparams
from utils.indexed_datasets import IndexedDataset
from utils.matlab_resize import imresize
import os
import pdb
import sys
import torch
from os.path import join

class SRDataSet(Dataset):
    def __init__(self, prefix='train'):
        self.hparams = hparams
        self.data_dir = join(hparams['binary_data_dir'], prefix)
        self.prefix = prefix
        self.task = hparams['task']
        
        self.X = os.listdir(self.data_dir)

        self.len = len(self.X)
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        assert hparams['data_interp'] in ['bilinear', 'bicubic']
        self.data_augmentation = hparams['data_augmentation']
        
        #if self.task == 'srdiff':
        with open('configs/OASIS3_participant_data.txt') as f:
            lines = f.readlines()
            lines = [l.split("\t") for l in lines]
            
            lines = [l[:3] for l in lines]
            
            lines = np.array(lines)[1:]
            
            id_patient = lines[:, 0]
            days_to_visit = lines[:, 1].astype(int)
            age_at_visit = lines[:, 2].astype(float)
            
            _, idxs = np.unique(id_patient, return_index=True)

            self.id_patient  = id_patient[idxs]
            self.age_first_visit  = age_at_visit[idxs]
            self.day_first_visit  = days_to_visit[idxs]        

        #BUILDING DICT CONTAINING CONDITIONS DATA
        self.patients_conditions = {}
        with open('configs/OASIS3_patients_condition.txt') as f:
            lines = f.readlines()
            lines = [l.split("\t") for l in lines]
            lines = [l[:4] for l in lines]
            lines = np.array(lines)[1:]
            
            id_patient = lines[:, 0]
            days_to_visit = lines[:, 1].astype(int)
            age_at_visit = lines[:, 2].astype(float)
            condition = lines[:, 3].astype(int)
            
            self.patients_conditions['id_patient'] = id_patient
            self.patients_conditions['days_to_visit'] = days_to_visit
            self.patients_conditions['age_at_visit'] = age_at_visit
            self.patients_conditions['condition'] = condition
        
    def _get_item(self, index):
        return self.X[index]

    def __getitem__(self, index):
        pair = self._get_item(index)
        hparams = self.hparams
        
        patient, mri_days, mri_days_next = pair.split("_")
        patient = "OAS" + patient
        mri_days = int(mri_days)
        mri_days_next = int(mri_days_next)
        mri_list = os.listdir(os.path.join(self.data_dir, pair))

        new_width = 200
        new_height = 160

        #COMPUTE AGES PER MRI
        age_first_visit = self.age_first_visit[self.id_patient == patient]
        day_first_visit = self.day_first_visit[self.id_patient == patient]
        
        #EXCLUDE NOT PRESENT PATIENT IN THE FILE PARTICIPANT_DATA.txt
        if len(age_first_visit) < 1:
            return {
        'img_hr': torch.zeros((3,new_height,new_width)), 'img_lr': torch.zeros((3,new_height,new_width)), 'img_lr_up': torch.zeros((3,new_height,new_width)), "item_name": "empty", 'diff_ages':0, 'patient_condition':0, 'age':0
            }

        mri_days = mri_days - day_first_visit
        mri_days_next = mri_days_next - day_first_visit

        if mri_days <= 0 or mri_days_next <= 0:
            return {
        'img_hr': torch.zeros((3,new_height,new_width)), 'img_lr': torch.zeros((3,new_height,new_width)), 'img_lr_up': torch.zeros((3,new_height,new_width)), "item_name": "empty", 'diff_ages':0, 'patient_condition':0, 'age':0
            }
      
        #COMPUTE DIFFERENT IN AGES
        #mri_diff_ages = ((1/365.0) * mri_days) #+ age_first_visit #YEAR BASED
        mri_diff_ages_next = ((1/30.0) * (mri_days_next - mri_days)) #+ age_first_visit #MONTH BASED
        age = ((1/365.0) * mri_days) + age_first_visit
        
        #SELECT TWO CONSECUTIVES MRIs
        mri_list.sort()
        mri = mri_list[0]
        mri_next = mri_list[1]
        
        #READ CONDITION
        patient_conditions = self.patients_conditions['condition'][(self.patients_conditions['id_patient'] == patient)]
        if len(patient_conditions) > 0:
            patient_conditions = [patient_conditions[np.argmin(np.abs(np.array(patient_conditions)-mri_days[0]))]]
        
        if len(patient_conditions) < 1:
            return {
        'img_hr': torch.zeros((3,new_height,new_width)), 'img_lr': torch.zeros((3,new_height,new_width)), 'img_lr_up': torch.zeros((3,new_height,new_width)), "item_name": "empty", 'diff_ages':0, 'patient_condition':0, 'age':0
            }
        
        data = nib.load(os.path.join(self.data_dir, pair, mri)).get_fdata()
        data_next = nib.load(os.path.join(self.data_dir, pair, mri_next)).get_fdata()

        #Select random slice
        # rand_idx_slice = int(np.random.rand() * data.shape[2])
        img_hr = data_next[:, :, 90:93]#rand_idx_slice]
        if img_hr.max() > 0:
            img_hr *= 255.0 / img_hr.max()

        #Select random slice
        img_lr_up = data[:, :, 90:93]#rand_idx_slice]
        if img_lr_up.max() > 0:
            img_lr_up *= 255.0 / img_lr_up.max()
        
        img_hr = Image.fromarray(np.uint8(img_hr))
        # img_lr = Image.fromarray(np.uint8(img_lr_up))
        img_lr_up = Image.fromarray(np.uint8(img_lr_up))

        #Get center crop dimensions
        width, height = img_hr.size   # Get dimensions
        
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2   

        img_hr = img_hr.crop((left, top, right, bottom))
        # img_lr = img_lr.crop((left, top, right, bottom))
        img_lr_up = img_lr_up.crop((left, top, right, bottom))
                 
        # img_hr, img_lr, img_lr_up = [self.to_tensor_norm(x).float() for x in [img_hr, img_lr, img_lr_up]]
        img_hr, img_lr_up = [self.to_tensor_norm(x).float() for x in [img_hr, img_lr_up]]
        img_lr = img_lr_up.clone()

        return {
            'img_hr': img_hr, 'img_lr': img_lr, 'img_lr_up': img_lr_up, 'item_name':(patient + "_" + str(mri_days)+"_"+str(mri_days_next)), 'diff_ages':mri_diff_ages_next[0], 'patient_condition':patient_conditions[0], 'age':age[0]
        }

    def pre_process(self, img_hr):
        return img_hr

    def __len__(self):
        return self.len
