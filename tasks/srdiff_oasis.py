import torchvision.transforms as transforms
from PIL import Image

from tasks.srdiff import SRDiffTrainer
from utils.dataset import SRDataSet
from utils.hparams import hparams
from utils.indexed_datasets import IndexedDataset
import os

class OasisDataSet(SRDataSet):
    def __init__(self, prefix='train'):
        super().__init__(prefix)
        preprocess_transforms = []
        if prefix == 'train' and self.data_augmentation:
            preprocess_transforms += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20, resample=Image.BICUBIC),
                
            ]
        self.pre_process_transforms = transforms.Compose(preprocess_transforms + [
            transforms.Resize((160, 160)),
        ])


    def _get_item(self, index):
        return self.X[index]

    def pre_process(self, img_hr):
        """
        Args:
            img_hr: PIL, [h, w, c]
        Returns: PIL, [h, w, c]
        """
        img_hr = self.pre_process_transforms(img_hr)
        return img_hr


class SRDiffOasis(SRDiffTrainer):
    def __init__(self):
        super().__init__()
        self.dataset_cls = OasisDataSet
