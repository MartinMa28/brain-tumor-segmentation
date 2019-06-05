from torchvision import datasets, transforms
from torchvision.transforms import functional as F
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import os


class BRATS2018_3D(Dataset):
    """
    BraTS2018 classification dataset:
    LGG - 0, HGG - 1
    """
    def __init__(self, root_dir, data_set='train', transform=None):
        self.root_dir = os.path.expanduser(root_dir)
        self.transform = transform
        self.data_set = data_set
        self.base_dir = os.path.join(self.root_dir, 'CLS')
        self.case_dir = os.path.join(self.base_dir, self.data_set)
        
        with open(os.path.join(self.base_dir, self.data_set + '.txt')) as f:
            self.sample_list = [case.strip() for case in f.readlines()]

    
    def __len__(self):
        return len(self.sample_list)


    def __getitem__(self, idx):
        case_name = self.sample_list[idx]
        sc = np.load(os.path.join(self.case_dir, case_name + '_scan.npy'))
        grade = np.load(os.path.join(self.case_dir, case_name + '_grade.npy'))
        sample = (sc, grade)
        
        if self.transform:
            sample = self.transform(sample)

        return sample


class NormalizeBRATS3D():
    """
    Normalize BraTS2018 along axial, coronal, and sagittal axes
    """
    def __call__(self, sample):
        sc, label = sample

        mean = np.mean(sc, axis=(1, 2, 3), keepdims=True)
        std = np.std(sc, axis=(1, 2, 3), keepdims=True)

        no_zero_std = std + (std == 0).astype(np.int64) * np.ones(std.shape)

        sc = (sc - mean) / no_zero_std
        
        return sc, label


class CenterCropBRATS3D():
    """
    Center crop the BraTS from 4 * 240 * 240 * 155 to 4 * 224 * 224 * 144
    """
    def __call__(self, sample):
        sc, label = sample

        sc = sc[:, 40:-40, 48:-32, 3:-8]
        assert sc.shape == (4, 160, 160, 144)

        return sc, label