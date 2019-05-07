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
    def __init__(self, root_dir, transform=None):
        self.root_dir = os.path.expanduser(root_dir)
        self.transform = transform
        
        LGG_list = sorted(os.listdir(os.path.join(self.root_dir, 'LGG')))
        LGG_list = list(map(lambda x: 'LGG/' + x, LGG_list))
        HGG_list = sorted(os.listdir(os.path.join(self.root_dir, 'HGG')))
        HGG_list = list(map(lambda x: 'HGG/' + x, HGG_list))

        # label 0 denotes LGG, label 1 denotes HGG
        labels = [0] * len(LGG_list) + [1] * len(HGG_list)
        scan_list = LGG_list + HGG_list

        assert len(scan_list) == len(labels)

        self.sample_list = list(zip(scan_list, labels))

    
    def __len__(self):
        return len(self.sample_list)


    def __getitem__(self, idx):
        case_dir = os.path.join(self.root_dir, self.sample_list[idx][0])
        case_name = self.sample_list[idx][0][4:]
        t1 = nib.load(os.path.join(case_dir, case_name + '_t1.nii.gz')).get_data()
        t1ce = nib.load(os.path.join(case_dir, case_name + '_t1ce.nii.gz')).get_data()
        t2 = nib.load(os.path.join(case_dir, case_name + '_t2.nii.gz')).get_data()
        flair = nib.load(os.path.join(case_dir, case_name + '_flair.nii.gz')).get_data()

        assert t1.shape == (240, 240, 155)
        assert t1ce.shape == (240, 240, 155)
        assert t2.shape == (240, 240, 155)
        assert flair.shape == (240, 240, 155)

        sc = np.array([t1, t1ce, t2, flair])
        assert sc.shape == (4, 240, 240, 155)

        sample = (sc, np.array([self.sample_list[idx][1]]))

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