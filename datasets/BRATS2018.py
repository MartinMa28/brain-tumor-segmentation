from torchvision import datasets, transforms
from torchvision.transforms import functional as F
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from PIL import Image
import os


class BRATS2018(Dataset):
    def __init__(self, root_dir, grade='HGG', data_set='train', seg_type='et', scan_type='t1ce', transform=None):
        """
        root_dir: the directory of BRATS2018 dataset
        data_set: train or val
        scan_type: t1ce, flair or t2-flair
        seg_type: wt->whole tumor, et->enhancing tumor, tc->tumor core
        transform: PyTorch transformations
        """
        self.root_dir = os.path.expanduser(root_dir)
        self.data_set = data_set
        self.scan_type = scan_type
        self.seg_type = seg_type
        self.transform = transform
        
        self.base_dir = os.path.join(self.root_dir, 'SEG_{}/{}'.format(grade, self.data_set))
        dataset_txt_path = os.path.join(self.root_dir, 'SEG_{}/{}.txt'.format(grade, self.data_set))
        with open(dataset_txt_path, 'r') as f:
            self.sample_list = [x.strip() for x in f.readlines()]
        
        # assert len(self.sample_list) * 4 == len(os.listdir(self.base_dir))
        
    
    def __len__(self):
        return len(self.sample_list)

    
    def __getitem__(self, index):
        if self.scan_type == 't1ce':
            sc = np.load(os.path.join(self.base_dir, self.sample_list[index] + '_scan.npy'))[1]
            sc = np.expand_dims(sc, axis=0)
            assert sc.shape == (1, 240, 240)
        elif self.scan_type == 'flair':
            sc = np.load(os.path.join(self.base_dir, self.sample_list[index] + '_scan.npy'))[3]
            sc = np.expand_dims(sc, axis=0)
            assert sc.shape == (1, 240, 240)
        elif self.scan_type == 't2-flair':
            sc = np.load(os.path.join(self.base_dir, self.sample_list[index] + '_scan.npy'))[2:]
            assert sc.shape == (2, 240, 240)
        else:
            sc = np.load(os.path.join(self.base_dir, self.sample_list[index] + '_scan.npy'))
            assert sc.shape == (4, 240, 240)
        
        mask = np.load(os.path.join(self.base_dir, self.sample_list[index] + '_{}.npy'.format(self.seg_type)))
        
        sample = (sc, mask)
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample




# transforms
class ToTensor():
    """
    Convert ndarray samples to Tensors
    """
    def __call__(self, sample):
        sc, mask = sample
        
        sc = torch.from_numpy(sc).float()
        mask = torch.from_numpy(mask).to(torch.int64)
        
        return sc, mask
    

class NormalizeBRATS():
    """
    Subtract the mean and divide by the standard deviation
    """
    def __call__(self, sample):
        sc, mask = sample
        
        mean = np.mean(sc, axis=(1, 2), keepdims=True)
        std = np.std(sc, axis=(1, 2))
        
        no_zero_std = np.array([1 if st == 0. else st for st in std])
        no_zero_std = np.expand_dims(no_zero_std, axis=1)
        no_zero_std = np.expand_dims(no_zero_std, axis=2)
        
        sc = (sc - mean) / no_zero_std
        
        return sc, mask


class ZeroPad():
    """
    Zero-pad the scan and the mask to 256 * 256
    """
    def __call__(self, sample):
        sc, mask = sample
        
        sc = np.pad(sc, pad_width=((0, 0), (8, 8), (8, 8)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))
        mask = np.pad(mask, pad_width=((8, 8), (8, 8)), mode='constant', constant_values=((0, 0), (0, 0)))
        
        return sc, mask

class ToTensorVal():
    """
    Convert ndarray samples to Tensors
    """
    def __call__(self, sc):
        sc = torch.from_numpy(sc).float()
        
        return sc


class NormalizeBRATSVal():
    """
    Subtract the mean and divide by the standard deviation
    """
    def __call__(self, sc):
        mean = np.mean(sc, axis=(1, 2), keepdims=True)
        std = np.std(sc, axis=(1, 2))
        
        no_zero_std = np.array([1 if st == 0. else st for st in std])
        no_zero_std = np.expand_dims(no_zero_std, axis=1)
        no_zero_std = np.expand_dims(no_zero_std, axis=2)
        
        sc = (sc - mean) / no_zero_std
        
        return sc


class ZeroPadVal():
    """
    Zero-pad the scan and the mask to 256 * 256
    """
    def __call__(self, sc):
        sc = np.pad(sc, pad_width=((0, 0), (8, 8), (8, 8)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))
        
        return sc