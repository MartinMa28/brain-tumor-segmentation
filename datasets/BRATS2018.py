from torchvision import datasets, transforms
from torchvision.transforms import functional as F
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from PIL import Image
import os


class BRATS2018(Dataset):
    def __init__(self, root_dir, data_set='train', seg_type='wt', scan_type='t1ce', transform=None):
        """
        root_dir: the directory of BRATS2018 dataset
        data_set: train or val
        seg_type: wt->whole tumor, et->enhancing tumor, tc->tumor core
        transform: PyTorch transformations
        """
        self.root_dir = os.path.expanduser(root_dir)
        self.data_set = data_set
        self.scan_type = scan_type
        self.seg_type = seg_type
        self.transform = transform
        
        self.base_dir = os.path.join(self.root_dir, f'seg/{self.data_set}')
        dataset_txt_path = os.path.join(self.root_dir, f'seg/{self.data_set}.txt')
        with open(dataset_txt_path, 'r') as f:
            self.sample_list = [x.strip() for x in f.readlines()]
        
        assert len(self.sample_list) * 4 == len(os.listdir(self.base_dir))
        
    
    def __len__(self):
        return len(self.sample_list)

    
    def __getitem__(self, index):
        if self.scan_type == 't1ce':
            sc = np.load(os.path.join(self.base_dir, self.sample_list[index] + '_scan.npy'))[1]
            assert sc.shape == (240, 240)
        else:
            sc = np.load(os.path.join(self.base_dir, self.sample_list[index] + '_scan.npy'))
            sc = np.array([sc[2], sc[3]])
            assert sc.shape == (2, 240, 240)
        
        mask = np.load(os.path.join(self.base_dir, self.sample_list[index] + f'_{self.seg_type}.npy'))
        
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
        mask = torch.from_numpy(mask).float()
        
        return sc, mask
    

class NormalizeBRATS():
    """
    Subtract the mean and divide by the standard deviation
    """
    def __call__(self, sample):
        sc, mask = sample
        
        mean = torch.mean(sc)
        std = torch.std(sc)
        
        sc = sc.sub(mean).div(std)
        
        return sc, mask   