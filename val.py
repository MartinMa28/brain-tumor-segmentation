import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader
import os
import nibabel as nib
import torch.nn.functional as F

from metrics.torch_seg_metrics import dice_score
from datasets.BRATS2018 import BRATS2018, NormalizeBRATS, ToTensor
from metrics.metrics import Evaluator
from models.unet import UNet


def validate(state_dict_path, use_gpu, device):
    model = UNet(n_channels=1, n_classes=2)
    model.load_state_dict(torch.load(state_dict_path, map_location='cpu' if not use_gpu else device))
    model.to(device)
    val_transforms = transforms.Compose([
        ToTensor(), 
        NormalizeBRATS()])

    BraTS_val_ds = BRATS2018('./BRATS2018',\
        data_set='val',\
        seg_type='et',\
        scan_type='t1ce',\
        transform=val_transforms)

    data_loader = DataLoader(BraTS_val_ds, batch_size=2, shuffle=False, num_workers=0)

    running_dice_score = 0.

    for batch_ind, batch in enumerate(data_loader):
        imgs, targets = batch
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        model.eval()

        with torch.no_grad():
            outputs = model(imgs)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

            running_dice_score += dice_score(preds, targets) * targets.size(0)
            print('running dice score: {:.6f}'.format(running_dice_score))
    
    dice = running_dice_score / len(BraTS_val_ds)
    print('mean dice score of the validating set: {:.6f}'.format(dice))
        



if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    validate('../scores/UNet_BraTS_Weighted/trained_model.pt', use_gpu, device)