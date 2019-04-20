from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs, FCN8s_bilinear, FCN8sScaledBN, FCN8sScaledOGBN, FCN8sScaled
from models.unet import UNet, UNetWithBilinear, UNetWithVGGEncoder
from models.vgg_encoder import VGGEncoder
from datasets.BRATS2018 import BRATS2018, NormalizeBRATS, ToTensor

from torchvision import transforms
import copy
from metrics.metrics import Evaluator

import numpy as np
import time
import datetime
import sys
import os
import math

import logging
from logging.config import fileConfig

# global variables
if not os.path.exists('logs/'):
    os.makedirs('logs/')
    os.mknod('logs/basic_logs.log')
    
fileConfig('./logging_conf.ini')
logger = logging.getLogger('main')


# 20 classes and background for VOC segmentation
n_classes = 2
batch_size = 4
epochs = 50
lr = 1e-2
#momentum = 0
w_decay = 1e-5
step_size = 5
gamma = 0.5
configs = "UNets-BRATS2018_batch{}_training_epochs{}_Adam_scheduler-step{}-gamma{}_lr{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, w_decay)
print('Configs: ')
print(configs)

input_data_type = sys.argv[1]
if input_data_type not in ['t1ce', 't2-flair']:
    raise ValueError('Only supports scan types of t1ce or t2-flair')

score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)

use_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if use_gpu else 'cpu')
IU_scores    = np.zeros((epochs, n_classes))
pixel_scores = np.zeros(epochs)
# global variables

def get_dataset_dataloader(input_data_type, batch_size):
    data_transforms = transforms.Compose([
            ToTensor(),
            NormalizeBRATS()
        ])
    
    if input_data_type == 't1ce':
        data_set = {
            phase: BRATS2018('./BRATS2018/',\
                            data_set=phase,\
                            seg_type='et',\
                            transform=data_transforms)
            for phase in ['train', 'val']
        }
    elif input_data_type == 't2-flair':
        data_set = {
            phase: BRATS2018('./BRATS2018/',\
                            data_set=phase,\
                            scan_type='t2-flair',\
                            seg_type='wt',\
                            transform=data_transforms)
            for phase in ['train', 'val']
        }
    else:
        raise ValueError('Scan type must be t1ce or t2-flair!')
    

    data_loader = {
        'train': DataLoader(data_set['train'], batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(data_set['val'], batch_size=batch_size, shuffle=False, num_workers=0)
    }

    return data_set, data_loader

def get_fcn_model(num_classes, use_gpu):
    vgg_model = VGGNet(requires_grad=True, remove_fc=True, batch_norm=True)
    fcn_model = FCN8sScaledBN(pretrained_net=vgg_model, n_class=num_classes)

    if use_gpu:
        ts = time.time()
        vgg_model = vgg_model.cuda()
        fcn_model = fcn_model.cuda()
        num_gpu = list(range(torch.cuda.device_count()))
        fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
        
        print("Finish cuda loading, time elapsed {}".format(time.time() - ts))
    
    return fcn_model

def get_unet_model(num_classes, use_gpu):
    # vgg_model = VGGEncoder(pretrained=True, requires_grad=True, remove_fc=True)
    # unet = UNetWithVGGEncoder(vgg_model, num_classes)
    unet = UNet(1, num_classes)
    if use_gpu:
        ts = time.time()
        unet = unet.cuda()
        num_gpu = list(range(torch.cuda.device_count()))
        unet = nn.DataParallel(unet, device_ids=num_gpu)

        print("Finish cuda loading, time elapsed {}".format(time.time() - ts))
    
    return unet

def time_stamp() -> str:
    ts = time.time()
    time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return time_stamp

# Borrows and modifies iou() from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions    
def iou(pred, target, num_classes):
    ious = np.zeros(num_classes)
    for cl in range(num_classes):
        pred_inds = (pred == cl)
        target_inds = (target == cl)
        intersection = pred_inds[target_inds].sum().to(torch.float32)
        union = pred_inds.sum() + target_inds.sum() - intersection
        union = union.to(torch.float32)
        if union == 0:
            # if there is no ground truth, do not include in evaluation
            ious[cl] = float('nan')  
        else:
            ious[cl] = float(intersection) / max(union, 1)

    return ious.reshape((1, num_classes))

def pixelwise_acc(pred, target):
    pred = pred.float()
    target = target.float()
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total

def dice_score(preds, targets):
    num = preds.size(0)              # batch size
    preds_flat = preds.view(num, -1).float()
    targets_flat = targets.view(num, -1).float()
    
    intersection = (preds_flat * targets_flat).sum()
    
    return (2. * intersection)/(preds_flat.sum() + targets_flat.sum())


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
    
    def dice_coef(self, preds, targets):
        smooth = 0.01
        num = preds.size(0)              # batch size
        preds_flat = preds.view(num, -1).float()
        targets_flat = targets.view(num, -1).float()

        intersection = (preds_flat * targets_flat).sum()
        logger.debug(f'intersection: {intersection}, sum_preds: {preds_flat.sum()}, sum_targets: {targets_flat.sum()}')

        return (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)

        score = self.dice_coef(probs[:, 1, :, :], targets)
        score = 1 - score

        return score


def train(input_data_type, num_classes, batch_size, epochs, use_gpu, learning_rate, w_decay):
    model = get_unet_model(num_classes, use_gpu)
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]).to(device))
    criterion = SoftDiceLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=w_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 5 epochs

    data_set, data_loader = get_dataset_dataloader(input_data_type, batch_size)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    epoch_loss = np.zeros((2, epochs))
    epoch_acc = np.zeros((2, epochs))
    epoch_class_acc = np.zeros((2, epochs))
    epoch_mean_iou = np.zeros((2, epochs))
    epoch_mean_dice = np.zeros((2, epochs))
    evaluator = Evaluator(num_classes)

    for epoch in range(epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, epochs))
        logger.info('-' * 28)
        
        
        for phase_ind, phase in enumerate(['train', 'val']):
            if phase == 'train':
                model.train()
                logger.info(phase)
            else:
                model.eval()
                logger.info(phase)
            
            evaluator.reset()
            running_loss = 0.0
            running_dice = 0.0
            num_of_batches = math.ceil(len(data_set[phase]) / batch_size)
            
            
            for batch_ind, batch in enumerate(data_loader[phase]):
                imgs, targets = batch
                imgs = imgs.to(device)
                targets = targets.to(device)

                # zero the learnable parameters gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                running_loss += loss * imgs.size(0)
                dice = (dice_score(preds, targets) * imgs.size(0))
                running_dice = np.nansum([dice, running_dice], axis=0)
                logger.debug('Batch {} running loss: {:.4f}, dice score: {:.4f}'.format(batch_ind,\
                    running_loss,\
                    running_dice))

                # test the iou and pixelwise accuracy using evaluator
                preds = preds.cpu().numpy()
                targets = targets.cpu().numpy()
                evaluator.add_batch(targets, preds)

            
            epoch_loss[phase_ind, epoch] = running_loss / len(data_set[phase])
            epoch_mean_dice[phase_ind, epoch] = running_dice / len(data_set[phase])
            epoch_acc[phase_ind, epoch] = evaluator.Pixel_Accuracy()
            epoch_class_acc[phase_ind, epoch] = evaluator.Pixel_Accuracy_Class()
            epoch_mean_iou[phase_ind, epoch] = evaluator.Mean_Intersection_over_Union()
            
            logger.info('{} loss: {:.4f}, acc: {:.4f}, class acc: {:.4f}, mean iou: {:.6f}, mean dice score: {:.6f}'.format(phase,\
                epoch_loss[phase_ind, epoch],\
                epoch_acc[phase_ind, epoch],\
                epoch_class_acc[phase_ind, epoch],\
                epoch_mean_iou[phase_ind, epoch],\
                epoch_mean_dice[phase_ind, epoch]))


            if phase == 'val' and epoch_acc[phase_ind, epoch] > best_acc:
                best_acc = epoch_acc[phase_ind, epoch]
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    logger.info('Training completed in {}m {}s'.format(int(time_elapsed / 60),\
        int(time_elapsed) % 60))
    
    # load best model weights
    model.load_state_dict(best_model_wts)

    # save numpy results
    np.save(os.path.join(score_dir, 'epoch_accuracy'), epoch_acc)
    np.save(os.path.join(score_dir, 'epoch_mean_iou'), epoch_mean_iou)
    np.save(os.path.join(score_dir, 'epoch_mean_dice'), epoch_mean_dice)

    return model

if __name__ == "__main__":
    model = train(input_data_type, n_classes, batch_size, epochs, use_gpu, lr, w_decay)
    if use_gpu:
        logger.info('Saved model.module.state_dict')
        torch.save(model.module.state_dict(), os.path.join(score_dir, 'trained_model.pt'))
    else:
        logger.info('Saved model.state_dict')
        torch.save(model.state_dict(), os.path.join(score_dir, 'trained_model.pt'))




