from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs, FCN8s_bilinear, FCN8sScaledBN, FCN8sScaledOGBN, FCN8sScaled
from unet import UNet, UNetWithBilinear, UNetWithVGGEncoder
from vgg_encoder import VGGEncoder
from Cityscapes_loader import CityScapesDataset
from CamVid_loader import CamVidDataset
from VOC_loader import VOCSeg
from VOC_Aug_loader import VOCSegAug
from VOC_loader import RandomCrop, RandomHorizontalFlip, ToTensor, CenterCrop, NormalizeVOC
from torchvision import transforms
import copy
from metrics import Evaluator

from matplotlib import pyplot as plt
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
n_classes = 20 + 1
batch_size = 2
epochs = 3
lr = 1e-4
#momentum = 0
w_decay = 1e-5
step_size = 10
gamma = 0.5
configs = "FCNs-CrossEntropyLoss_batch{}_training_epochs{}_Adam_scheduler-step{}-gamma{}_lr{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, w_decay)
print('Configs: ')
print(configs)

data_set_type = sys.argv[1]
if data_set_type not in ['VOC', 'VOCAug']:
    raise ValueError('Only supports Pascal VOC and augmented Pascal VOC!')

score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)

use_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if use_gpu else 'cpu')
IU_scores    = np.zeros((epochs, n_classes))
pixel_scores = np.zeros(epochs)
# global variables

def get_dataset_dataloader(data_set_type, batch_size):
    data_transforms = {
        'train': transforms.Compose([
            RandomCrop(512),
            RandomHorizontalFlip(),
            ToTensor(),
            NormalizeVOC([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        'val': transforms.Compose([
            CenterCrop(512),
            ToTensor(),
            NormalizeVOC([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    if data_set_type == 'VOC':
        data_set = {
            phase: VOCSeg('VOC/', '2012', image_set=phase, download=False,\
            transform=data_transforms[phase]) 
            for phase in ['train', 'val']
            }
    elif data_set_type == 'VOCAug':
        data_set = {
            phase: VOCSegAug('VOCAug/', data_set=phase, transform=data_transforms[phase])
            for phase in ['train', 'val']
        }
    else:
        raise ValueError('Dateset must be VOC or VOCAug!')
    

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
    vgg_model = VGGEncoder(pretrained=True, requires_grad=True, remove_fc=True)
    unet = UNetWithVGGEncoder(vgg_model, num_classes)
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
    correct = (pred == target).sum().to(torch.float32)
    total   = (target == target).sum().to(torch.float32)
    return correct / total


def train(data_set_type, num_classes, batch_size, epochs, use_gpu, learning_rate, w_decay):
    model = get_fcn_model(num_classes, use_gpu)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=w_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 5 epochs

    data_set, data_loader = get_dataset_dataloader(data_set_type, batch_size)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    epoch_loss = np.zeros((2, epochs))
    epoch_acc = np.zeros((2, epochs))
    epoch_iou = np.zeros((2, epochs, num_classes))
    epoch_mean_iou = np.zeros((2, epochs))
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
            running_acc = 0.0
            num_of_batches = math.ceil(len(data_set[phase]) / batch_size)
            running_iou = np.zeros((num_of_batches, num_classes))
            
            
            for batch_ind, batch in enumerate(data_loader[phase]):
                imgs, targets = batch
                imgs = Variable(imgs).float()
                imgs = imgs.to(device)
                targets = Variable(targets).type(torch.LongTensor)
                targets = targets.to(device)

                # zero the learnable parameters gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # computes loss and acc for current iteration
                preds = torch.argmax(outputs, dim=1)
                ious = iou(preds, targets, num_classes)
                
                running_loss += loss * imgs.size(0)
                running_acc += pixelwise_acc(preds, targets) * imgs.size(0)
                running_iou[batch_ind, :] = ious
                logger.debug('Batch {} running loss: {}'.format(batch_ind, running_loss))

                # test the iou and pixelwise accuracy using evaluator
                preds = preds.cpu().numpy()
                targets = targets.cpu().numpy()
                evaluator.add_batch(targets, preds)

            
            epoch_loss[phase_ind, epoch] = running_loss / len(data_set[phase])
            epoch_acc[phase_ind, epoch] = running_acc / len(data_set[phase])
            epoch_iou[phase_ind, epoch] = np.nanmean(running_iou, axis=0)
            epoch_mean_iou[phase_ind, epoch] = np.nanmean(epoch_iou[phase_ind, epoch])
            
            logger.info('{} loss: {:.4f}, acc: {:.4f}, mean iou: {:.6f}'.format(phase,\
                epoch_loss[phase_ind, epoch], epoch_acc[phase_ind, epoch],\
                epoch_mean_iou[phase_ind, epoch]))

            eva_pixel_acc = evaluator.Pixel_Accuracy()
            eva_pixel_acc_class = evaluator.Pixel_Accuracy_Class()
            eva_mIOU = evaluator.Mean_Intersection_over_Union()
            logger.info('{} - Evaluator - acc: {:.4f}, acc class: {:.4f}, mean iou: {:.6f}'.format(phase,\
                eva_pixel_acc, eva_pixel_acc_class, eva_mIOU))

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
    np.save(os.path.join(score_dir, 'epoch_iou'), epoch_iou)

    return model

if __name__ == "__main__":
    model = train(data_set_type, n_classes, batch_size, epochs, use_gpu, lr, w_decay)
    if use_gpu:
        logger.info('Saved model.module.state_dict')
        torch.save(model.module.state_dict(), os.path.join(score_dir, 'trained_model.pt'))
    else:
        logger.info('Saved model.state_dict')
        torch.save(model.state_dict(), os.path.join(score_dir, 'trained_model.pt'))




