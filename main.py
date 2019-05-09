from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from models.fcn import VGGNet, FCN8sScaledBN, FCN8sScaled
from models.unet import UNet
from models.unet_resnet_encoder import UNetWithResnet50Encoder
from models.vgg_encoder import VGGEncoder
from models.resnet3D import resnet50_3D
from datasets.BRATS2018 import BRATS2018, NormalizeBRATS, ToTensor, ZeroPad
from datasets.BRATS2018_3D import BRATS2018_3D, NormalizeBRATS3D, CenterCropBRATS3D

from torchvision import transforms
import copy
from metrics.metrics import Evaluator
from metrics.torch_seg_metrics import *
from hyper_param_config import *

import numpy as np
import time
import datetime
import argparse
import os
import math
import signal

import logging
from logging.config import fileConfig

# global variables
if not os.path.exists('logs/'):
    os.makedirs('logs/')
    os.mknod('logs/basic_logs.log')
    
fileConfig('./logging_conf.ini')
logger = logging.getLogger('main')


logger.info('Configs: ')
logger.info(configs)

score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)


parser = argparse.ArgumentParser()
parser.add_argument("task", help='Train a segmentation model or a classification model.',\
    default='seg', choices=['seg', 'cls'])
parser.add_argument('-i', '--input', help='input MRI scan modalities', default='all',\
    choices=['t1ce', 'flair', 't2-flair', 'all'])
parser.add_argument('-g', '--grade', help='the grade of training data (HGG or LGG)',\
    default='HGG', choices=['HGG', 'LGG'])
parser.add_argument('--seg_task', help='segmentaiton seg_task', default='seg',\
    choices=['seg', 'wt', 'et', 'tc'])
parser.add_argument('--pre_trained', help='whether training from a pre-trained model',\
    action='store_true')
args = parser.parse_args()

use_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if use_gpu else 'cpu')
# global variables

def get_dataset_dataloader(input_data_type, seg_type, batch_size, grade='HGG'):
    data_transforms = transforms.Compose([
            NormalizeBRATS(),
            ToTensor()
        ])
    
    if input_data_type == 't1ce':
        data_set = {
            phase: BRATS2018('./BRATS2018/',\
                            grade=grade,\
                            data_set=phase,\
                            seg_type=seg_type,\
                            transform=data_transforms)
            for phase in ['train', 'val']
        }
    elif input_data_type == 'flair':
        data_set = {
            phase: BRATS2018('./BRATS2018/',\
                            grade=grade,\
                            data_set=phase,\
                            scan_type='flair',\
                            seg_type=seg_type,\
                            transform=data_transforms)
            for phase in ['train', 'val']
        }
    elif input_data_type == 't2-flair':
        data_set = {
            phase: BRATS2018('./BRATS2018/',\
                            grade=grade,\
                            data_set=phase,\
                            scan_type='t2-flair',\
                            seg_type=seg_type,\
                            transform=data_transforms)
            for phase in ['train', 'val']
        }
    elif input_data_type == 'all':
        data_set = {
            phase: BRATS2018('./BRATS2018/',\
                            grade=grade,\
                            data_set=phase,\
                            scan_type='all',\
                            seg_type=seg_type,\
                            transform=data_transforms)
            for phase in ['train', 'val']
        }
    else:
        raise ValueError('Scan type must be t1ce, flair, t2-flair, or all!')
    

    data_loader = {
        'train': DataLoader(data_set['train'], batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(data_set['val'], batch_size=batch_size, shuffle=False, num_workers=0)
    }

    return data_set, data_loader



def time_stamp() -> str:
    ts = time.time()
    time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return time_stamp



class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
    
    def dice_coef(self, preds, targets):
        smooth = 5e-3
        num = preds.size(0)              # batch size
        preds_flat = preds.view(num, -1).float()
        targets_flat = targets.view(num, -1).float()

        intersection = (preds_flat * targets_flat).sum()
        logger.debug('intersection: {:.4f}, sum_preds: {:.4f}, sum_targets: {:.4f}'.format(intersection,\
            preds_flat.sum(),\
            targets_flat.sum()))

        return (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        score = self.dice_coef(probs, targets)
        score = 1 - score

        return score


def train(input_data_type, grade, seg_type, num_classes, batch_size, epochs, use_gpu, learning_rate, w_decay, pre_trained=False):
    logger.info('Start training using {} modal.'.format(input_data_type))
    model = UNet(4, 4, residual=True, expansion=2)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=w_decay)
    
    if pre_trained:
        checkpoint = torch.load(pre_trained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if use_gpu:
        ts = time.time()
        model.to(device)

        print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 5 epochs

    data_set, data_loader = get_dataset_dataloader(input_data_type, seg_type, batch_size, grade=grade)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_iou = 0.0

    epoch_loss = np.zeros((2, epochs))
    epoch_acc = np.zeros((2, epochs))
    epoch_class_acc = np.zeros((2, epochs))
    epoch_mean_iou = np.zeros((2, epochs))
    evaluator = Evaluator(num_classes)

    def term_int_handler(signal_num, frame):
        np.save(os.path.join(score_dir, 'epoch_accuracy'), epoch_acc)
        np.save(os.path.join(score_dir, 'epoch_mean_iou'), epoch_mean_iou)
        np.save(os.path.join(score_dir, 'epoch_loss'), epoch_loss)

        model.load_state_dict(best_model_wts)

        logger.info('Got terminated and saved model.state_dict')
        torch.save(model.state_dict(), os.path.join(score_dir, 'terminated_model.pt'))
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(score_dir, 'terminated_model.tar'))
        
        quit()

    signal.signal(signal.SIGINT, term_int_handler)
    signal.signal(signal.SIGTERM, term_int_handler)
    

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

                preds = torch.argmax(F.softmax(outputs, dim=1), dim=1, keepdim=True)
                running_loss += loss * imgs.size(0)
                logger.debug('Batch {} running loss: {:.4f}'.format(batch_ind,\
                    running_loss))

                # test the iou and pixelwise accuracy using evaluator
                preds = torch.squeeze(preds, dim=1)
                preds = preds.cpu().numpy()
                targets = targets.cpu().numpy()
                evaluator.add_batch(targets, preds)

            
            epoch_loss[phase_ind, epoch] = running_loss / len(data_set[phase])
            epoch_acc[phase_ind, epoch] = evaluator.Pixel_Accuracy()
            epoch_class_acc[phase_ind, epoch] = evaluator.Pixel_Accuracy_Class()
            epoch_mean_iou[phase_ind, epoch] = evaluator.Mean_Intersection_over_Union()
            
            logger.info('{} loss: {:.4f}, acc: {:.4f}, class acc: {:.4f}, mean iou: {:.6f}'.format(phase,\
                epoch_loss[phase_ind, epoch],\
                epoch_acc[phase_ind, epoch],\
                epoch_class_acc[phase_ind, epoch],\
                epoch_mean_iou[phase_ind, epoch]))


            if phase == 'val' and epoch_mean_iou[phase_ind, epoch] > best_iou:
                best_iou = epoch_mean_iou[phase_ind, epoch]
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'val' and (epoch + 1) % 10 == 0:
                logger.info('Saved model.state_dict in epoch {}'.format(epoch + 1))
                torch.save(model.state_dict(), os.path.join(score_dir, 'epoch{}_model.pt'.format(epoch + 1)))
        
        print()
    
    time_elapsed = time.time() - since
    logger.info('Training completed in {}m {}s'.format(int(time_elapsed / 60),\
        int(time_elapsed) % 60))
    
    # load best model weights
    model.load_state_dict(best_model_wts)

    # save numpy results
    np.save(os.path.join(score_dir, 'epoch_accuracy'), epoch_acc)
    np.save(os.path.join(score_dir, 'epoch_mean_iou'), epoch_mean_iou)
    np.save(os.path.join(score_dir, 'epoch_loss'), epoch_loss)

    return model, optimizer

def train_classification(num_classes, batch_size, epochs, use_gpu, learning_rate, w_decay, pre_trained=False):
    logger.info('Starts training a classification model.')
    model = resnet50_3D(num_classes=num_classes)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=w_decay)
    
    if pre_trained:
        checkpoint = torch.load(pre_trained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if use_gpu:
        ts = time.time()
        model.to(device)

        print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    data_transforms = transforms.Compose([
        CenterCropBRATS3D(),
        NormalizeBRATS3D(),
        ToTensor()
    ])

    data_set = {
            phase: BRATS2018_3D('BRATS2018/',\
                            data_set=phase,\
                            transform=data_transforms)
            for phase in ['train', 'val']
        }
    data_loader = {
        'train': DataLoader(data_set['train'], batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(data_set['val'], batch_size=batch_size, shuffle=False, num_workers=0)
    }

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    epoch_loss = np.zeros((2, epochs))
    epoch_acc = np.zeros((2, epochs))
    epoch_class_acc = np.zeros((2, epochs, num_classes))
    
    def term_int_handler(signal_num, frame):
        np.save(os.path.join(score_dir, 'epoch_accuracy'), epoch_acc)
        np.save(os.path.join(score_dir, 'epoch_loss'), epoch_loss)

        model.load_state_dict(best_model_wts)

        logger.info('Got terminated and saved model.state_dict')
        torch.save(model.state_dict(), os.path.join(score_dir, 'terminated_model.pt'))
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(score_dir, 'terminated_model.tar'))
        
        quit()

    signal.signal(signal.SIGINT, term_int_handler)
    signal.signal(signal.SIGTERM, term_int_handler)

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
            
            running_loss = torch.tensor(0, dtype=torch.float32)
            running_acc = torch.tensor(0, dtype=torch.float32)
            running_class_acc = torch.zeros(num_classes, dtype=torch.float32)

            for batch_ind, batch in enumerate(data_loader[phase]):
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.view(-1,).to(device)

                # zero the learnable parameters gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                preds = torch.argmax(F.softmax(outputs, dim=1), dim=1).view(-1,)
                running_loss += loss * imgs.size(0)
                running_acc += torch.sum(preds == labels)
                running_class_acc += torch.tensor([torch.sum((preds == labels) * (labels == l)).float() 
                                            / torch.sum(labels == l) for l in [0., 1]]) * imgs.size(0)
                logger.debug('Batch {} running loss: {:.4f}'.format(batch_ind,\
                    running_loss))
            
            epoch_loss[phase_ind, epoch] = running_loss.cpu().numpy() / len(data_set[phase])
            epoch_acc[phase_ind, epoch] = running_acc.cpu().numpy() / len(data_set[phase])
            epoch_class_acc[phase_ind, epoch, :] = running_class_acc.cpu().numpy() / len(data_set[phase])

            logger.info('{} loss: {:.4f}, acc: {:.4f}, class acc: {:.4f}'.format(phase,\
                epoch_loss[phase_ind, epoch],\
                epoch_acc[phase_ind, epoch],\
                epoch_class_acc[phase_ind, epoch]))

            if phase == 'val' and epoch_acc[phase_ind, epoch] > best_acc:
                best_acc = epoch_acc[phase_ind, epoch]
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'val' and (epoch + 1) % 10 == 0:
                logger.info('Saved model.state_dict in epoch {}'.format(epoch + 1))
                torch.save(model.state_dict(), os.path.join(score_dir, 'epoch{}_model.pt'.format(epoch + 1)))

    
    time_elapsed = time.time() - since
    logger.info('Training completed in {}m {}s'.format(int(time_elapsed / 60),\
        int(time_elapsed) % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # save numpy results
    np.save(os.path.join(score_dir, 'epoch_accuracy'), epoch_acc)
    np.save(os.path.join(score_dir, 'epoch_class_accuracy'), epoch_class_acc)
    np.save(os.path.join(score_dir, 'epoch_loss'), epoch_loss)

    return model, optimizer
    
            

if __name__ == "__main__":
    if args.task == 'seg':
        model, optimizer = train(args.input, args.grade, args.seg_task, n_classes, batch_size, epochs, use_gpu, lr, w_decay, args.pre_trained)
    else:
        model, optimizer = train_classification(n_classes, batch_size, epochs, use_gpu, lr, w_decay, args.pre_trained)
    
    logger.info('Saved model.state_dict')
    torch.save(model.state_dict(), os.path.join(score_dir, 'trained_model.pt'))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(score_dir, 'trained_model_checkpoint.tar'))
