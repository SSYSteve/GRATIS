import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging

from model.MEFL import GRATIS
from dataset import *
from utils import *
from conf import get_config,set_logger,set_outdir,set_env
from model.frozen_layers import *
from link_pair_metrics import *

def get_dataloader(conf):
    print('==> Preparing data...')
    if conf.dataset == 'BP4D':
        valset = BP4D(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 2)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    elif conf.dataset == 'DISFA':
        valset = DISFA(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 2)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    return val_loader, len(valset)

# Val
def val(net, val_loader, conf):
    predicted_all = np.array([])
    targets_all = np.array([])
    pairwise = dict()
    
    pair_num = 66 if conf.dataset == 'BP4D' else 28
    for i in range(pair_num):
        pairwise[str(i)] = 0

    correct = 0
    total = 0
    net.eval()
    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
        targets = targets.long()
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            predicted = torch.argmax(outputs, 2)
            total += targets.size(0)*targets.size(1)
            correct += (predicted == targets).sum().item()
            
            predicted_all = np.append(predicted_all, predicted.cpu().detach().numpy())
            targets_all = np.append(targets_all, targets.cpu().detach().numpy())
            
            for x in range(predicted.size(0)):
                for y in range(predicted.size(1)):
                    if predicted[x][y] == targets[x][y]:
                        pairwise[str(y)] += 1
    
    acc = correct / total
    
    for k,v in pairwise.items():
        pairwise[k] = pair_num * v / total

    return acc, pairwise, targets_all, predicted_all

def main(conf):
    if conf.dataset == 'BP4D':
        dataset_info = BP4D_infolist
    elif conf.dataset == 'DISFA':
        dataset_info = DISFA_infolist

    # data
    val_loader, val_data_num = get_dataloader(conf)
    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold, conf.N_fold, val_data_num))

    net = GRATIS(num_classes=conf.num_classes, backbone=conf.arc, gnn_type=conf.gnn_type, feed_type=conf.feed_type)
    net = getFrozenMEFARG(net=net, conf=conf)
    
    # resume
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()

    #test
    val_acc, pairwise, targets, predictions = val(net, val_loader, conf)
    
    confusion_uar(targets, predictions)
    # log
    
    infostr = {'val_acc {:.2f}' .format(100.* val_acc)}
    logging.info(infostr)
    infostr = {'Acc-list:'}
    logging.info(infostr)



# ---------------------------------------------------------------------------------


if __name__=="__main__":
    conf = get_config()
    conf.evaluate = True
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)

