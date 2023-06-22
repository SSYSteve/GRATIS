import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging

from model.MEFL import MEFARG
from dataset import *
from utils import *
from conf import get_config,set_logger,set_outdir,set_env
from model.frozen_layers import *


def comp_confmat(actual, predicted):
    classes = np.unique(actual)
    confmat = np.zeros((len(classes), len(classes)))

    for i in range(len(classes)):
        for j in range(len(classes)):
           confmat[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))

    return confmat.astype(int)

def disfa_get_pairwise_scores(pair_scores, extension):
    av = 0.0
    vals = []
    for k,v in pair_scores.items():
        av += v
        vals.append(v*100)
    pairwise_dict = dict()
    aus = [1,2,4,6,9,12,25,26]

    pair_ind = 0
    keys = []
    top_acc = max(vals)
    bot_acc = min(vals)
    top_pair = ""
    bot_pair = ""

    for i in range(8):
        for j in range(i+1, 8):
            key = "AU" + str(aus[i]) + " & AU" +str(aus[j])
            keys.append(key)
            pairwise_dict[key] = vals[pair_ind]
            top_pair = top_pair if vals[pair_ind] is not max(vals) else key
            bot_pair = bot_pair if vals[pair_ind] is not min(vals) else key
            pair_ind += 1

    infostr = {'Top Accuracy {:.2f}' .format(top_acc)}
    logging.info(infostr)
    infostr = {'Top Pair {:.2f}' .format(top_pair)}
    logging.info(infostr)
    infostr = {'Bottom Accuracy {:.2f}' .format(bot_ac)}
    logging.info(infostr)
    infostr = {'Bottom Pair {:.2f}' .format(bot_pair)}
    logging.info(infostr)
    infostr = {'Std Deviation {:.2f}' .format(np.std(vals))}
    logging.info(infostr)
    
    pairs = keys
    accuracies = vals
    df = pd.DataFrame({"Pairs": pairs, "Accuracies": accuracies})
    df.to_csv(filename+'.csv')  
    print(df)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    return df.to_dict()


def bp4d_get_pairwise_scores(pair_scores, extension):
    av = 0.0
    vals = []
    for k,v in pair_scores.items():
        av += v
        vals.append(v*100)
    pairwise_dict = dict()
    aus = [1,2,4,6,7,10,12,14,15,17,23,24]

    
    pair_ind = 0
    keys = []
    top_acc = max(vals)
    bot_acc = min(vals)
    top_pair = ""
    bot_pair = ""

    for i in range(8):
        for j in range(i+1, 8):
            key = "AU" + str(aus[i]) + " & AU" +str(aus[j])
            keys.append(key)
            pairwise_dict[key] = vals[pair_ind]
            top_pair = top_pair if vals[pair_ind] is not max(vals) else key
            bot_pair = bot_pair if vals[pair_ind] is not min(vals) else key
            pair_ind += 1

    infostr = {'Top Accuracy {:.2f}' .format(top_acc)}
    logging.info(infostr)
    infostr = {'Top Pair {:.2f}' .format(top_pair)}
    logging.info(infostr)
    infostr = {'Bottom Accuracy {:.2f}' .format(bot_ac)}
    logging.info(infostr)
    infostr = {'Bottom Pair {:.2f}' .format(bot_pair)}
    logging.info(infostr)
    infostr = {'Std Deviation {:.2f}' .format(np.std(vals))}
    logging.info(infostr)
    
    pairs = keys
    accuracies = vals
    df = pd.DataFrame({"Pairs": pairs, "Accuracies": accuracies})
    df.to_csv(filename+'.csv')  
    print(df)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    return df.to_dict()


def confusion_uar(targets, predictions):
    confmat = comp_confmat(targets, predictions)
    total = np.sum(confmat)
    diags = np.sum(np.diag(confmat))
    check = diags / total
    infostr = {'Diagonal accuracy from Conf. Matrix: {:.2f}' .format(100*check)}
    logging.info(infostr)
    recall_vals = []
    for i in range(len(confmat)):
        tp = confmat[i][i]
        fn = sum(confmat[i]) - confmat[i][i]
        recall_vals.append(tp / (tp+fn))
    
    uar = sum(recall_vals) / len(recall_vals)
    infostr = {'UAR Score: {:.2f}' .format(uar)}
    logging.info(infostr)

    return uar
