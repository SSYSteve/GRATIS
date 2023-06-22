import torch
import numpy as np
import torchvision
import torchvision.models as models
import os
import pathlib

def get_need_to_train(conf):

    if conf.gnn_type == "GAT":
        
        if conf.feed_type == "vertex+edge":
            f = open(conf.root_dir + "model/layers_gat_both.txt", "r+")
        else:
            f = open(conf.root_dir + "model/layers_gat_single.txt", "r+")

    elif conf.gnn_type == "GCN":
        
        if conf.feed_type == "vertex+edge":
            f = open(conf.root_dir + "model/layers_gcn_both.txt", "r+")
        else:
            f = open(conf.root_dir + "model/layers_gcn_single.txt", "r+")
   
    params = []

    for line in f.readlines():
        line = line[:-1]
        params.append(line)

    params_final = []
    for param in params:
        param = param[1:-1]
        params_final.append(param)
    return params_final


def getFrozenMEFARG(net, conf):

    # get the pretrained weights MEFARG_swin_base_BP4D_fold1.pth
    net_dict = net.state_dict()
    
    load_file = conf.root_dir + "checkpoints/MEFARG_"
    load_file += "swin_base_" if "swin" in conf.arc else "resnet50_"
    load_file += conf.dataset + "_fold" + str(conf.fold) + ".pth"
    pretrained_dict = torch.load(load_file)['state_dict']

    net_dict.update(pretrained_dict)
    net.load_state_dict(pretrained_dict, strict=False)

    need_to_train = get_need_to_train(conf)

    # freeze the backbone and AFG
    all_param_names = list(net.state_dict().keys())

    freeze_these = []
    for name in all_param_names:
        if name not in need_to_train:
            freeze_these.append(name)

    for name, param in net.named_parameters():
      if name in freeze_these:
          param.requires_grad = False

    return net
