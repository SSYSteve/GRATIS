import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import dgl

class GD(nn.Module):
    
    def forward(self, g):
        initial_node = g.ndata['feat'][:]
        initial_feature = g.edata['feat']
        return g, initial_node, initial_feature