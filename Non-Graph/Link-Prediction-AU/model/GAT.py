import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import math
from .graph import create_e_matrix
from .graph_edge_model import GEM
from .basic_block import *

# GAT Used to Learn Multi-dimensional Edge Features and Node Features
class GAT(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GAT, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_classes = num_classes
        start, end = create_e_matrix(self.num_classes)
        self.start = Variable(start, requires_grad=False).cuda()
        self.end = Variable(end, requires_grad=False).cuda()

        dim_in = self.in_channels
        dim_out = self.in_channels

        self.fc_h1 = nn.Linear(dim_in, dim_out, bias=False)
        self.fc_e1 = nn.Linear(dim_in, dim_out, bias=False)
        self.fc_proj1 = nn.Linear(3 * dim_in, dim_out, bias=False)
        self.attn_fc1 = nn.Linear(3 * dim_in, 1, bias=False)

        self.fc_h2 = nn.Linear(dim_in, dim_out, bias=False)
        self.fc_e2 = nn.Linear(dim_in, dim_out, bias=False)
        self.fc_proj2 = nn.Linear(3 * dim_in, dim_out, bias=False)
        self.attn_fc2 = nn.Linear(3 * dim_in, 1, bias=False)

        self.softmax = nn.Softmax(2)

        self.bnv1 = nn.BatchNorm1d(num_classes)
        self.bne1 = nn.BatchNorm1d(num_classes * num_classes)

        self.bnv2 = nn.BatchNorm1d(num_classes)
        self.bne2 = nn.BatchNorm1d(num_classes * num_classes)

        self.act = nn.ELU()
        self.leaky_relu = nn.LeakyReLU()

        self.init_weights_linear(dim_in, 1)

    def init_weights_linear(self, dim_in, gain):
        # conv1
        scale = gain * np.sqrt(2.0 / dim_in)
        self.fc_h1.weight.data.normal_(0, scale)
        self.fc_e1.weight.data.normal_(0, scale)
        self.fc_proj1.weight.data.normal_(0, scale)

        self.fc_h2.weight.data.normal_(0, scale)
        self.fc_e2.weight.data.normal_(0, scale)
        self.fc_proj2.weight.data.normal_(0, scale)
        #
        bn_init(self.bnv1)#, 1e-6)
        bn_init(self.bne1)#, 1e-6)
        bn_init(self.bnv2)#, 1e-6)
        bn_init(self.bne2)#, 1e-6)

    def forward(self, x, edge):
        start = self.start.cuda(x.get_device())
        end = self.end.cuda(x.get_device())
        res = x

        z_h = self.fc_h1(x)  # V x d_out
        z_e = self.fc_e1(edge)  # E x d_out

        z = torch.cat((torch.einsum('ev, bvd -> bed', (start, z_h)), torch.einsum('ev, bvd -> bed', (end, z_h)), z_e),
                      dim=-1)
        z_e = self.fc_proj1(z)
        attn = self.leaky_relu(self.attn_fc1(z))
        b, _, _ = attn.shape
        attn = attn.view(b, self.num_classes, self.num_classes, 1)
        attn = self.softmax(attn)
        attn = attn.view(b, -1, 1)

        source_z_h = torch.einsum('ev, bvd -> bed', (start, z_h))
        z_h = torch.einsum('ve, bed -> bvd', (end.t(), attn * source_z_h))
        x = self.act(res + self.bnv1(z_h))
        edge = self.act(self.bne1(z_e))

        res = x

        z_h = self.fc_h2(x)  # V x d_out
        z_e = self.fc_e2(edge)  # E x d_out

        z = torch.cat((torch.einsum('ev, bvd -> bed', (start, z_h)), torch.einsum('ev, bvd -> bed', (end, z_h)), z_e),
                      dim=-1)
        z_e = self.fc_proj2(z)
        attn = self.leaky_relu(self.attn_fc2(z))
        b, _, _ = attn.shape
        attn = attn.view(b, self.num_classes, self.num_classes, 1)
        attn = self.softmax(attn)
        attn = attn.view(b, -1, 1)

        source_z_h = torch.einsum('ev, bvd -> bed', (start, z_h))
        z_h = torch.einsum('ve, bed -> bvd', (end.t(), attn * source_z_h))
        x = self.act(res + self.bnv2(z_h))
        edge = self.act(self.bne2(z_e))

        return x, edge
