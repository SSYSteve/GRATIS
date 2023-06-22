import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
from .swin_transformer import swin_transformer_tiny, swin_transformer_small, swin_transformer_base
from .resnet import resnet18, resnet50, resnet101
from .graph import create_e_matrix
from .graph_edge_model import GEM
from .basic_block import *
from .GAT import *
from .GCN import *

class Head(nn.Module):
    def __init__(self, in_channels, num_classes, gnn_type, feed_type):
        super(Head, self).__init__()
        # The head of network
        # Input: the feature maps x from backbone
        # Output: the AU link predictions cl_link for each pair of AUs
        # Modules: 1. AFG extracts individual Au feature maps U_1 ---- U_N
        #          2. GEM: graph edge modeling for learning multi-dimensional edge features
        #          3. GNN for graph learning with node and multi-dimensional edge features
        # sc: individually calculate cosine similarity between node features and a trainable vector.
        # TODO TODO TODO edge fc: for edge prediction
        self.feed_type = feed_type
        self.in_channels = in_channels
        self.num_classes = num_classes
        class_linear_layers = []
        for i in range(self.num_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)
        self.edge_extractor = GEM(self.in_channels, self.num_classes)
        if gnn_type == "GCN":
            self.gnn = GCN(self.in_channels, self.num_classes)
        else:
            self.gnn = GAT(self.in_channels, self.num_classes)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.edge_fc = nn.Linear(self.in_channels, 4)
        self.relu = nn.ReLU()
        
        if feed_type != "vertex+edge":
            self.fc1 = nn.Linear(self.in_channels*2, self.in_channels)
            self.fc2 = nn.Linear(self.in_channels, self.in_channels // 2)
            self.fc3 = nn.Linear(self.in_channels // 2, 4)
            
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)
        
        else:
            self.edge_fc_1 = nn.Linear(self.in_channels*2, self.in_channels)
            self.edge_fc_2 = nn.Linear(self.in_channels, self.in_channels // 2)
            self.edge_fc_3 = nn.Linear(self.in_channels // 2, 4)

            self.node_fc_1 = nn.Linear(self.in_channels*2, self.in_channels)
            self.node_fc_2 = nn.Linear(self.in_channels, self.in_channels // 2)
            self.node_fc_3 = nn.Linear(self.in_channels // 2, 4)

            self.final_fc = nn.Linear(8, 4)

            nn.init.xavier_uniform_(self.edge_fc_1.weight)
            nn.init.xavier_uniform_(self.edge_fc_2.weight)
            nn.init.xavier_uniform_(self.edge_fc_3.weight)
            nn.init.xavier_uniform_(self.node_fc_1.weight)
            nn.init.xavier_uniform_(self.node_fc_2.weight)
            nn.init.xavier_uniform_(self.node_fc_3.weight)
            nn.init.xavier_uniform_(self.final_fc.weight)

        nn.init.xavier_uniform_(self.edge_fc.weight)
        nn.init.xavier_uniform_(self.sc)

    def forward(self, x):
        # AFG
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)

        # MEFL
        f_e = self.edge_extractor(f_u, x)
        f_e = f_e.mean(dim=-2)
        f_v, f_e = self.gnn(f_v, f_e)
        
        cl_link = torch.cuda.FloatTensor([])
        for i in range(f_v.size(1)):
            for j in range(i+1, f_v.size(1)):
                in_fir = self.num_classes*i + j
                in_sec = self.num_classes*j + i
                edge_1 = f_e[:, in_fir, :].reshape(f_e.size(0), 1, f_e.size(2))
                edge_2 = f_e[:, in_sec, :].reshape(f_e.size(0), 1, f_e.size(2))
                edges = torch.cat((edge_1, edge_2), 2)
                
                node_1 = f_v[:, i, :].reshape(f_v.size(0), 1, f_v.size(2))
                node_2 = f_v[:, j, :].reshape(f_v.size(0), 1, f_v.size(2))
                nodes = torch.cat((node_1, node_2), 2)

                if self.feed_type == "edge":
                    predicted = self.relu(self.fc1(edges))
                    predicted = self.relu(self.fc2(predicted))
                    predicted = self.fc3(predicted)

                    cl_link = torch.cat((cl_link, predicted), 1)
                    
                elif self.feed_type == "vertex":
                    predicted = self.relu(self.fc1(nodes))
                    predicted = self.relu(self.fc2(predicted))
                    predicted = self.fc3(predicted)

                    cl_link = torch.cat((cl_link, predicted), 1)                   
                
                elif self.feed_type == "vertex+edge":
                    edges = self.relu(self.edge_fc_1(edges))
                    edges = self.relu(self.edge_fc_2(edges))
                    edges = self.edge_fc_3(edges)

                    nodes = self.relu(self.node_fc_1(nodes))
                    nodes = self.relu(self.node_fc_2(nodes))
                    nodes = self.node_fc_3(nodes)

                    all_feats = torch.cat((edges, nodes), 2)
                    all_feats = self.final_fc(all_feats)

                    cl_link = torch.cat((cl_link, all_feats), 1)

        return cl_link


class MEFARG(nn.Module):
    def __init__(self, num_classes=12, backbone='swin_transformer_base', gnn_type='GCN', feed_type='vertex'):
        super(MEFARG, self).__init__()
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            self.out_channels = self.in_channels // 2
            self.backbone.head = None

        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18()
            elif backbone == 'resnet101':
                self.backbone = resnet101()
            else:
                self.backbone = resnet50()
            self.in_channels = self.backbone.fc.weight.shape[1]
            self.out_channels = self.in_channels // 4
            self.backbone.fc = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = Head(self.out_channels, num_classes, gnn_type, feed_type)

    def forward(self, x):
        # x: b d c
        x = self.backbone(x)
        x = self.global_linear(x)
        cl_link = self.head(x)
        return cl_link
