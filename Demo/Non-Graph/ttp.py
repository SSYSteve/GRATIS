import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
from resnet import resnet50
from scipy.spatial import distance
import dgl
from graph import create_e_matrix
from dgl.nn import GraphConv

# The toy GCN is from https://docs.dgl.ai/tutorials/blitz/5_graph_classification.html
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata["feat"] = h
        return g, dgl.mean_nodes(g, "feat")


class LinearBlock(nn.Module):
    def __init__(self, in_features,out_features=None,drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.fc.weight.data.normal_(0, math.sqrt(2. / out_features))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x).permute(0, 2, 1)
        x = self.relu(self.bn(x)).permute(0, 2, 1)
        return x

class VFE(nn.Module):
    def __init__(self, in_channels, nodes):
        super(VFE, self).__init__()
        # Input: the feature maps x from backbone
        self.in_channels = in_channels
        self.num_nodes = nodes
        self.num_classes = nodes
        class_linear_layers = []
        for i in range(self.num_nodes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)

    def forward(self, x):
    
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1))             # Latent Feature Maps [1, Node, 49, 512] From X = [1, 49, 512]
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)                            # Mean is GAP layer   [1, Node, 512]
        
        # f_v is the basic vertex, each with a dimension #in_channels
        
        # We note that we have an intermidate supervision block, the dashed yellow block, in Figure 4.
        # The intermidate supervision block uses f_v to first train a supervision task to update the update the params in [layer in enumerate(self.class_linears)]
        # However, the intermidate supervision block can be optional, as we can train these layers in an end-to-end fashion.
        return f_u, f_v
        

class TTP(nn.Module):
    def __init__(self):
        super(TTP, self).__init__()
        self.backbone = resnet50(pretrained=False) # True if you have the pretrained models
        self.in_channels = self.backbone.fc[0].weight.shape[1]
        self.out_channels = self.in_channels // 4
        self.backbone.fc = None
        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.vfe_nodes = 4
        self.VFE = VFE(self.out_channels, self.vfe_nodes)
        number_classes = 10
        self.gcn = GCN(self.out_channels, self.out_channels, number_classes) # number_classes depends on your task
    def forward(self, x, g = None, gnn_supervise = False):
        x = self.backbone(x)                       # [1, 49, 2048]
        x = self.global_linear(x)                  # [1, 49, 512]
        node_feature_beforeGAP, basic_nodes = self.VFE(x)                  
        # Note that this VFE is different from the VFE in gd.py
        # The VFE here should have pretrained weights from gd.py with the dashed intermidate supervision block
        # However, as we stated in the VFE of the gd.py, the pretraining of VFE is optional and can be trained end-to-end here.
        if g is None:
            g = dgl.DGLGraph()
            # C nearest neighbor, we use C = vfe_nodes for simplicity
            C_near = self.vfe_nodes
            f_e = torch.zeros(basic_nodes.shape[0], self.vfe_nodes * self.vfe_nodes, 1).cpu().detach().numpy()
            for m in range(basic_nodes.shape[0]):
                index_edge = 0
                for i in range(C_near):
                    for j in range(C_near):
                        a = basic_nodes[m,i,:]
                        b = basic_nodes[m,j,:]
                        f_e[m,index_edge,:].fill(distance.euclidean(a.cpu().detach().numpy(), b.cpu().detach().numpy()))
                        g.add_edges(torch.tensor(i), torch.tensor(j))
                        index_edge = index_edge + 1
        g.ndata['feat'] = basic_nodes[0,...]
        g.edata['feat'] = torch.tensor(f_e[0,...])
        
        ttp_node_beforegcn = g.ndata['feat'][:]
        ttp_feature_beforegcn = g.edata['feat']
        # additionally, we can attached a GCN in the end to supervise the ttp learning, as indicated in the right cornor of the Figure 4
        # here we use a two layer GCN as an example
        if gnn_supervise == True:
            g, __ = self.gcn(g, ttp_node_beforegcn)
            ttp_node_aftergcn = g.ndata['feat'][:]
            ttp_feature_aftergcn = g.edata['feat']
        return node_feature_beforeGAP, x, g