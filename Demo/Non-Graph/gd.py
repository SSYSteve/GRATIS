import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
from resnet import resnet50
from scipy.spatial import distance
import dgl

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
        return f_v
        

class GD(nn.Module):
    def __init__(self):
        super(GD, self).__init__()
        self.backbone = resnet50(pretrained=False) # True if you have the pretrained models
        self.in_channels = self.backbone.fc[0].weight.shape[1]
        self.out_channels = self.in_channels // 4
        self.backbone.fc = None
        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.vfe_nodes = 4
        self.VFE = VFE(self.out_channels, self.vfe_nodes)
    def forward(self, x):
        x = self.backbone(x)                       # [1, 49, 2048]
        x = self.global_linear(x)                  # [1, 49, 512]
        basic_nodes = self.VFE(x)
        g = dgl.DGLGraph()
        # Predefined Rules
        f_e = torch.zeros(basic_nodes.shape[0], self.vfe_nodes * self.vfe_nodes, 1).cpu().detach().numpy()
        for m in range(basic_nodes.shape[0]):
            for i in range(self.vfe_nodes):
                for j in range(self.vfe_nodes):
                    a = basic_nodes[m,i,:]
                    b = basic_nodes[m,j,:]
                    f_e[m,i*j,:].fill(distance.euclidean(a.cpu().detach().numpy(), b.cpu().detach().numpy()))
                    g = dgl.add_edges(g, torch.tensor(i), torch.tensor(j))
        g.ndata['feat'] = basic_nodes[0,...]
        g.edata['feat'] = torch.tensor(f_e[0,...])
        initial_node = g.ndata['feat'][:]
        initial_feature = g.edata['feat']
        return g, initial_node, initial_feature