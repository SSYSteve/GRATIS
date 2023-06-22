# import torch
# import torch.nn as nn
# import numpy as np
# import torch.nn.functional as F
# from torch.autograd import Variable
# import math
# from .swin_transformer import swin_transformer_tiny, swin_transformer_small, swin_transformer_base
# from .resnet import resnet18, resnet50, resnet101
# from .graph import create_e_matrix
# from .graph_edge_model import GEM
# from .basic_block import *
#
#
# # Gated GCN Used to Learn Multi-dimensional Edge Features and Node Features
# class GNN(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(GNN, self).__init__()
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         # GNN Matrix: E x N
#         # Start Matrix Item:  define the source node of one edge
#         # End Matrix Item:  define the target node of one edge
#         # Algorithm details in Residual Gated Graph Convnets: arXiv preprint arXiv:1711.07553
#         # or Benchmarking Graph Neural Networks: arXiv preprint arXiv:2003.00982v3
#
#         start, end = create_e_matrix(self.num_classes)
#         self.start = Variable(start, requires_grad=False)
#         self.end = Variable(end, requires_grad=False)
#
#         dim_in = self.in_channels
#         dim_out = self.in_channels
#
#         self.U1 = nn.Linear(dim_in, dim_out, bias=False)
#         self.V1 = nn.Linear(dim_in, dim_out, bias=False)
#         self.A1 = nn.Linear(dim_in, dim_out, bias=False)
#         self.B1 = nn.Linear(dim_in, dim_out, bias=False)
#         self.E1 = nn.Linear(dim_in, dim_out, bias=False)
#
#         self.U2 = nn.Linear(dim_in, dim_out, bias=False)
#         self.V2 = nn.Linear(dim_in, dim_out, bias=False)
#         self.A2 = nn.Linear(dim_in, dim_out, bias=False)
#         self.B2 = nn.Linear(dim_in, dim_out, bias=False)
#         self.E2 = nn.Linear(dim_in, dim_out, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax(2)
#         self.bnv1 = nn.BatchNorm1d(num_classes)
#         self.bne1 = nn.BatchNorm1d(num_classes*num_classes)
#
#         self.bnv2 = nn.BatchNorm1d(num_classes)
#         self.bne2 = nn.BatchNorm1d(num_classes * num_classes)
#
#         self.act = nn.ReLU()
#
#         self.init_weights_linear(dim_in, 1)
#
#     def init_weights_linear(self, dim_in, gain):
#         # conv1
#         scale = gain * np.sqrt(2.0 / dim_in)
#         self.U1.weight.data.normal_(0, scale)
#         self.V1.weight.data.normal_(0, scale)
#         self.A1.weight.data.normal_(0, scale)
#         self.B1.weight.data.normal_(0, scale)
#         self.E1.weight.data.normal_(0, scale)
#
#         self.U2.weight.data.normal_(0, scale)
#         self.V2.weight.data.normal_(0, scale)
#         self.A2.weight.data.normal_(0, scale)
#         self.B2.weight.data.normal_(0, scale)
#         self.E2.weight.data.normal_(0, scale)
#
#         bn_init(self.bnv1)
#         bn_init(self.bne1)
#         bn_init(self.bnv2)
#         bn_init(self.bne2)
#
#     def forward(self, x, edge):
#         # device
#         dev = x.get_device()
#         if dev >= 0:
#             start = self.start.to(dev)
#             end = self.end.to(dev)
#
#         # GNN Layer 1:
#         res = x
#         Vix = self.A1(x)  # V x d_out
#         Vjx = self.B1(x)  # V x d_out
#         e = self.E1(edge)  # E x d_out
#         edge = edge + self.act(self.bne1(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec',(start, Vjx)) + e))  # E x d_out
#
#         e = self.sigmoid(edge)
#         b, _, c = e.shape
#         e = e.view(b,self.num_classes, self.num_classes, c)
#         e = self.softmax(e)
#         e = e.view(b, -1, c)
#
#         Ujx = self.V1(x)  # V x H_out
#         Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
#         Uix = self.U1(x)  # V x H_out
#         x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
#         x = self.act(res + self.bnv1(x))
#         res = x
#
#         # GNN Layer 2:
#         Vix = self.A2(x)  # V x d_out
#         Vjx = self.B2(x)  # V x d_out
#         e = self.E2(edge)  # E x d_out
#         edge = edge + self.act(self.bne2(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec', (start, Vjx)) + e))  # E x d_out
#
#         e = self.sigmoid(edge)
#         b, _, c = e.shape
#         e = e.view(b, self.num_classes, self.num_classes, c)
#         e = self.softmax(e)
#         e = e.view(b, -1, c)
#
#         Ujx = self.V2(x)  # V x H_out
#         Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
#         Uix = self.U2(x)  # V x H_out
#         x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
#         x = self.act(res + self.bnv2(x))
#         return x, edge
# #
# #
# # class GNN(nn.Module):
# #     def __init__(self, in_channels, num_classes):
# #         super(GNN, self).__init__()
# #         self.in_channels = in_channels
# #         self.num_classes = num_classes
# #         start, end = create_e_matrix(self.num_classes)
# #         self.start = Variable(start, requires_grad=False).cuda()
# #         self.end = Variable(end, requires_grad=False).cuda()
# #
# #         dim_in = self.in_channels
# #         dim_out = self.in_channels
# #
# #         self.fc_h1 = nn.Linear(dim_in, dim_out, bias=False)
# #         self.fc_e1 = nn.Linear(dim_in, dim_out, bias=False)
# #         self.fc_proj1 = nn.Linear(3 * dim_in, dim_out, bias=False)
# #         self.attn_fc1 = nn.Linear(3 * dim_in, 1, bias=False)
# #
# #         self.fc_h2 = nn.Linear(dim_in, dim_out, bias=False)
# #         self.fc_e2 = nn.Linear(dim_in, dim_out, bias=False)
# #         self.fc_proj2 = nn.Linear(3 * dim_in, dim_out, bias=False)
# #         self.attn_fc2 = nn.Linear(3 * dim_in, 1, bias=False)
# #
# #         self.softmax = nn.Softmax(2)
# #
# #         self.bnv1 = nn.BatchNorm1d(num_classes)
# #         self.bne1 = nn.BatchNorm1d(num_classes*num_classes)
# #
# #         self.bnv2 = nn.BatchNorm1d(num_classes)
# #         self.bne2 = nn.BatchNorm1d(num_classes * num_classes)
# #
# #         self.act = nn.ELU()
# #         self.leaky_relu = nn.LeakyReLU()
# #
# #         self.init_weights_linear(dim_in, 1)
# #
# #     def bn_init(self, bn, scale):
# #         nn.init.constant_(bn.weight, scale)
# #         nn.init.constant_(bn.bias, 0)
# #
# #     def init_weights_linear(self, dim_in, gain):
# #         # conv1
# #         scale = gain * np.sqrt(2.0 / dim_in)
# #         self.fc_h1.weight.data.normal_(0, scale)
# #         self.fc_e1.weight.data.normal_(0, scale)
# #         self.fc_proj1.weight.data.normal_(0, scale)
# #
# #         self.fc_h2.weight.data.normal_(0, scale)
# #         self.fc_e2.weight.data.normal_(0, scale)
# #         self.fc_proj2.weight.data.normal_(0, scale)
# #         #
# #         self.bn_init(self.bnv1, 1e-6)
# #         self.bn_init(self.bne1, 1e-6)
# #         self.bn_init(self.bnv2, 1e-6)
# #         self.bn_init(self.bne2, 1e-6)
# #
# #     def forward(self, x, edge):
# #         start = self.start.cuda(x.get_device())
# #         end = self.end.cuda(x.get_device())
# #         res = x
# #
# #         z_h = self.fc_h1(x)  # V x d_out
# #         z_e = self.fc_e1(edge)  # E x d_out
# #
# #         z = torch.cat((torch.einsum('ev, bvd -> bed',(start, z_h)),torch.einsum('ev, bvd -> bed', (end, z_h)),z_e),dim=-1)
# #         z_e = self.fc_proj1(z)
# #         attn = self.leaky_relu(self.attn_fc1(z))
# #         b, _, _ = attn.shape
# #         attn = attn.view(b,self.num_classes,self.num_classes,1)
# #         attn = self.softmax(attn)
# #         attn = attn.view(b,-1,1)
# #
# #         source_z_h = torch.einsum('ev, bvd -> bed', (start, z_h))
# #         z_h = torch.einsum('ve, bed -> bvd', (end.t(), attn * source_z_h))
# #         x = self.act(res + self.bnv1(z_h))
# #         edge = self.act(self.bne1(z_e))
# #         # x = self.act(z_h)
# #         # edge = self.act(z_e)
# #
# #         res = x
# #
# #         z_h = self.fc_h2(x)  # V x d_out
# #         z_e = self.fc_e2(edge)  # E x d_out
# #
# #         z = torch.cat((torch.einsum('ev, bvd -> bed', (start, z_h)), torch.einsum('ev, bvd -> bed', (end, z_h)), z_e),
# #                       dim=-1)
# #         z_e = self.fc_proj2(z)
# #         attn = self.leaky_relu(self.attn_fc2(z))
# #         b, _, _ = attn.shape
# #         attn = attn.view(b,self.num_classes,self.num_classes,1)
# #         attn = self.softmax(attn)
# #         attn = attn.view(b,-1,1)
# #
# #         source_z_h = torch.einsum('ev, bvd -> bed', (start, z_h))
# #         z_h = torch.einsum('ve, bed -> bvd', (end.t(), attn * source_z_h))
# #         x = self.act(res + self.bnv2(z_h))
# #         edge = self.act(self.bne2(z_e))
# #         # x = self.act(z_h)
# #         # edge = self.act(z_e)
# #
# #         return x,edge
#
# class Head(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(Head, self).__init__()
#         # The head of network
#         # Input: the feature maps x from backbone
#         # Output: the AU recognition probabilities cl And the logits cl_edge of edge features for classification
#         # Modules: 1. AFG extracts individual Au feature maps U_1 ---- U_N
#         #          2. GEM: graph edge modeling for learning multi-dimensional edge features
#         #          3. Gated-GCN for graph learning with node and multi-dimensional edge features
#         # sc: individually calculate cosine similarity between node features and a trainable vector.
#         # edge fc: for edge prediction
#
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         class_linear_layers = []
#         for i in range(self.num_classes):
#             layer = LinearBlock(self.in_channels, self.in_channels)
#             class_linear_layers += [layer]
#         self.class_linears = nn.ModuleList(class_linear_layers)
#         self.edge_extractor = GEM(self.in_channels, self.num_classes)
#         self.gnn = GNN(self.in_channels, self.num_classes)
#         self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
#         self.edge_fc = nn.Linear(self.in_channels, 4)
#         self.relu = nn.ReLU()
#
#         nn.init.xavier_uniform_(self.edge_fc.weight)
#         nn.init.xavier_uniform_(self.sc)
#
#     def forward(self, x):
#         # AFG
#         f_u = []
#         for i, layer in enumerate(self.class_linears):
#             f_u.append(layer(x).unsqueeze(1))
#         f_u = torch.cat(f_u, dim=1)
#         f_v = f_u.mean(dim=-2)
#
#         # MEFL
#         f_e = self.edge_extractor(f_u, x)
#         f_e = f_e.mean(dim=-2)
#         f_v, f_e = self.gnn(f_v, f_e)
#
#         b, n, c = f_v.shape
#         sc = self.sc
#         sc = self.relu(sc)
#         sc = F.normalize(sc, p=2, dim=-1)
#         cl = F.normalize(f_v, p=2, dim=-1)
#         cl = (cl * sc.view(1, n, c)).sum(dim=-1, keepdim=False)
#         cl_edge = self.edge_fc(f_e)
#         return cl, cl_edge
#
#
# class MEFARG(nn.Module):
#     def __init__(self, num_classes=12, backbone='swin_transformer_base'):
#         super(MEFARG, self).__init__()
#         if 'transformer' in backbone:
#             if backbone == 'swin_transformer_tiny':
#                 self.backbone = swin_transformer_tiny()
#             elif backbone == 'swin_transformer_small':
#                 self.backbone = swin_transformer_small()
#             else:
#                 self.backbone = swin_transformer_base()
#             self.in_channels = self.backbone.num_features
#             self.out_channels = self.in_channels // 2
#             self.backbone.head = None
#
#         elif 'resnet' in backbone:
#             if backbone == 'resnet18':
#                 self.backbone = resnet18()
#             elif backbone == 'resnet101':
#                 self.backbone = resnet101()
#             else:
#                 self.backbone = resnet50()
#             self.in_channels = self.backbone.fc.weight.shape[1]
#             self.out_channels = self.in_channels // 4
#             self.backbone.fc = None
#         else:
#             raise Exception("Error: wrong backbone name: ", backbone)
#
#         self.global_linear = LinearBlock(self.in_channels, self.out_channels)
#         self.head = Head(self.out_channels, num_classes)
#
#     def forward(self, x):
#         # x: b d c
#         x = self.backbone(x)
#         x = self.global_linear(x)
#         cl, cl_edge = self.head(x)
#         return cl, cl_edge


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
from .graph import normalize_digraph


# Gated GCN Used to Learn Multi-dimensional Edge Features and Node Features
class GNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # GNN Matrix: E x N
        # Start Matrix Item:  define the source node of one edge
        # End Matrix Item:  define the target node of one edge
        # Algorithm details in Residual Gated Graph Convnets: arXiv preprint arXiv:1711.07553
        # or Benchmarking Graph Neural Networks: arXiv preprint arXiv:2003.00982v3

        start, end = create_e_matrix(self.num_classes)
        self.start = Variable(start, requires_grad=False)
        self.end = Variable(end, requires_grad=False)

        dim_in = self.in_channels
        dim_out = self.in_channels

        self.U1 = nn.Linear(dim_in, dim_out, bias=False)
        self.V1 = nn.Linear(dim_in, dim_out, bias=False)
        self.A1 = nn.Linear(dim_in, dim_out, bias=False)
        self.B1 = nn.Linear(dim_in, dim_out, bias=False)
        self.E1 = nn.Linear(dim_in, dim_out, bias=False)

        self.U2 = nn.Linear(dim_in, dim_out, bias=False)
        self.V2 = nn.Linear(dim_in, dim_out, bias=False)
        self.A2 = nn.Linear(dim_in, dim_out, bias=False)
        self.B2 = nn.Linear(dim_in, dim_out, bias=False)
        self.E2 = nn.Linear(dim_in, dim_out, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)
        self.bnv1 = nn.BatchNorm1d(num_classes)
        self.bne1 = nn.BatchNorm1d(num_classes*num_classes)

        self.bnv2 = nn.BatchNorm1d(num_classes)
        self.bne2 = nn.BatchNorm1d(num_classes * num_classes)

        self.act = nn.ReLU()

        self.init_weights_linear(dim_in, 1)

    def init_weights_linear(self, dim_in, gain):
        # conv1
        scale = gain * np.sqrt(2.0 / dim_in)
        self.U1.weight.data.normal_(0, scale)
        self.V1.weight.data.normal_(0, scale)
        self.A1.weight.data.normal_(0, scale)
        self.B1.weight.data.normal_(0, scale)
        self.E1.weight.data.normal_(0, scale)

        self.U2.weight.data.normal_(0, scale)
        self.V2.weight.data.normal_(0, scale)
        self.A2.weight.data.normal_(0, scale)
        self.B2.weight.data.normal_(0, scale)
        self.E2.weight.data.normal_(0, scale)

        bn_init(self.bnv1)
        bn_init(self.bne1)
        bn_init(self.bnv2)
        bn_init(self.bne2)

    def forward(self, x, edge):
        # device
        dev = x.get_device()
        if dev >= 0:
            start = self.start.to(dev)
            end = self.end.to(dev)

        # GNN Layer 1:
        res = x
        Vix = self.A1(x)  # V x d_out
        Vjx = self.B1(x)  # V x d_out
        e = self.E1(edge)  # E x d_out
        edge = edge + self.act(self.bne1(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec',(start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b,self.num_classes, self.num_classes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V1(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U1(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        x = self.act(res + self.bnv1(x))
        res = x

        # GNN Layer 2:
        Vix = self.A2(x)  # V x d_out
        Vjx = self.B2(x)  # V x d_out
        e = self.E2(edge)  # E x d_out
        edge = edge + self.act(self.bne2(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec', (start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, self.num_classes, self.num_classes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V2(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U2(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        x = self.act(res + self.bnv2(x))
        return x, edge
# #
# GAT GCN Used to Learn Multi-dimensional Edge Features and Node Features



class GNNMASK(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(GNNMASK, self).__init__()
        # in_channels: dim of node feature
        # num_classes: num of nodes
        # neighbor_num: K in paper and we select the top-K nearest neighbors for each node feature.
        # metric: metric for assessing node similarity. Used in FGG module to build a dynamical graph
        # X' = ReLU(X + BN(V(X) + A x U(X)) )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

        # network
        # self.U = nn.Linear(self.in_channels,self.in_channels)
        # self.V = nn.Linear(self.in_channels,self.in_channels)
        # self.bnv = nn.BatchNorm1d(num_classes)

        # init
        # self.U.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        # self.V.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        # self.bnv.weight.data.fill_(1)
        # self.bnv.bias.data.zero_()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, n, c = x.shape

        # build dynamical graph
        # si = x.detach()

    # elif self.metric == 'cosine':
        si = x.detach()
        si = F.normalize(si, p=2, dim=-1)
        si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
        # threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
        adj = si
        # adj = (si >= threshold).float()



        # si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
        # adj = self.sigmoid(si.float())

        # si = x.detach()
        # si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
        # threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
        # adj = (si >= threshold).float()

        # GNN process
        A = normalize_digraph(adj)
        # aggregate = torch.einsum('b i j, b j k->b i k', A, self.V(x))
        # x = self.relu(x + self.bnv(aggregate + self.U(x)))
        return A


#
#
# class GNN(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(GNN, self).__init__()
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         start, end = create_e_matrix(self.num_classes)
#         self.start = Variable(start, requires_grad=False).cuda()
#         self.end = Variable(end, requires_grad=False).cuda()
#
#         dim_in = self.in_channels
#         dim_out = self.in_channels
#
#         self.fc_h1 = nn.Linear(dim_in, dim_out, bias=False)
#         self.fc_e1 = nn.Linear(dim_in, dim_out, bias=False)
#         self.fc_proj1 = nn.Linear(3 * dim_in, dim_out, bias=False)
#         self.attn_fc1 = nn.Linear(3 * dim_in, 1, bias=False)
#
#         self.fc_h2 = nn.Linear(dim_in, dim_out, bias=False)
#         self.fc_e2 = nn.Linear(dim_in, dim_out, bias=False)
#         self.fc_proj2 = nn.Linear(3 * dim_in, dim_out, bias=False)
#         self.attn_fc2 = nn.Linear(3 * dim_in, 1, bias=False)
#
#         self.softmax = nn.Softmax(2)
#
#         self.bnv1 = nn.BatchNorm1d(num_classes)
#         self.bne1 = nn.BatchNorm1d(num_classes * num_classes)
#
#         self.bnv2 = nn.BatchNorm1d(num_classes)
#         self.bne2 = nn.BatchNorm1d(num_classes * num_classes)
#
#         self.act = nn.ELU()
#         self.leaky_relu = nn.LeakyReLU()
#
#
#         self.init_weights_linear(dim_in, 1)
#
#     def init_weights_linear(self, dim_in, gain):
#         # conv1
#         scale = gain * np.sqrt(2.0 / dim_in)
#         self.fc_h1.weight.data.normal_(0, scale)
#         self.fc_e1.weight.data.normal_(0, scale)
#         self.fc_proj1.weight.data.normal_(0, scale)
#
#         self.fc_h2.weight.data.normal_(0, scale)
#         self.fc_e2.weight.data.normal_(0, scale)
#         self.fc_proj2.weight.data.normal_(0, scale)
#
#         bn_init(self.bnv1)
#         bn_init(self.bne1)
#         bn_init(self.bnv2)
#         bn_init(self.bne2)
#
#     def forward(self, x, edge):
#         start = self.start.cuda(x.get_device())
#         end = self.end.cuda(x.get_device())
#         res = x
#
#         z_h = self.fc_h1(x)  # V x d_out
#         z_e = self.fc_e1(edge)  # E x d_out
#
#         z = torch.cat((torch.einsum('ev, bvd -> bed', (start, z_h)), torch.einsum('ev, bvd -> bed', (end, z_h)), z_e),
#                       dim=-1)
#         z_e = self.fc_proj1(z)
#         attn = self.leaky_relu(self.attn_fc1(z))
#         b, _, _ = attn.shape
#         attn = attn.view(b, self.num_classes, self.num_classes, 1)
#         attn = self.softmax(attn)
#         attn = attn.view(b, -1, 1)
#
#         source_z_h = torch.einsum('ev, bvd -> bed', (start, z_h))
#         z_h = torch.einsum('ve, bed -> bvd', (end.t(), attn * source_z_h))
#         x = self.act(res + self.bnv1(z_h))
#         edge = self.act(self.bne1(z_e))
#
#         res = x
#
#         z_h = self.fc_h2(x)  # V x d_out
#         z_e = self.fc_e2(edge)  # E x d_out
#
#         z = torch.cat((torch.einsum('ev, bvd -> bed', (start, z_h)), torch.einsum('ev, bvd -> bed', (end, z_h)), z_e),
#                       dim=-1)
#         z_e = self.fc_proj2(z)
#         attn = self.leaky_relu(self.attn_fc2(z))
#         b, _, _ = attn.shape
#         attn = attn.view(b, self.num_classes, self.num_classes, 1)
#         attn = self.softmax(attn)
#         attn = attn.view(b, -1, 1)
#
#         source_z_h = torch.einsum('ev, bvd -> bed', (start, z_h))
#         z_h = torch.einsum('ve, bed -> bvd', (end.t(), attn * source_z_h))
#         x = self.act(res + self.bnv2(z_h))
#         edge = self.act(self.bne2(z_e))
#
#         return x, edge


class Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Head, self).__init__()
        # The head of network
        # Input: the feature maps x from backbone
        # Output: the AU recognition probabilities cl And the logits cl_edge of edge features for classification
        # Modules: 1. AFG extracts individual Au feature maps U_1 ---- U_N
        #          2. GEM: graph edge modeling for learning multi-dimensional edge features
        #          3. Gated-GCN for graph learning with node and multi-dimensional edge features
        # sc: individually calculate cosine similarity between node features and a trainable vector.
        # edge fc: for edge prediction

        self.in_channels = in_channels
        self.num_classes = num_classes
        class_linear_layers = []
        for i in range(self.num_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)
        self.edge_extractor = GEM(self.in_channels, self.num_classes)
        self.gnn = GNN(self.in_channels, self.num_classes)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.edge_fc = nn.Linear(self.in_channels, 4)
        self.relu = nn.ReLU()
        self.mask = GNNMASK(in_channels, num_classes)
        nn.init.xavier_uniform_(self.edge_fc.weight)
        nn.init.xavier_uniform_(self.sc)

    def forward(self, x):
        # AFG
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)
        b, n, c = f_v.shape

        # MEFL
        # f_e = self.edge_extractor(f_u, x)
        # feat_end = f_v.repeat(1, 1, n).view(b, -1, c)
        # feat_start = f_v.repeat(1, n, 1).view(b, -1, c)

        # f_e = feat_start - feat_end
        # f_e = f_e.mean(dim=-2)

        # MEFL
        f_e = self.edge_extractor(f_u, x)
        f_e = f_e.mean(dim=-2)

        # feat_end = f_v.repeat(1, 1, n).view(b, -1, c)
        # feat_start = f_v.repeat(1, n, 1).view(b, -1, c)
        # f_e = feat_start - feat_end


        mask = self.mask(f_v).view(b,n*n,1)
        f_e = f_e * mask
        f_v, f_e = self.gnn(f_v, f_e)

        sc = self.sc
        sc = self.relu(sc)
        sc = F.normalize(sc, p=2, dim=-1)
        cl = F.normalize(f_v, p=2, dim=-1)
        cl = (cl * sc.view(1, n, c)).sum(dim=-1, keepdim=False)
        cl_edge = self.edge_fc(f_e)
        return cl, cl_edge


class MEFARG(nn.Module):
    def __init__(self, num_classes=12, backbone='swin_transformer_base'):
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
        self.head = Head(self.out_channels, num_classes)

    def forward(self, x):
        # x: b d c
        x = self.backbone(x)
        x = self.global_linear(x)
        cl, cl_edge = self.head(x)
        return cl, cl_edge
