import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import dgl

class TTP(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_thresh):
        super().__init__()
        self.proj_g1 = nn.Linear(in_dim,hidden_dim**2)
        self.bn_node_lr_g1 = nn.BatchNorm1d(hidden_dim**2)
        self.proj_g2 = nn.Linear(in_dim,hidden_dim)
        self.bn_node_lr_g2 = nn.BatchNorm1d(hidden_dim)
        self.hidden_dim = hidden_dim #lr_g
        self.proj_g = nn.Linear(hidden_dim, 1)
        self.edge_thresh = edge_thresh                       # theta, the threshold
    def forward(self, g, h, e):
        lr_gs = []
        gs = dgl.unbatch(g)
        for g in gs:
            N = g.number_of_nodes()
            h_single = g.ndata['feat'].to(h.device)
            # h_single is X_GCN 
            # print('X_GCN', h_single.shape)            # [N,K]
            
            h_proj1 = F.dropout(F.relu(self.bn_node_lr_g1(self.proj_g1(h_single))), 0.1, training=self.training).view(-1,self.hidden_dim)
            
            
            # h_proj1 is M_1 
            # print('M_1', h_proj1.shape)            # [N,D^2]   is viwed in [N * D, D], where D is hidden_dim
            
            
            h_proj2 = F.dropout(F.relu(self.bn_node_lr_g2(self.proj_g2(h_single))), 0.1, training=self.training).permute(1,0)
            
            
            # h_proj2 is M_2 
            # print('M_2', h_proj2.shape)            # [N,D]
            
            mm = torch.mm(h_proj1,h_proj2)
            mm = mm.view(N,self.hidden_dim,-1).permute(0,2,1)     # mm is the global contextual representation
            
            # print('X', mm.shape)                   #[N, N, D]
              
            mm = self.proj_g(mm).squeeze(-1)       #𝑋^ℎ=ℎ(𝑋) [N, N, D] -> [N, N]
            
            matrix = F.softmax(mm, dim=0) * F.softmax(mm, dim=1)
            
            diag_mm = torch.diag(mm)
            
            diag_mm = torch.diag_embed(diag_mm)
            
            mm -= diag_mm                          # substracting the diag elements
            
            
            matrix = F.softmax(mm, dim=0) * F.softmax(mm, dim=1)
            
            lr_connetion = torch.where(matrix>self.edge_thresh)
            # print('new connections:  ', lr_connetion[0], lr_connetion[1])
            g.add_edges(lr_connetion[0], lr_connetion[1])
            lr_gs.append(g)    
        g = dgl.batch(lr_gs).to(h.device)

        return g