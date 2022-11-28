import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from layers.gat_layer import GATLayer, CustomGATLayer, CustomGATLayerEdgeReprFeat, CustomGATLayerIsotropic
from layers.mlp_readout_layer import MLPReadout

class MERG(nn.Module):
    
    def __init__(self, in_dim,hidden_dim, dropout = 0.1):
        super().__init__()
        
        
        self.proj1 = nn.Linear(235868,512) #baseline4
        self.proj2 = nn.Linear(512,2358104) #baseline4
        self.proj3 = nn.Linear(in_dim,hidden_dim) #baseline4
        self.edge_proj = nn.Conv1d(in_channels=2,out_channels=1,kernel_size=3,padding=1) #baseline4
        self.edge_proj2 = nn.Linear(in_dim,hidden_dim) #baseline4
        self.bn_node_lr_e = nn.BatchNorm1d(hidden_dim) #baseline4
        self.dropout = dropout
        
    def forward(self, g, h, e):
        g.apply_edges(lambda edges: {'src' : edges.src['feat']})
        src = g.edata['src'].unsqueeze(1) #[M,1,D]
        g.apply_edges(lambda edges: {'dst' : edges.dst['feat']})
        dst = g.edata['dst'].unsqueeze(1) #[M,1,D]
        edge = torch.cat((src,dst),1).to(h.device).float() #[M,2,D]
        lr_e_local = self.edge_proj(edge).squeeze(1)#[M,D]
        lr_e_local = self.edge_proj2(lr_e_local)
        h_e = h.permute(1,0)
        h_e = self.proj1(h_e)
        h_e = F.relu(h_e)
        h_e = self.proj2(h_e)
        F.dropout(h_e, self.dropout, training=self.training)
        
        h_e = h_e.permute(1,0)
        lr_e_global = self.proj3(h_e)
        
        
        # input embedding
#        e = self.embedding_e(e.float()) + lr_e_local #baseline1-3
        e = self.embedding_e(e.float()) + lr_e_local + lr_e_global #baseline4        
        
        # bn=>relu=>dropout
        e = self.bn_node_lr_e(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return e
    
class GATNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.device = net_params['device']
        
        self.layer_type = {
            "dgl": GATLayer,
            "edgereprfeat": CustomGATLayerEdgeReprFeat,
            "edgefeat": CustomGATLayer,
            "isotropic": CustomGATLayerIsotropic,
        }.get(net_params['layer_type'], GATLayer)
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim * num_heads)
        
        if self.layer_type != GATLayer:
            self.edge_feat = net_params['edge_feat']
            self.embedding_e = nn.Linear(in_dim_edge, hidden_dim * num_heads)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([self.layer_type(hidden_dim * num_heads, hidden_dim, num_heads,
                                                     dropout, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(self.layer_type(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(2*out_dim, 1)
        
        self.merg = MERG(in_dim, hidden_dim)
        
    def forward(self, g, h, e):
        
        e = self.merg(g,h,e)
        
        h = self.embedding_h(h.float())
        h = self.in_feat_dropout(h)
        
        if self.layer_type == GATLayer:
            for conv in self.layers:
                h = conv(g, h)
        else:
            #if not self.edge_feat:
            #    e = torch.ones_like(e).to(self.device)
            e = self.embedding_e(e.float())
            
            for conv in self.layers:
                h, e = conv(g, h, e)
        
        g.ndata['h'] = h
        
        return h
    
    def edge_predictor(self, h_i, h_j):
        x = torch.cat([h_i, h_j], dim=1)
        x = self.MLP_layer(x)
        
        return torch.sigmoid(x)
    
    def loss(self, pos_out, neg_out):
        pos_loss = -torch.log(pos_out + 1e-15).mean()  # positive samples
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()  # negative samples
        loss = pos_loss + neg_loss
        
        return loss
