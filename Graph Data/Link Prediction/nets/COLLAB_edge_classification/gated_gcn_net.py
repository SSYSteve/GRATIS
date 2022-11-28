import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import dgl

from layers.gated_gcn_layer import GatedGCNLayer, GatedGCNLayerEdgeFeatOnly, GatedGCNLayerIsotropic
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
        
    
    
class GatedGCNNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.pos_enc = net_params['pos_enc']
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        
        self.layer_type = {
            "edgereprfeat": GatedGCNLayer,
            "edgefeat": GatedGCNLayerEdgeFeatOnly,
            "isotropic": GatedGCNLayerIsotropic,
        }.get(net_params['layer_type'], GatedGCNLayer)
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        self.layers = nn.ModuleList([ self.layer_type(hidden_dim, hidden_dim, dropout,
                                                      self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(self.layer_type(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        
        self.MLP_layer = MLPReadout(2*out_dim, 1)

        self.merg = MERG(in_dim, hidden_dim)

        
    def forward(self, g, h, e, h_pos_enc=None):
        #print('h',h.shape) #[235868, 128]
        #print('e',e.shape) #[2358104, 2]

        e = self.merg(g,h,e)
        
        h = self.embedding_h(h.float())
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float()) 
            h = h + h_pos_enc
        #if not self.edge_feat:
        #    e = torch.ones_like(e).to(self.device)
        
        # convnets
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
    