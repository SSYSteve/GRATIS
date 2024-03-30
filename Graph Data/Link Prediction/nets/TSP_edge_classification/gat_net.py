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

from layers.gated_gcn_layer import GatedGCNLayer
from layers.cross_attention_layer import CrossTransformerEncoder
from layers.transformer_layer import ResidualAttentionBlock

class CrossTransformer(nn.Module):

    def __init__(self, d_model, nhead=1, layer_nums=1, attention_type='linear'):
        super().__init__()
        
        encoder_layer = CrossTransformerEncoder(d_model, nhead, attention_type)
        self.VCR_layers = nn.ModuleList([encoder_layer for _ in range(layer_nums)])
        self.VVR_layers = nn.ModuleList([encoder_layer for _ in range(layer_nums)])
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, qfea, kfea, mask0=None, mask1=None):
        """
        Args:
            qfea (torch.Tensor): [B, N, D]
            kfea (torch.Tensor): [B, D]
            mask0 (torch.Tensor): [B, N] (optional)
            mask1 (torch.Tensor): [B, N] (optional)
        """
        
        B,N,D = qfea.shape
        kfea = kfea.unsqueeze(1).repeat(1, N, 1) #[B,N,D]
        
        mask1 = torch.ones([B,N]).to(qfea.device)
        for layer in self.VCR_layers:
            qfea = layer(qfea, kfea, mask0, mask1)
            #kfea = layer(kfea, qfea, mask1, mask0)
        
        qfea_end = qfea.repeat(1,1,N).view(B,-1,D)
        qfea_start = qfea.repeat(1,N,1).view(B,-1,D)
        #mask2 = mask0.repeat([1,N])
        for layer in self.VVR_layers:
            #qfea_start = layer(qfea_start, qfea_end, mask2, mask2)
            qfea_start = layer(qfea_start, qfea_end)

        return qfea_start.view([B,N,N,D]) #[B,N*N,D]
    

class MEFG(nn.Module):
    
    def __init__(self, in_dim,hidden_dim, max_node_num, global_layer_num = 2, dropout = 0.1):
        super().__init__()
        self.edge_proj = nn.Conv1d(in_channels=2,out_channels=1,kernel_size=3,padding=1)
        self.edge_proj2 = nn.Linear(in_dim,hidden_dim) #baseline4
        self.edge_proj3 = nn.Linear(in_dim,hidden_dim)
        self.edge_proj4 = nn.Linear(hidden_dim,hidden_dim)
        self.hidden_dim = hidden_dim #baseline4
        self.bn_node_lr_e = nn.BatchNorm1d(hidden_dim)
        
        self.max_node_num = max_node_num
        
        self.global_layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                    True, True) for _ in range(global_layer_num -1) ]) 
        self.global_layers.append(GatedGCNLayer(hidden_dim, hidden_dim, dropout, True, True))

        #self.global_layers = nn.ModuleList([ ResidualAttentionBlock( d_model = hidden_dim, n_head = 1)
        #                                    for _ in range(global_layer_num) ]) 
        
        self.CrossT = CrossTransformer(hidden_dim, nhead=1, layer_nums=1, attention_type='linear')
        
    def forward(self, g, h, e):      
        g.apply_edges(lambda edges: {'src' : edges.src['emb_h']})
        src = g.edata['src'].unsqueeze(1) #[M,1,D]
        g.apply_edges(lambda edges: {'dst' : edges.dst['emb_h']})
        dst = g.edata['dst'].unsqueeze(1) #[M,1,D]
        edge = torch.cat((src,dst),1).to(h.device) #[M,2,D]
        lr_e_local = self.edge_proj(edge).squeeze(1)#[M,D]
        lr_e_local = self.edge_proj2(lr_e_local)
        
        hs = []
        gs = dgl.unbatch(g)
        mask0 = torch.zeros([len(gs),self.max_node_num]).to(h.device)
        for i,g0 in enumerate(gs):
            Ng = g0.number_of_nodes()
            padding = nn.ConstantPad1d((0,self.max_node_num - Ng),0)
            pad_h = padding(g0.ndata['emb_h'].T).T #[Nmax, D] #feat
            hs.append(pad_h.unsqueeze(0))
            mask0[i,:Ng] = 1
        hs = torch.cat(hs,0).to(h.device) #[B,Nmax,Din]
        hs = self.edge_proj3(hs) #[B,Nmax,hidden]
        
        
        # Gated-GCN for extract global feature
        hs2g = h
        for conv in self.global_layers:
            hs2g, _ = conv(g, hs2g, e)
        g.ndata['hs2g'] = hs2g
        global_g = dgl.mean_nodes(g, 'hs2g') #[B,D]
        
        
        edge = self.CrossT(hs, global_g, mask0) #[B,N,N,D]
        
        index_edge = []
        for i,g0 in enumerate(gs):
            index_edge.append(edge[i, g0.all_edges()[0],g0.all_edges()[1],:])
        
        index_edge = torch.cat(index_edge,0)
        
        lr_e_global = self.edge_proj4(index_edge)
        
        if e is not None:
            lr_e = e + lr_e_local + lr_e_global 
        else:
            lr_e = lr_e_local + lr_e_global 
#        lr_e = lr_e_local + lr_e_global 
    
        # bn=>relu=>dropout
        lr_e = self.bn_node_lr_e(lr_e)
        lr_e = F.relu(lr_e)
        lr_e = F.dropout(lr_e, 0.1, training=self.training)
        
        return e
    
class GATNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
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
        self.MLP_layer = MLPReadout(2*out_dim, n_classes)
        
    def forward(self, g, h, e):
        h = self.embedding_h(h.float())
        h = self.in_feat_dropout(h)
        
        if self.layer_type == GATLayer:
            for conv in self.layers:
                h = conv(g, h)
        else:
            if not self.edge_feat:
                e = torch.ones_like(e).to(self.device)
            e = self.embedding_e(e.float())
            
            for conv in self.layers:
                h, e = conv(g, h, e)
        
        g.ndata['h'] = h
        
        def _edge_feat(edges):
            e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            e = self.MLP_layer(e)
            return {'e': e}
        g.apply_edges(_edge_feat)
        
        return g.edata['e']
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss(weight=None)
        loss = criterion(pred, label)

        return loss