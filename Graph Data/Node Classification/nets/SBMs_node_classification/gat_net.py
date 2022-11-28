import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from layers.gat_layer import GATLayer, CustomGATLayerEdgeReprFeat, CustomGATLayer
from layers.gated_gcn_layer import GatedGCNLayer
from layers.mlp_readout_layer import MLPReadout

from layers.cross_attention_layer import CrossTransformerEncoder
from layers.transformer_layer import ResidualAttentionBlock

class CrossTransformer(nn.Module):

    def __init__(self, d_model, nhead=1, layer_nums=1, attention_type='linear'):
        super().__init__()
        
        encoder_layer = CrossTransformerEncoder(d_model, nhead, attention_type)
        self.FAM_layers = nn.ModuleList([encoder_layer for _ in range(layer_nums)])
        self.ARM_layers = nn.ModuleList([encoder_layer for _ in range(layer_nums)])
        
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
        for layer in self.FAM_layers:
            qfea = layer(qfea, kfea, mask0, mask1)
            #kfea = layer(kfea, qfea, mask1, mask0)
        
        qfea_end = qfea.repeat(1,1,N).view(B,-1,D)
        qfea_start = qfea.repeat(1,N,1).view(B,-1,D)
        #mask2 = mask0.repeat([1,N])
        for layer in self.ARM_layers:
            #qfea_start = layer(qfea_start, qfea_end, mask2, mask2)
            qfea_start = layer(qfea_start, qfea_end)

        return qfea_start.view([B,N,N,D]) #[B,N*N,D]
    
class GTP(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_thresh):
        super().__init__()
        self.proj_g1 = nn.Embedding(in_dim,hidden_dim**2) #lr_g
        self.bn_node_lr_g1 = nn.BatchNorm1d(hidden_dim**2)
        self.proj_g2 = nn.Embedding(in_dim,hidden_dim) #lr_g
        self.bn_node_lr_g2 = nn.BatchNorm1d(hidden_dim)
        self.hidden_dim = hidden_dim #lr_g
        self.proj_g = nn.Linear(hidden_dim, 1)
        self.edge_thresh = edge_thresh
    def forward(self, g, h, e):
        lr_gs = []
        gs = dgl.unbatch(g)
        for g in gs:
            N = g.number_of_nodes()
            h_single = g.ndata['feat'].to(h.device)
            h_proj1 = F.dropout(F.relu(self.bn_node_lr_g1(self.proj_g1(h_single))), 0.1, training=self.training).view(-1,self.hidden_dim)
            h_proj2 = F.dropout(F.relu(self.bn_node_lr_g2(self.proj_g2(h_single))), 0.1, training=self.training).permute(1,0)
            mm = torch.mm(h_proj1,h_proj2)
            mm = mm.view(N,self.hidden_dim,-1).permute(0,2,1) #[N, N, D]            
        
            mm = self.proj_g(mm).squeeze(-1)
            diag_mm = torch.diag(mm)
            diag_mm = torch.diag_embed(diag_mm)
            mm -= diag_mm  
            #matrix = torch.sigmoid(mm)
            #matrix = F.softmax(mm, dim=0)
            matrix = F.softmax(mm, dim=0) * F.softmax(mm, dim=1)
            
            #binarized = BinarizedF()
            #matrix = binarized.apply(matrix) #(0/1)
            lr_connetion = torch.where(matrix>self.edge_thresh)

            g.add_edges(lr_connetion[0], lr_connetion[1])
            lr_gs.append(g)    
        g = dgl.batch(lr_gs).to(h.device)

        return g    

class MERG(nn.Module):
    
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
        
        if e is None:
            e = torch.ones([g.number_of_edges() ,h.shape[-1]]).to(h.device)
        # Gated-GCN for extract global feature
        hs2g = h
        for conv in self.global_layers:
            hs2g, _ = conv(g, hs2g, e)
        g.ndata['hs2g'] = hs2g
        global_g = dgl.mean_nodes(g, 'hs2g') #[B,D]
        
        '''
        # Transformer for extract global feature
        mask_t = mask0.unsqueeze(1)*mask0.unsqueeze(2)
        mask_t = (mask_t==0)
        #mask_t = None
        
        hs2g = hs.permute((1,0,2))
        for conv in self.global_layers:
            hs2g = conv(hs2g, mask_t)
        global_g = hs2g.permute((1,0,2)).mean(1) #[B,D]
        '''
        
        edge = self.CrossT(hs, global_g, mask0) #[B,N,N,D]
        index_edge = []
        for i,g0 in enumerate(gs):
            index_edge.append(edge[i, g0.all_edges()[0],g0.all_edges()[1],:])
        index_edge = torch.cat(index_edge,0)
        
        lr_e_global = self.edge_proj4(index_edge)
        lr_e = lr_e_local + lr_e_global 
    
        # bn=>relu=>dropout
        lr_e = self.bn_node_lr_e(lr_e)
        lr_e = F.relu(lr_e)
        lr_e = F.dropout(lr_e, 0.1, training=self.training)
        
        return e

class GATNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
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
        
        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim * num_heads) # node feat is an integer
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([CustomGATLayer(hidden_dim * num_heads, hidden_dim, num_heads,
                                              dropout, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(CustomGATLayer(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        
        max_node_num = net_params['node_num']
        self.gtp = GTP(in_dim_node, hidden_dim * num_heads, edge_thresh=0.8)
        self.merg = MERG(hidden_dim * num_heads,hidden_dim * num_heads, max_node_num)
        

    def forward(self, g, h, e):

    
        lr_g = self.gtp(g,h,e)
        g = lr_g
        
        h = self.embedding_h(h)
        g.ndata['emb_h'] = h

        e = None
        lr_e = self.merg(g,h,e)
        e = lr_e
        
        # res gated convnets
        for conv in self.layers:
            h, e = conv(g, h, e)

        # output
        h_out = self.MLP_layer(h)

        return h_out
    
    
    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss



        
