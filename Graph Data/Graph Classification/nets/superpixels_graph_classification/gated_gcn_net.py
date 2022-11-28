import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import dgl

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
        #assert self.d_model == qfea.size(2), "the feature number of src and transformer must be equal"
        
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
        self.proj_g1 = nn.Linear(in_dim,hidden_dim**2) #lr_g
        self.bn_node_lr_g1 = nn.BatchNorm1d(hidden_dim**2)
        self.proj_g2 = nn.Linear(in_dim,hidden_dim) #lr_g
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

        #self.global_layers = nn.ModuleList([ ResidualAttentionBlock( d_model = hidden_dim, n_head = 1)
        #                                    for _ in range(global_layer_num) ]) 
        
        self.CrossT = CrossTransformer(hidden_dim, nhead=1, layer_nums=1, attention_type='linear')
        
    def forward(self, g, h, e):
        
        g.apply_edges(lambda edges: {'src' : edges.src['feat']})
        src = g.edata['src'].unsqueeze(1) #[M,1,D]
        g.apply_edges(lambda edges: {'dst' : edges.dst['feat']})
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
            pad_h = padding(g0.ndata['feat'].T).T #[Nmax, D]
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
        
        if e is not None:
            e = e + lr_e_local + lr_e_global 
        else:
            e = lr_e_local + lr_e_global 
#        lr_e = lr_e_local + lr_e_global 
    
        # bn=>relu=>dropout
        e = self.bn_node_lr_e(e)
        e = F.relu(e)
        e = F.dropout(e, 0.1, training=self.training)
        
        return e
        
class GatedGCNNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                    self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        
        max_node_num = net_params['node_num']
        self.gtp = GTP(in_dim, hidden_dim, edge_thresh=0.3)
        self.merg = MERG(in_dim,hidden_dim, max_node_num)

        
    def forward(self, g, h, e):
        '''
        g [B*M]
        h [B*N,D]
        e [B*M,1]
        '''
        lr_g = self.gtp(g,h,e)
        g = lr_g
        
        h = self.embedding_h(h)
        #if not self.edge_feat: # edge feature set to 1
        #    e = torch.ones_like(e).to(self.device)
        #e = self.embedding_e(e)
        e = None
        
        #import pdb;pdb.set_trace()
        lr_e = self.merg(g,h,e)
        e = lr_e
        
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
        
        # hg:[B,hidden_dim]
            
        return self.MLP_layer(hg)
        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss