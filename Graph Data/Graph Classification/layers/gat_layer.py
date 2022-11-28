import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""

class MERG(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.bn_node_lr_e_local = nn.BatchNorm1d(hidden_dim)
        self.bn_node_lr_e_global = nn.BatchNorm1d(hidden_dim)
        self.proj1 = nn.Linear(in_dim,hidden_dim**2)
        self.proj2 = nn.Linear(in_dim,hidden_dim)
        self.edge_proj = nn.Conv1d(in_channels=2,out_channels=1,kernel_size=3,padding=1)
        self.edge_proj2 = nn.Linear(in_dim,hidden_dim)
        self.edge_proj3 = nn.Linear(hidden_dim,hidden_dim)
        self.hidden_dim = hidden_dim
        #self.bn_local = nn.BatchNorm1d(in_dim) #baseline4'
        self.bn_local = nn.LayerNorm(in_dim)
        self.bn_global = nn.BatchNorm1d(hidden_dim) #baseline4

    def forward(self, g, h, e):
        # modified baseline4
        g.apply_edges(lambda edges: {'src' : edges.src['local']})
        src = g.edata['src'].unsqueeze(1) #[M,1,D]
        g.apply_edges(lambda edges: {'dst' : edges.dst['local']})
        dst = g.edata['dst'].unsqueeze(1) #[M,1,D]
        edge = torch.cat((src,dst),1).to(h.device) #[M,2,D]
        
        edge = self.bn_local(edge)
        lr_e_local = self.edge_proj(edge).squeeze(1)#[M,D]
        lr_e_local = F.dropout(F.relu(lr_e_local), 0.1, training=self.training)
        lr_e_local = self.edge_proj2(lr_e_local)
        
        N = h.shape[0]
        h_proj1 = F.dropout(F.relu(self.proj1(h)), 0.1, training=self.training)
        h_proj1 = h_proj1.view(-1,self.hidden_dim)
        h_proj2 = F.dropout(F.relu(self.proj2(h)), 0.1, training=self.training)
        h_proj2 = h_proj2.permute(1,0)
        mm = torch.mm(h_proj1,h_proj2)
        mm = mm.view(N,self.hidden_dim,-1).permute(0,2,1) #[N, N, D]
        lr_e_global = mm[g.all_edges()[0],g.all_edges()[1],:] #[M,D]
        
        lr_e_global = self.edge_proj3(self.bn_global(lr_e_global))
        # bn=>relu=>dropout
        lr_e_global = self.bn_node_lr_e_global(lr_e_global)
        lr_e_global = F.relu(lr_e_global)
        lr_e_global = F.dropout(lr_e_global, 0.1, training=self.training)  

        lr_e_local = self.bn_node_lr_e_local(lr_e_local)
        lr_e_local = F.relu(lr_e_local)
        lr_e_local = F.dropout(lr_e_local, 0.1, training=self.training) 
        
        e = lr_e_local + lr_e_global + e #baseline4

        return e
    
class GATLayer(nn.Module):
    """
    Parameters
    ----------
    in_dim : 
        Number of input features.
    out_dim : 
        Number of output features.
    num_heads : int
        Number of heads in Multi-Head Attention.
    dropout :
        Required for dropout of attn and feat in GATConv
    batch_norm :
        boolean flag for batch_norm layer.
    residual : 
        If True, use residual connection inside this layer. Default: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        
    Using dgl builtin GATConv by default:
    https://github.com/graphdeeplearning/benchmarking-gnns/commit/206e888ecc0f8d941c54e061d5dffcc7ae2142fc
    """    
    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, residual=False, activation=F.elu):
        super().__init__()
        self.residual = residual
        self.activation = activation
        self.batch_norm = batch_norm
            
        if in_dim != (out_dim*num_heads):
            self.residual = False

        self.gatconv = GATConv(in_dim, out_dim, num_heads, dropout, dropout)

        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_dim * num_heads)

    def forward(self, g, h):
        h_in = h # for residual connection

        h = self.gatconv(g, h).flatten(1)
            
        if self.batch_norm:
            h = self.batchnorm_h(h)
            
        if self.activation:
            h = self.activation(h)
            
        if self.residual:
            h = h_in + h # residual connection

        return h
    

##############################################################
#
# Additional layers for edge feature/representation analysis
#
##############################################################


class CustomGATHeadLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, batch_norm):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']
        
        if self.batch_norm:
            h = self.batchnorm_h(h)
        
        h = F.elu(h)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h

    
class CustomGATLayer(nn.Module):
    """
        Param: [in_dim, out_dim, n_heads]
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, residual=True):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.residual = residual

        if in_dim != (out_dim*num_heads):
            self.residual = False

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(CustomGATHeadLayer(in_dim, out_dim, dropout, batch_norm))
        self.merge = 'cat' 

    def forward(self, g, h, e):
        h_in = h # for residual connection
        
        head_outs = [attn_head(g, h) for attn_head in self.heads]

        if self.merge == 'cat':
            h = torch.cat(head_outs, dim=1)
        else:
            h = torch.mean(torch.stack(head_outs))

        if self.residual:
            h = h_in + h # residual connection
        
        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)

    
##############################################################


class CustomGATHeadLayerEdgeReprFeat(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, batch_norm, edge_lr=False):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        self.fc_h = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_e = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_proj = nn.Linear(3* out_dim, out_dim)
        self.attn_fc = nn.Linear(3* out_dim, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.batchnorm_e = nn.BatchNorm1d(out_dim)
        
        #self.edge_lr = edge_lr
        #if self.edge_lr:
        #    self.merg = MERG(in_dim, out_dim)

    def edge_attention(self, edges):
        z = torch.cat([edges.data['z_e'], edges.src['z_h'], edges.dst['z_h']], dim=1)
        e_proj = self.fc_proj(z)
        attn = F.leaky_relu(self.attn_fc(z))
        return {'attn': attn, 'e_proj': e_proj}

    def message_func(self, edges):
        return {'z': edges.src['z_h'], 'attn': edges.data['attn']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['attn'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}
    
    def forward(self, g, h, e):
        
        #g.ndata['local']  = h
        #if self.edge_lr:
        #    e = self.merg(g, h, e)
            
        z_h = self.fc_h(h)
        z_e = self.fc_e(e)
        g.ndata['z_h'] = z_h
        g.edata['z_e'] = z_e
        
        g.apply_edges(self.edge_attention)
        
        g.update_all(self.message_func, self.reduce_func)
        
        h = g.ndata['h']
        e = g.edata['e_proj']
        
        if self.batch_norm:
            h = self.batchnorm_h(h)
            e = self.batchnorm_e(e)
        
        h = F.elu(h)
        e = F.elu(e)
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e
    

class CustomGATLayerEdgeReprFeat(nn.Module):
    """
        Param: [in_dim, out_dim, n_heads]
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, residual=True, edge_lr=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.residual = residual

        if in_dim != (out_dim*num_heads):
            self.residual = False

        self.heads = nn.ModuleList()

        for i in range(num_heads):
            self.heads.append(CustomGATHeadLayerEdgeReprFeat(in_dim, out_dim, dropout, batch_norm))
        self.merge = 'cat' 
        
        self.edge_lr = edge_lr
        if self.edge_lr:
            self.merg = MERG(in_dim, in_dim)

    def forward(self, g, h, e):
        
        g.ndata['local']  = h
        if self.edge_lr:
            e = self.merg(g, h, e)
            
        h_in = h # for residual connection
        e_in = e

        head_outs_h = []
        head_outs_e = []
        for attn_head in self.heads:
            h_temp, e_temp = attn_head(g, h, e)
            head_outs_h.append(h_temp)
            head_outs_e.append(e_temp)

        if self.merge == 'cat':
            h = torch.cat(head_outs_h, dim=1)
            e = torch.cat(head_outs_e, dim=1)
        else:
            raise NotImplementedError

        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e

        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)

    
##############################################################


class CustomGATHeadLayerIsotropic(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, batch_norm):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)

    def message_func(self, edges):
        return {'z': edges.src['z']}

    def reduce_func(self, nodes):
        h = torch.sum(nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        z = self.fc(h)
        g.ndata['z'] = z
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']
        
        if self.batch_norm:
            h = self.batchnorm_h(h)
        
        h = F.elu(h)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h

    
class CustomGATLayerIsotropic(nn.Module):
    """
        Param: [in_dim, out_dim, n_heads]
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, residual=True):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.residual = residual

        if in_dim != (out_dim*num_heads):
            self.residual = False

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(CustomGATHeadLayerIsotropic(in_dim, out_dim, dropout, batch_norm))
        self.merge = 'cat' 

    def forward(self, g, h, e):
        h_in = h # for residual connection
        
        head_outs = [attn_head(g, h) for attn_head in self.heads]

        if self.merge == 'cat':
            h = torch.cat(head_outs, dim=1)
        else:
            h = torch.mean(torch.stack(head_outs))

        if self.residual:
            h = h_in + h # residual connection
        
        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
