import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import dgl

# We note the code here follows the exact description of our paper.
# However, for different tasks, the code might need some tweaking.
# For easier reproduction of our reported results, we included the task-specific code in separate folders.


# 

# We start with a randomly initialized graph.
    # We first create a simple graph with four nodes and five edges.
    # We then assign 1*3 random values for the node feature denoted as 'feat'.
    # We also assign 1*1 random values for the edge feature denoted as 'feat'.


##############################################################################################
####################################  Input Graph D_in  ######################################
##############################################################################################

u, v = torch.tensor([0, 0, 0 , 1, 1]), torch.tensor([1, 2, 3, 2, 3])
g = dgl.graph((u, v))
d_n = 3
d_f = 1
g.ndata['feat'] = torch.rand(g.num_nodes(), d_n)
g.edata['feat'] = torch.rand(g.num_edges(), d_f)

print('Original D_in   ', g)

##############################################################################################
####################################  Graph Definition  ######################################
##############################################################################################

from gd import GD

graph_definition = GD()

input_graph, input_node_fea, input_edge_fea = graph_definition(g)


##############################################################################################
########################  Task-specific Topology Prediction  #################################
##############################################################################################

from ttp import TTP

ttp = TTP(in_dim = 3, hidden_dim = 13, edge_thresh=0.2)

after_ttp_graph = ttp(input_graph, input_node_fea, input_edge_fea)
print('Basic graph   ', input_graph)
print('TTP learned graph   ', after_ttp_graph)

ttp_node = after_ttp_graph.ndata['feat'][:]
ttp_feature = after_ttp_graph.edata['feat']

in_dim = 3
hidden_dim = 13
embedding_h = nn.Linear(in_dim, hidden_dim)
embed_ttp_node = embedding_h(ttp_node)


##############################################################################################
########################  Multi-dimensional Edge Feature Generation  #########################
##############################################################################################
# 
from mefg import MEFG
max_node_num = 100
mefg = MEFG(in_dim,hidden_dim, max_node_num)
learned_multi_edge = mefg(after_ttp_graph,embed_ttp_node,e=None)

print('Oringinal Edge Feature   ', input_edge_fea)
print('TTP Edge Feature', ttp_feature)
print('MEFG learned Multiple Edge Feature   ', learned_multi_edge)

# Further Analysis ...