import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import dgl

# We note the code here follows the exact description of our paper.
# However, for different tasks, the code might need some tweaking.
# For easier reproduction of our reported results, we included the task-specific code in separate folders.

##############################################################################################
####################################  Input Graph D_in  ######################################
##############################################################################################

# We start with a randomly initialized 224 * 224 graph.

input_image = torch.randn((1,3,224,224))

##############################################################################################
####################################  Graph Definition  ######################################
##############################################################################################

from gd import GD

graph_definition = GD()

input_graph, input_node_fea, input_edge_fea = graph_definition(input_image)
print('Basic graph   ', input_graph)

##############################################################################################
########################  Task-specific Topology Prediction  #################################
##############################################################################################

from ttp import TTP

ttp = TTP()

node_feature_before_gap, global_feature, after_ttp_graph = ttp(input_image, None, False)

ttp_node = after_ttp_graph.ndata['feat'][:]
ttp_feature = after_ttp_graph.edata['feat']

print('TTP learned graph   ', after_ttp_graph)
print('Oringinal Edge Feature   ', input_edge_fea)
print('TTP Edge Feature', ttp_feature)

##############################################################################################
########################  Multi-dimensional Edge Feature Generation  #########################
##############################################################################################

from mefg import MEFG, GNNMASK, GNN
dimens = 512
nodes_num = 4
mefg = MEFG(dimens, nodes_num)
f_e = mefg(node_feature_before_gap, global_feature)
f_e = f_e.mean(dim=-2)

print('MEFG learned Multiple Edge Feature   ', f_e.shape)

# Further Analysis ...

mask = GNNMASK(dimens, nodes_num)
gnn = GNN(dimens, nodes_num)

mask = mask(ttp_node.unsqueeze(0)).view(1,nodes_num*nodes_num,1)
f_e = f_e * mask
f_v, f_e = gnn(ttp_node.unsqueeze(0), f_e)
