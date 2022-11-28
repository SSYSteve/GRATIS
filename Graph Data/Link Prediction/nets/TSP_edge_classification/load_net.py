"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.TSP_edge_classification.gated_gcn_net import GatedGCNNet
from nets.TSP_edge_classification.gat_net import GATNet


def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GAT': GAT,
    }
        
    return models[MODEL_NAME](net_params)