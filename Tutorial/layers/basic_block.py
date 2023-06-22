import torch
import torch.nn as nn
import math

def bn_init(bn):
    bn.weight.data.fill_(1)
    bn.bias.data.zero_()


class LinearBlock(nn.Module):
    def __init__(self, in_features,out_features=None,drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.fc.weight.data.normal_(0, math.sqrt(2. / out_features))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x)#.permute(0, 2, 1)
        x = self.relu(self.bn(x))#.permute(0, 2, 1)
        return x

class LinearBlock_3d(nn.Module):
    def __init__(self, in_features,out_features, norm_dim ,drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.LayerNorm(norm_dim)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.fc.weight.data.normal_(0, math.sqrt(2. / out_features))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.drop(x) #[B,N,D]
        x = self.fc(x).permute(1, 0, 2)
        x = self.relu(self.bn(x).permute(1, 0, 2))
        return x