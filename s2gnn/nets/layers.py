__author__ = "Jhony H. Giraldo"
__license__ = "MIT"

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class CascadeLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kwargs):
        super(CascadeLayer, self).__init__()

        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.convs = torch.nn.ModuleList()

        for _ in range(kwargs["alpha"]):
            self.convs.append(
                GCNConv(
                    in_channels, 
                    out_channels, 
                    cached=False, 
                    add_self_loops=False
                )
            )

        self.dropout = kwargs["dropout"]

    def forward(self, x, data):
        edge_indexs, edge_attrs = data.edge_index, data.edge_attr
        hs = []
        h = self.lin(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        hs.append(h)

        for i, conv in enumerate(self.convs):
            h = conv(x, edge_indexs[i], edge_weight=edge_attrs[i])
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)

        return hs


class LinearCombinationLayer(torch.nn.Module):
    def __init__(self, alpha):
        super(LinearCombinationLayer, self).__init__()

        self.params = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(1)) 
            for _ in range(alpha+1)
        ])

    def forward(self, hs):
        output = 0
        for i, param in enumerate(self.params):
            output = output + param * hs[i]
        return output


class ConcatLinearTransformationLayer(torch.nn.Module):
    def __init__(self, alpha, in_channels, out_channels):
        super(ConcatLinearTransformationLayer, self).__init__()

        self.lin = torch.nn.Linear((alpha + 1) * in_channels, out_channels)

    def forward(self, hs):
        x = hs[0]

        for i in range(1, len(hs)):
            x = torch.cat((x, hs[i]), 1)

        output = self.lin(x)

        return output