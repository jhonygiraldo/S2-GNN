__author__ = "Jhony H. Giraldo"
__license__ = "MIT"

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (
    GCNConv,
    ChebConv,
    GraphConv,
    GATConv,
    TransformerConv,
    SGConv,
    ClusterGCNConv,
    FiLMConv, 
    SuperGATConv,
    GATv2Conv,
    ARMAConv,
    global_mean_pool
)

from s2gnn.nets.layers import (
    CascadeLayer,
    LinearCombinationLayer,
    ConcatLinearTransformationLayer
)


class SSobGNN(torch.nn.Module):
    r"""Parametrized S-SobGNN model.
    args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        kwargs (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, kwargs):
        super(SSobGNN, self).__init__()

        self.aggregation = kwargs["aggregation"]
        self.graph_classification = kwargs["graph_classification"]

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            CascadeLayer(
                in_channels,
                kwargs["hidden_units"],
                kwargs
            )
        )

        for _ in range(number_layers - 2):
            self.convs.append(
                CascadeLayer(
                    kwargs["hidden_units"],
                    kwargs["hidden_units"],
                    kwargs
                )
            )

        self.convs.append(
            CascadeLayer(
                kwargs["hidden_units"],
                out_channels,
                kwargs
                )
            )

        if kwargs["aggregation"] == 'linear':
            self.linear_combination_layers = torch.nn.ModuleList()

            for _ in range(number_layers):
                self.linear_combination_layers.append(
                    LinearCombinationLayer(
                        kwargs["alpha"]
                    )
                )
    
        if kwargs["aggregation"] == 'concat':
            self.concat_layers = torch.nn.ModuleList()

            for _ in range(number_layers-1):
                self.concat_layers.append(
                    ConcatLinearTransformationLayer(
                        kwargs["alpha"],
                        kwargs["hidden_units"],
                        kwargs["hidden_units"]
                    )
                )

            self.concat_layers.append(
                ConcatLinearTransformationLayer(
                    kwargs["alpha"],
                    out_channels,
                    out_channels
                )
            )

        if self.graph_classification:
            self.pool = global_mean_pool

    def forward(self, data):
        x = data.x

        for i, conv_layer in enumerate(self.convs):
            hs = conv_layer(x, data)

            if self.aggregation == 'linear':
                x = self.linear_combination_layers[i](hs)

            if self.aggregation == 'concat':
                x = self.concat_layers[i](hs)

        if self.graph_classification:
            x = self.pool(x, data.batch)

        return x.log_softmax(dim=-1)


class SIGN(torch.nn.Module):
    r"""Parametrized SIGN model.
    args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        kwargs (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, kwargs):
        super(SIGN, self).__init__()

        self.graph_classification = kwargs["graph_classification"]

        if self.graph_classification:
            self.pool = global_mean_pool

        self.convs = torch.nn.ModuleList()

        for _ in range(kwargs["k_sign"]):
            self.convs.append(
                Linear(
                    in_channels,
                    kwargs["hidden_units"]
                )
            )

        self.lin = Linear(
            kwargs["k_sign"] * kwargs["hidden_units"],
            out_channels
        )
        self.dropout = kwargs["dropout"]

    def forward(self, data):
        xs = data.xs
        hs = []

        for x, lin in zip(xs, self.convs):
            h = lin(x)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)

        h = torch.cat(hs, dim=-1)
        h = self.lin(h)

        if self.graph_classification:
            h = self.pool(h, data.batch)

        return h.log_softmax(dim=-1)


class GCN(torch.nn.Module):
    r"""Parametrized GCN model.
    args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        kwargs (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, kwargs):
        super(GCN, self).__init__()

        self.graph_classification = kwargs["graph_classification"]

        if self.graph_classification:
            self.pool = global_mean_pool

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(
                in_channels, 
                kwargs["hidden_units"], 
                cached=True
            )
        )

        for _ in range(number_layers - 2):
            self.convs.append(
                GCNConv(
                    kwargs["hidden_units"], 
                    kwargs["hidden_units"], 
                    cached=True
                )
            )

        self.convs.append(
            GCNConv(
                kwargs["hidden_units"], 
                out_channels, 
                cached=True
            )
        )
        self.dropout = kwargs["dropout"]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.graph_classification:
            x = self.pool(x, data.batch)

        return x.log_softmax(dim=-1)


class Cheby(torch.nn.Module):
    r"""Parametrized ChebyConv model.
    args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        kwargs (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, kwargs):
        super(Cheby, self).__init__()

        self.K = kwargs["K_Cheby"]
        self.graph_classification = kwargs["graph_classification"]

        if self.graph_classification:
            self.pool = global_mean_pool

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            ChebConv(
                in_channels, 
                kwargs["hidden_units"], 
                K=self.K
            )
        )

        for _ in range(number_layers - 2):
            self.convs.append(
                ChebConv(
                    kwargs["hidden_units"],
                    kwargs["hidden_units"],
                    K=self.K
                )
            )

        self.convs.append(
            ChebConv(
                kwargs["hidden_units"],
                out_channels,
                K=self.K
            )
        )
        self.dropout = kwargs["dropout"]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.graph_classification:
            x = self.pool(x, data.batch)

        return x.log_softmax(dim=-1)


class kGNN(torch.nn.Module):
    r"""Parametrized k-GNN model.
    args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        kwargs (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, kwargs):
        super(kGNN, self).__init__()

        self.graph_classification = kwargs["graph_classification"]

        if self.graph_classification:
            self.pool = global_mean_pool

        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConv(in_channels, kwargs["hidden_units"]))

        for _ in range(number_layers - 2):
            self.convs.append(
                GraphConv(
                    kwargs["hidden_units"],
                    kwargs["hidden_units"]
                )
            )

        self.convs.append(
            GraphConv(
                kwargs["hidden_units"],
                out_channels
                )
            )

        self.dropout = kwargs["dropout"]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.graph_classification:
            x = self.pool(x, data.batch)

        return x.log_softmax(dim=-1)


class GAT(torch.nn.Module):
    r"""Parametrized GAT model.
    args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        kwargs (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, kwargs):
        super(GAT, self).__init__()

        self.graph_classification = kwargs["graph_classification"]

        if self.graph_classification:
            self.pool = global_mean_pool

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATConv(
                in_channels,
                kwargs["hidden_units"],
                heads=int(kwargs["heads_attention"]),
                concat=False
                )
            )

        for _ in range(number_layers - 2):
            self.convs.append(
                GATConv(
                    int(kwargs["hidden_units"]),
                    kwargs["hidden_units"],
                    heads=int(kwargs["heads_attention"]),
                    concat=False
                    )
                )

        self.convs.append(
            GATConv(
                int(kwargs["hidden_units"]),
                out_channels,
                heads=int(kwargs["heads_attention"]),
                concat=False
                )
            )
        self.dropout = kwargs["dropout"]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.graph_classification:
            x = self.pool(x, data.batch)

        return x.log_softmax(dim=-1)


class Transformer(torch.nn.Module):
    r"""Parametrized Transformer model.
    args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        kwargs (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, kwargs):
        super(Transformer, self).__init__()

        self.graph_classification = kwargs["graph_classification"]

        if self.graph_classification:
            self.pool = global_mean_pool

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            TransformerConv(
                in_channels,
                kwargs["hidden_units"],
                heads=int(kwargs["heads_attention"]),
                concat=False
                )
            )

        for _ in range(number_layers - 2):
            self.convs.append(
                TransformerConv(
                    int(kwargs["hidden_units"]),
                    kwargs["hidden_units"],
                    heads=int(kwargs["heads_attention"]),
                    concat=False
                    )
                )

        self.convs.append(
            TransformerConv(
                int(kwargs["hidden_units"]),
                out_channels,
                heads=int(kwargs["heads_attention"]),
                concat=False
            )
        )
        self.dropout = kwargs["dropout"]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.graph_classification:
            x = self.pool(x, data.batch)

        return x.log_softmax(dim=-1)


class SGC(torch.nn.Module):
    r"""Parametrized SGC model.
    args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        kwargs (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, kwargs):
        super(SGC, self).__init__()

        self.graph_classification = kwargs["graph_classification"]

        if self.graph_classification:
            self.pool = global_mean_pool

        self.conv = SGConv(
            in_channels, 
            out_channels, 
            K=number_layers, 
            cached=True
        )
        self.dropout = kwargs["dropout"]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv(x, edge_index, edge_weight=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.graph_classification:
            x = self.pool(x, data.batch)

        return x.log_softmax(dim=-1)


class ClusterGCN(torch.nn.Module):
    r"""Parametrized ClusterGCNConv model.
    args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        kwargs (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, kwargs):
        super(ClusterGCN, self).__init__()

        self.graph_classification = kwargs["graph_classification"]

        if self.graph_classification:
            self.pool = global_mean_pool

        self.convs = torch.nn.ModuleList()
        self.convs.append(ClusterGCNConv(in_channels, kwargs["hidden_units"]))

        for _ in range(number_layers - 2):
            self.convs.append(
                ClusterGCNConv(
                    kwargs["hidden_units"], 
                    kwargs["hidden_units"]
                )
            )

        self.convs.append(ClusterGCNConv(kwargs["hidden_units"], out_channels))
        self.dropout = kwargs["dropout"]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.graph_classification:
            x = self.pool(x, data.batch)

        return x.log_softmax(dim=-1)


class FiLM(torch.nn.Module):
    r"""Parametrized FiLM model.
    args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        kwargs (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, kwargs):
        super(FiLM, self).__init__()

        self.graph_classification = kwargs["graph_classification"]

        if self.graph_classification:
            self.pool = global_mean_pool

        self.convs = torch.nn.ModuleList()
        self.convs.append(FiLMConv(in_channels, kwargs["hidden_units"]))

        for _ in range(number_layers - 2):
            self.convs.append(
                FiLMConv(
                    kwargs["hidden_units"],
                    kwargs["hidden_units"]
                )
            )

        self.convs.append(FiLMConv(kwargs["hidden_units"], out_channels))
        self.dropout = kwargs["dropout"]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv in self.convs:
            x = conv(x, edge_index, edge_type=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.graph_classification:
            x = self.pool(x, data.batch)

        return x.log_softmax(dim=-1)


class SuperGAT(torch.nn.Module):
    r"""Parametrized SuperGAT model.
    args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        kwargs (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, kwargs):
        super(SuperGAT, self).__init__()

        self.graph_classification = kwargs["graph_classification"]

        if self.graph_classification:
            self.pool = global_mean_pool

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SuperGATConv(
                in_channels,
                kwargs["hidden_units"], 
                heads=int(kwargs["heads_attention"]), 
                concat=False
            )
        )

        for _ in range(number_layers - 2):
            self.convs.append(
                SuperGATConv(
                    kwargs["hidden_units"], 
                    kwargs["hidden_units"], 
                    heads=int(kwargs["heads_attention"]), 
                    concat=False
                )
            )

        self.convs.append(
            SuperGATConv(
                kwargs["hidden_units"],
                out_channels,
                heads=int(kwargs["heads_attention"]), 
                concat=False
            )
        )
        self.dropout = kwargs["dropout"]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.graph_classification:
            x = self.pool(x, data.batch)

        return x.log_softmax(dim=-1)


class GATv2(torch.nn.Module):
    r"""Parametrized GATv2 model.
    args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        kwargs (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, kwargs):
        super(GATv2, self).__init__()

        self.graph_classification = kwargs["graph_classification"]

        if self.graph_classification:
            self.pool = global_mean_pool

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATv2Conv(
                in_channels,
                kwargs["hidden_units"],
                heads=int(kwargs["heads_attention"]),
                concat=False))

        for _ in range(number_layers - 2):
            self.convs.append(
                GATv2Conv(
                    int(kwargs["hidden_units"]), 
                    kwargs["hidden_units"], 
                    heads=int(kwargs["heads_attention"]),
                    concat=False
                )
            )

        self.convs.append(
            GATv2Conv(
                int(kwargs["hidden_units"]), 
                out_channels, 
                heads=int(kwargs["headsAttention"]), 
                concat=False
            )
        )
        self.dropout = kwargs["dropout"]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.graph_classification:
            x = self.pool(x, data.batch)

        return x.log_softmax(dim=-1)


class ARMA(torch.nn.Module):
    r"""Parametrized ARMA model.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        kwargs (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, kwargs):
        super(ARMA, self).__init__()

        self.graph_classification = kwargs["graph_classification"]

        if self.graph_classification:
            self.pool = global_mean_pool

        self.convs = torch.nn.ModuleList()
        self.convs.append(ARMAConv(in_channels, kwargs["hidden_units"]))

        for _ in range(number_layers - 2):
            self.convs.append(
                ARMAConv(
                    kwargs["hidden_units"],
                    kwargs["hidden_units"]
                )
            )

        self.convs.append(ARMAConv(kwargs["hidden_units"], out_channels))
        self.dropout = kwargs["dropout"]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.graph_classification:
            x = self.pool(x, data.batch)