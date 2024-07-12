import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
import torch_sparse


class MyGINConv(MessagePassing):
    def __init__(self, dim_emb, norm):
        """
        GINConv with built-in MLP (linear -> batch norm -> relu -> linear)
        Args
            dim_emb: embedding dimension
            norm: normlisation layer, either 'bn' (batch norm) or 'ln' (layer norm)
        """

        super().__init__(aggr="add")

        if norm == "bn":
            norm_layer = torch.nn.BatchNorm1d(2 * dim_emb)
        elif norm == "ln":
            norm_layer = torch.nn.LayerNorm(2 * dim_emb)
        else:
            raise NotImplementedError

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_emb, 2 * dim_emb),
            norm_layer,
            torch.nn.ReLU(),
            torch.nn.Linear(2 * dim_emb, dim_emb),
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index):
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x))
        return out

    def message(self, x_j):
        return torch.nn.functional.relu(x_j)

    def update(self, aggr_out):
        return aggr_out

    def message_and_aggregate(self, adj_t, x):
        """
        See: https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html
        """
        return torch_sparse.matmul(adj_t, torch.nn.functional.relu(x), reduce=self.aggr)



class GINModel(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        drop_prob: float = 0.5,
        interleave_diff_graph: bool = False,
        only_original_graph: bool = False,
        only_diff_graph: bool = False,
        global_pool_aggr: str = "global_add_pool",
        norm: str = "bn",
    ):
        """
        Args
            in_channels: dimension of input features
            hidden_channels: dimension of hidden layers
            num_layers: number of layers
            out_channels: dimension of output
            drop_prob: dropout probability

            exactly one of the following three options must be set to True:
            interleave_diff_graph: every even layer conv layer message passes on the diffusion graph instead
            only_original_graph:
            only_diff_graph:

            global_pool_aggr: pooling function. None makes this a node-level model
            norm: normalisation in GINConv layers (bn for batch norm, ln for layer norm)
        """

        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.drop_prob = drop_prob

        assert (
            sum([only_original_graph, only_diff_graph, interleave_diff_graph]) == 1
        ), "Only one of only_original_graph, only_diff_graph, interleave_diff_graph can be used"

        self.interleave_diff_graph = interleave_diff_graph
        self.only_original_graph = only_original_graph
        self.only_diff_graph = only_diff_graph

        self.lin_in = nn.Linear(in_channels, hidden_channels)
        if global_pool_aggr == "global_add_pool":
            self.pool = global_add_pool
        elif global_pool_aggr == "global_mean_pool":
            self.pool = global_mean_pool
        elif global_pool_aggr is None:
            self.pool = None
        else:
            raise NotImplementedError

        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(MyGINConv(hidden_channels, norm))

        self.drop = nn.Dropout(p=drop_prob)

        # self.lin_out = nn.Linear(hidden_channels, out_channels)
        self.lin_out = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 2 * hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_channels, out_channels),
        )

    def forward(self, data):
        x = data.x
        x = self.lin_in(x)

        if hasattr(data, 'adj_t'): 
            edge_index = data.adj_t # sparse representation
        else: 
            edge_index = data.edge_index

        for i, conv in enumerate(self.convs):
            if self.only_original_graph:
                x = x + conv(x, edge_index)
            elif self.only_diff_graph:
                x = x + conv(x, data.rewire_edge_index)
            elif self.interleave_diff_graph:
                if i % 2 == 0:
                    x = x + conv(x, edge_index)
                else:
                    x = x + conv(x, data.rewire_edge_index)
            else:
                raise Exception("No message passing scheme chosen")

            x = self.drop(x)

        if self.pool is not None:
            x = self.pool(x, data.batch)  # (num_nodes, d) -> (batch_size, d)
        x = self.lin_out(x)

        if self.pool is not None:
            x = x.view(-1)

        return x
