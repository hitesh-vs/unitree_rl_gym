"""Graph Convolutional Network (GCN) for processing robot kinematic graphs.

Implements a multi-layer GCN that transforms node features using normalised
adjacency matrices to produce structural embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """Single graph convolution layer: H' = sigma(D^{-1/2} A D^{-1/2} H W)."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj_normalized):
        """Apply graph convolution.

        Args:
            x (torch.Tensor): Node features of shape (num_nodes, in_features)
                or (batch_size, num_nodes, in_features).
            adj_normalized (torch.Tensor): Normalised adjacency matrix of
                shape (num_nodes, num_nodes).

        Returns:
            torch.Tensor: Updated node features with the same batch dimensions.
        """
        support = self.linear(x)
        # Handle both batched and unbatched inputs
        return torch.matmul(adj_normalized, support)


class GCN(nn.Module):
    """Multi-layer Graph Convolutional Network.

    Processes a robot kinematic graph to produce structural node embeddings
    that capture connectivity patterns.
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0):
        """Initialise GCN layers.

        Args:
            input_dim (int): Input node feature dimension.
            hidden_dims (list[int]): Hidden layer dimensions.
            output_dim (int): Output embedding dimension.
            dropout (float): Dropout probability applied between hidden layers.
        """
        super().__init__()
        self.dropout = dropout

        layer_dims = [input_dim] + list(hidden_dims) + [output_dim]
        self.layers = nn.ModuleList(
            GraphConvLayer(layer_dims[i], layer_dims[i + 1])
            for i in range(len(layer_dims) - 1)
        )

    def forward(self, node_features, adj_normalized):
        """Forward pass through all GCN layers.

        Args:
            node_features (torch.Tensor): Shape (num_nodes, input_dim) or
                (batch_size, num_nodes, input_dim).
            adj_normalized (torch.Tensor): Shape (num_nodes, num_nodes).

        Returns:
            torch.Tensor: Node embeddings with last dim equal to output_dim,
                preserving input batch dimensions.
        """
        x = node_features
        for i, layer in enumerate(self.layers):
            x = layer(x, adj_normalized)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def get_graph_embedding(self, node_features, adj_normalized, pool='mean'):
        """Produce a single graph-level embedding by pooling node embeddings.

        Args:
            node_features (torch.Tensor): Node feature tensor.
            adj_normalized (torch.Tensor): Normalised adjacency tensor.
            pool (str): Pooling method – one of 'mean', 'max', or 'sum'.

        Returns:
            torch.Tensor: Graph-level embedding tensor.

        Raises:
            ValueError: If *pool* is not a recognised pooling method.
        """
        node_embeddings = self.forward(node_features, adj_normalized)
        if pool == 'mean':
            return node_embeddings.mean(dim=-2)
        if pool == 'max':
            return node_embeddings.max(dim=-2).values
        if pool == 'sum':
            return node_embeddings.sum(dim=-2)
        raise ValueError(f"Unknown pooling method: '{pool}'. Use 'mean', 'max', or 'sum'.")
