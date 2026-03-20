"""URDF Graph Parser for constructing robot kinematic graphs.

Parses URDF files to extract joint connectivity and builds graph
representations with node features for GCN processing.
"""

import xml.etree.ElementTree as ET
import numpy as np
import torch


def parse_urdf_graph(urdf_path):
    """Parse a URDF file and construct a kinematic graph.

    Args:
        urdf_path (str): Path to the URDF file.

    Returns:
        dict: Graph data containing:
            - 'nodes': list of link names
            - 'edges': list of (parent_idx, child_idx) tuples
            - 'adj_matrix': numpy array of shape (N, N)
            - 'num_nodes': number of nodes
            - 'link_to_idx': dict mapping link name -> node index
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Collect all links
    links = [link.get('name') for link in root.findall('link')]
    link_to_idx = {link: i for i, link in enumerate(links)}

    # Collect joints and build edges
    edges = []
    for joint in root.findall('joint'):
        parent = joint.find('parent')
        child = joint.find('child')
        if parent is not None and child is not None:
            parent_name = parent.get('link')
            child_name = child.get('link')
            if parent_name in link_to_idx and child_name in link_to_idx:
                edges.append((link_to_idx[parent_name], link_to_idx[child_name]))

    num_nodes = len(links)

    # Build undirected adjacency matrix with self-loops
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for parent_idx, child_idx in edges:
        adj_matrix[parent_idx, child_idx] = 1.0
        adj_matrix[child_idx, parent_idx] = 1.0
    np.fill_diagonal(adj_matrix, 1.0)

    return {
        'nodes': links,
        'edges': edges,
        'adj_matrix': adj_matrix,
        'num_nodes': num_nodes,
        'link_to_idx': link_to_idx,
    }


def _bfs_depths(adj, root=0):
    """BFS to compute depth of each node from root.

    Args:
        adj (numpy.ndarray): Adjacency matrix (without self-loops).
        root (int): Root node index.

    Returns:
        list: Depth of each node; unreachable nodes get depth 0.
    """
    num_nodes = adj.shape[0]
    depths = [-1] * num_nodes
    depths[root] = 0
    queue = [root]
    while queue:
        node = queue.pop(0)
        for neighbor in range(num_nodes):
            if adj[node, neighbor] > 0 and depths[neighbor] == -1:
                depths[neighbor] = depths[node] + 1
                queue.append(neighbor)
    # Unreachable nodes get depth 0
    for i in range(num_nodes):
        if depths[i] == -1:
            depths[i] = 0
    return depths


def compute_node_features(graph_data, mode='onehot'):
    """Compute node features for the graph.

    Args:
        graph_data (dict): Graph data from parse_urdf_graph().
        mode (str): Feature computation mode:
            - 'none': Zero features (shape: (N, 1))
            - 'onehot': One-hot encoding of node index (shape: (N, N))
            - 'topological': Degree + normalised degree + BFS depth (shape: (N, 3))

    Returns:
        numpy.ndarray: Node feature matrix of shape (num_nodes, feature_dim).

    Raises:
        ValueError: If mode is not one of the supported values.
    """
    num_nodes = graph_data['num_nodes']
    adj = graph_data['adj_matrix']

    # Remove self-loops for degree computation
    adj_no_self = adj - np.eye(num_nodes, dtype=np.float32)

    if mode == 'none':
        return np.zeros((num_nodes, 1), dtype=np.float32)

    if mode == 'onehot':
        return np.eye(num_nodes, dtype=np.float32)

    if mode == 'topological':
        degrees = adj_no_self.sum(axis=1, keepdims=True)
        max_degree = max(float(degrees.max()), 1.0)
        norm_degrees = degrees / max_degree

        depths = _bfs_depths(adj_no_self, root=0)
        depth_arr = np.array(depths, dtype=np.float32).reshape(-1, 1)
        max_depth = max(float(depth_arr.max()), 1.0)
        norm_depths = depth_arr / max_depth

        return np.concatenate([degrees, norm_degrees, norm_depths], axis=1)

    raise ValueError(
        f"Unknown node feature mode: '{mode}'. Use 'none', 'onehot', or 'topological'."
    )


def normalize_adjacency(adj_matrix):
    """Compute symmetric normalised adjacency: D^{-1/2} A D^{-1/2}.

    Args:
        adj_matrix (numpy.ndarray): Adjacency matrix of shape (N, N).

    Returns:
        numpy.ndarray: Normalised adjacency matrix of the same shape.
    """
    degree = adj_matrix.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ adj_matrix @ D_inv_sqrt


def build_graph_tensors(urdf_path, mode='onehot', device='cpu'):
    """Build all graph tensors from a URDF file.

    Parses the URDF, computes node features and the normalised adjacency
    matrix, and returns everything as PyTorch tensors on the requested device.

    Args:
        urdf_path (str): Path to the URDF file.
        mode (str): Node feature mode ('none', 'onehot', or 'topological').
        device (str or torch.device): Target device for returned tensors.

    Returns:
        dict: Graph tensors:
            - 'node_features': Tensor of shape (num_nodes, feature_dim)
            - 'adj_normalized': Normalised adjacency tensor (num_nodes, num_nodes)
            - 'num_nodes': int, number of nodes
            - 'feature_dim': int, node feature dimension
    """
    graph_data = parse_urdf_graph(urdf_path)
    node_features = compute_node_features(graph_data, mode=mode)
    adj_normalized = normalize_adjacency(graph_data['adj_matrix'])

    return {
        'node_features': torch.tensor(node_features, dtype=torch.float32, device=device),
        'adj_normalized': torch.tensor(adj_normalized, dtype=torch.float32, device=device),
        'num_nodes': graph_data['num_nodes'],
        'feature_dim': node_features.shape[1],
    }
