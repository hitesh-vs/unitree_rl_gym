"""Graph Observation Adapter.

Converts flat IsaacGym observation tensors (shape: ``(num_envs, obs_dim)``)
to the dictionary format expected by the Transformer+GCN
:class:`~metamorph.algos.ppo.model.ActorCritic` policy.

Graph data is static (precomputed once at environment initialisation) and is
broadcast across all parallel environments with zero copying overhead.
"""

import torch


class GraphObsAdapter:
    """Adapter between IsaacGym flat obs tensors and the policy dict format.

    IsaacGym returns observations as flat tensors of shape
    ``(num_envs, obs_dim)``.  The ActorCritic policy expects a ``dict``
    with at least a ``'proprioceptive'`` key and, optionally, graph data
    tensors.

    Graph data is set once via :meth:`set_graph_data` (or at construction
    time) and is automatically appended to every observation dict returned
    by :meth:`convert`.  Because graph topology is fixed for a given robot,
    the same tensors are reused for every environment without duplication.
    """

    def __init__(self, graph_tensors=None, device='cuda'):
        """Initialise the adapter.

        Args:
            graph_tensors (dict, optional): Precomputed graph data from
                :func:`graphs.parser.build_graph_tensors`.  Expected keys:
                ``'node_features'``, ``'adj_normalized'``, ``'num_nodes'``,
                ``'feature_dim'``.  If ``None``, only proprioceptive data is
                included in converted dicts.
            device (str or torch.device): Target device for tensors.
        """
        self.device = device
        self.graph_tensors = None

        if graph_tensors is not None:
            self.set_graph_data(graph_tensors)

    def set_graph_data(self, graph_tensors):
        """Store static graph data for injection into observation dicts.

        This method validates the provided tensors, moves them to the correct
        device, and caches them for reuse on every :meth:`convert` call.

        Args:
            graph_tensors (dict): Graph tensors from
                :func:`graphs.parser.build_graph_tensors`.

        Raises:
            ValueError: If required keys are missing from *graph_tensors*.
        """
        required_keys = {'node_features', 'adj_normalized', 'num_nodes', 'feature_dim'}
        missing = required_keys - set(graph_tensors.keys())
        if missing:
            raise ValueError(f"graph_tensors missing required keys: {missing}")

        self.graph_tensors = {
            'graph_node_features': graph_tensors['node_features'].to(self.device),
            'graph_adj_normalized': graph_tensors['adj_normalized'].to(self.device),
            'graph_num_nodes': graph_tensors['num_nodes'],
            'graph_feature_dim': graph_tensors['feature_dim'],
        }

    def convert(self, obs):
        """Convert a flat observation tensor to a policy-compatible dict.

        Args:
            obs (torch.Tensor): Flat observation tensor of shape
                ``(num_envs, obs_dim)``.

        Returns:
            dict: Observation dictionary with keys:

                - ``'proprioceptive'``: ``(num_envs, obs_dim)`` tensor on
                  :attr:`device`.
                - ``'graph_node_features'`` *(optional)*: ``(num_nodes,
                  feature_dim)`` tensor when graph data has been set.
                - ``'graph_adj_normalized'`` *(optional)*: ``(num_nodes,
                  num_nodes)`` tensor when graph data has been set.
                - ``'graph_num_nodes'``, ``'graph_feature_dim'`` scalars.

        Raises:
            TypeError: If *obs* is not a :class:`torch.Tensor`.
        """
        if not isinstance(obs, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(obs).__name__}")

        obs_dict = {'proprioceptive': obs.to(self.device)}
        if self.graph_tensors is not None:
            obs_dict.update(self.graph_tensors)
        return obs_dict

    def __call__(self, obs):
        """Alias for :meth:`convert`."""
        return self.convert(obs)
