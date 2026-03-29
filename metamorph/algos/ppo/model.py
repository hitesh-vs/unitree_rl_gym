"""ActorCritic model combining TransformerModel with an optional GCN.

Supports three graph encoding modes:
- ``'none'``: Standard transformer without graph information.
- ``'onehot'``: One-hot node features processed by a GCN.
- ``'topological'``: Topological node features processed by a GCN.
"""

import torch
import torch.nn as nn

from .transformer import TransformerModel
from .gcn import GCN


class ActorCritic(nn.Module):
    """Actor-Critic model with Transformer + optional GCN policy.

    Both the actor (policy) and the critic (value function) share the same
    :class:`~metamorph.algos.ppo.transformer.TransformerModel` architecture
    but have separate weights.

    Graph structural embeddings (when enabled) are computed once per update
    step – the GCN is shared between actor and critic.
    """

    def __init__(self, cfg):
        """Initialise the ActorCritic model.

        Args:
            cfg (ConfigNode): Configuration object.  Required sub-nodes:

                ``cfg.model``:
                    - ``obs_dim`` (int): Total observation dimension.
                    - ``act_dim`` (int): Action dimension.
                    - ``num_limbs`` (int): Number of limbs / graph nodes.
                    - ``d_model`` (int): Transformer hidden dimension.
                    - ``num_heads`` (int): Number of attention heads.
                    - ``num_layers`` (int): Number of transformer layers.
                    - ``dim_feedforward`` (int): Feedforward dimension.
                    - ``dropout`` (float): Dropout probability.
                    - ``graph_mode`` (str): ``'none'``, ``'onehot'``, or
                      ``'topological'``.
                    - ``gcn_hidden_dims`` (list[int]): GCN hidden dimensions.
                    - ``gcn_output_dim`` (int): GCN output dimension.
                    - ``node_feature_dim`` (int): Input node feature dim.
        """
        super().__init__()
        self.cfg = cfg

        m = cfg.model
        obs_dim = m.obs_dim
        act_dim = m.act_dim
        num_limbs = m.num_limbs
        obs_per_limb = obs_dim // num_limbs
        d_model = m.d_model
        num_heads = m.num_heads
        num_layers = m.num_layers
        dim_feedforward = m.dim_feedforward
        dropout = m.dropout
        graph_mode = getattr(m, 'graph_mode', 'none')
        gcn_hidden = getattr(m, 'gcn_hidden_dims', [64])
        gcn_output_dim = getattr(m, 'gcn_output_dim', 32)
        node_feature_dim = getattr(m, 'node_feature_dim', num_limbs)

        use_gcn = graph_mode in ('onehot', 'topological')
        use_fixed_attention = getattr(m, 'use_fixed_attention', True)
        gcn_normalization = getattr(m, 'gcn_normalization', 'layernorm')

        # Shared GCN for structural embeddings (used by both actor and critic)
        if use_gcn:
            self.gcn = GCN(
                input_dim=node_feature_dim,
                hidden_dims=gcn_hidden,
                output_dim=gcn_output_dim,
                normalization=gcn_normalization,
            )
        else:
            self.gcn = None

        shared_kwargs = dict(
            obs_per_limb=obs_per_limb,
            num_limbs=num_limbs,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_gcn=use_gcn,
            gcn_output_dim=gcn_output_dim,
            use_fixed_attention=use_fixed_attention,
        )

        self.actor = TransformerModel(**shared_kwargs, output_dim=act_dim)
        self.critic = TransformerModel(**shared_kwargs, output_dim=1)

        # Learnable action log standard deviation
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def _get_gcn_embeddings(self, obs_dict, batch_size):
        """Compute batched GCN embeddings if graph data is available.

        Args:
            obs_dict (dict): Observation dict that may contain
                ``'graph_node_features'`` and ``'graph_adj_normalized'``.
            batch_size (int): Number of environments.

        Returns:
            tuple[torch.Tensor or None, torch.Tensor or None]:
                ``(gcn_embeddings, adj_normalized)`` where
                ``gcn_embeddings`` has shape
                ``(batch_size, num_limbs, gcn_output_dim)`` or ``None``
                when the GCN is disabled; and ``adj_normalized`` is the
                ``(num_limbs, num_limbs)`` adjacency tensor used for
                fixed attention (``None`` when not available).
        """
        if self.gcn is None:
            return None, None
        node_features = obs_dict['graph_node_features']
        adj = obs_dict['graph_adj_normalized']
        # (num_nodes, gcn_output_dim) -> (batch, num_nodes, gcn_output_dim)
        node_emb = self.gcn(node_features, adj)
        return node_emb.unsqueeze(0).expand(batch_size, -1, -1), adj

    def forward(self, obs_dict):
        """Compute action distribution parameters and value estimate.

        Args:
            obs_dict (dict): Observation dictionary with:
                - ``'proprioceptive'``: (batch, obs_dim) tensor.
                - ``'graph_node_features'`` (optional): (num_nodes, feat_dim).
                - ``'graph_adj_normalized'`` (optional): (num_nodes, num_nodes).

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                ``(action_mean, action_std, value)`` each with batch dimension.
        """
        obs = obs_dict['proprioceptive']
        gcn_embeddings, adj = self._get_gcn_embeddings(obs_dict, obs.shape[0])

        action_mean = self.actor(obs, gcn_embeddings, adj_mask=adj)
        value = self.critic(obs, gcn_embeddings, adj_mask=adj)
        action_std = self.log_std.exp().expand_as(action_mean)

        return action_mean, action_std, value

    def act(self, obs_dict):
        """Sample an action from the policy.

        Args:
            obs_dict (dict): Observation dictionary (see :meth:`forward`).

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                ``(action, log_prob, value)`` where action and log_prob have
                shape (batch,) and value has shape (batch,).
        """
        action_mean, action_std, value = self.forward(obs_dict)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value.squeeze(-1)

    def evaluate(self, obs_dict, actions):
        """Evaluate log-probability and entropy for given actions.

        Args:
            obs_dict (dict): Observation dictionary (see :meth:`forward`).
            actions (torch.Tensor): Actions of shape (batch, act_dim).

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                ``(log_prob, entropy, value)`` each of shape (batch,).
        """
        action_mean, action_std, value = self.forward(obs_dict)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value.squeeze(-1)

    def get_value(self, obs_dict):
        """Return value estimate only.

        Args:
            obs_dict (dict): Observation dictionary (see :meth:`forward`).

        Returns:
            torch.Tensor: Value estimates of shape (batch,).
        """
        _, _, value = self.forward(obs_dict)
        return value.squeeze(-1)
