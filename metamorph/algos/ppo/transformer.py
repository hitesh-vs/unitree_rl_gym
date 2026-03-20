"""Transformer encoder layers for processing sequential robot limb observations.

Implements multi-head self-attention with pre-norm residual connections and
optional **fixed attention** – an adjacency-constrained attention mechanism
where each limb token may only attend to its kinematically connected
neighbours.  When enabled, the attention logits for non-adjacent pairs are
masked to ``-∞`` before the softmax, fixing the *structure* of attention
to the robot's kinematic graph while leaving the *values* fully learned.

Default hyperparameters (also reflected in ``config/g1/default.yaml``):

* ``d_model = 128``
* ``num_heads = 2``
* ``num_layers = 5``
* ``dim_feedforward = 1024``
* ``dropout = 0.0``
* ``use_fixed_attention = True``
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional fixed (topology-masked) attention.

    When a boolean ``attn_mask`` of shape ``(seq_len, seq_len)`` is supplied,
    positions where the mask is ``False`` (non-adjacent limb pairs) have their
    attention logits set to ``-1e9`` before softmax, preventing information
    flow across kinematically disconnected joints.  Positions where the mask is
    ``True`` are attended to normally, with weights learned through Q/K/V
    projections.

    This "fixed" structural constraint is the key difference from vanilla
    self-attention: the *pattern* of allowable attention is fixed by the
    robot topology, while the actual attention scores within that pattern
    remain fully differentiable.
    """

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """Compute multi-head self-attention.

        Args:
            x (torch.Tensor): Input of shape (batch, seq_len, d_model).
            attn_mask (torch.Tensor, optional): Boolean tensor of shape
                ``(seq_len, seq_len)`` or ``(batch, num_heads, seq_len,
                seq_len)``.  Entries that are ``False`` are masked to
                ``-1e9`` (blocked attention), entries that are ``True``
                are allowed.  Defaults to ``None`` (full self-attention).

        Returns:
            torch.Tensor: Output of shape (batch, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.shape
        sqrt_d_k = math.sqrt(self.d_k)

        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt_d_k

        if attn_mask is not None:
            # Broadcast to (batch, num_heads, seq_len, seq_len)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            # False → blocked; mask those positions to -inf
            scores = scores.masked_fill(~attn_mask, -1e9)

        attn = self.dropout(F.softmax(scores, dim=-1))

        out = (
            torch.matmul(attn, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with pre-norm residual connections.

    Uses ReLU activation in the feed-forward sublayer and LayerNorm before
    each sublayer (pre-norm architecture).
    """

    def __init__(self, d_model, num_heads, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """Apply one transformer encoder layer.

        Args:
            x (torch.Tensor): Shape (batch, seq_len, d_model).
            attn_mask (torch.Tensor, optional): Boolean attention mask of shape
                ``(seq_len, seq_len)``; see
                :class:`MultiHeadSelfAttention` for details.

        Returns:
            torch.Tensor: Same shape as input.
        """
        x = x + self.dropout(self.self_attn(self.norm1(x), attn_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class TransformerModel(nn.Module):
    """Transformer-based model for processing per-limb observations.

    Processes per-limb observations through:

    1. Optional GCN structural embedding concatenation.
    2. Limb embedding (linear projection to ``d_model``).
    3. Transformer encoder layers (with optional fixed attention).
    4. Output projection from flattened hidden states.

    **Fixed attention** (enabled by ``use_fixed_attention=True``): the
    adjacency mask derived from the robot's kinematic graph is passed to
    every encoder layer.  Each limb token can only attend to its
    kinematically adjacent neighbours.  The mask is supplied at call time
    via the ``adj_mask`` argument to :meth:`forward`.

    This class is used for both the actor and critic networks inside
    :class:`~metamorph.algos.ppo.model.ActorCritic`.

    Default hyperparameters match the specification in
    ``config/g1/default.yaml``:

    * ``d_model = 128``
    * ``num_heads = 2``
    * ``num_layers = 5``
    * ``dim_feedforward = 1024``
    * ``use_fixed_attention = True``
    """

    def __init__(
        self,
        obs_per_limb,
        num_limbs,
        d_model=128,
        num_heads=2,
        num_layers=5,
        dim_feedforward=1024,
        dropout=0.0,
        output_dim=None,
        use_gcn=False,
        gcn_output_dim=23,
        use_fixed_attention=True,
    ):
        """Initialise the Transformer model.

        Args:
            obs_per_limb (int): Observation size per limb.
            num_limbs (int): Number of limbs (graph nodes).
            d_model (int): Transformer model dimension.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            dim_feedforward (int): Feedforward layer dimension.
            dropout (float): Dropout probability.
            output_dim (int, optional): Output dimension. Defaults to d_model.
            use_gcn (bool): Whether to inject GCN structural embeddings.
            gcn_output_dim (int): GCN embedding dimension (used when
                use_gcn=True).
            use_fixed_attention (bool): When ``True``, attention is restricted
                to kinematically adjacent limb pairs using the adjacency mask
                passed to :meth:`forward`.  Defaults to ``True``.
        """
        super().__init__()
        self.num_limbs = num_limbs
        self.use_gcn = use_gcn
        self.obs_per_limb = obs_per_limb
        self.use_fixed_attention = use_fixed_attention

        limb_embed_input = obs_per_limb + (gcn_output_dim if use_gcn else 0)
        self.limb_embed = nn.Linear(limb_embed_input, d_model)

        self.encoder_layers = nn.ModuleList(
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(d_model)

        out_dim = output_dim if output_dim is not None else d_model
        self.output_proj = nn.Linear(d_model * num_limbs, out_dim)

    def _build_adj_attn_mask(self, adj_normalized):
        """Convert a normalised adjacency matrix to a boolean attention mask.

        Entries with a non-zero value in *adj_normalized* are ``True``
        (attention allowed); zero entries are ``False`` (blocked).

        Args:
            adj_normalized (torch.Tensor): Shape (num_limbs, num_limbs).

        Returns:
            torch.Tensor: Boolean mask of shape (num_limbs, num_limbs) on the
                same device as *adj_normalized*.
        """
        return (adj_normalized > 0)

    def forward(self, obs, gcn_embeddings=None, adj_mask=None):
        """Forward pass.

        Args:
            obs (torch.Tensor): Shape (batch, num_limbs * obs_per_limb) or
                (batch, num_limbs, obs_per_limb).
            gcn_embeddings (torch.Tensor, optional): Shape
                (batch, num_limbs, gcn_output_dim).
            adj_mask (torch.Tensor, optional): Boolean or float adjacency
                tensor of shape ``(num_limbs, num_limbs)`` used to build the
                fixed attention mask when ``use_fixed_attention`` is ``True``.
                If ``None`` and fixed attention is enabled, full self-attention
                is used as a fallback.

        Returns:
            torch.Tensor: Shape (batch, output_dim).
        """
        if obs.dim() == 2:
            batch_size = obs.shape[0]
            usable = self.num_limbs * self.obs_per_limb
            if obs.shape[1] != usable:
                import warnings
                warnings.warn(
                    f"[TransformerModel] obs_dim ({obs.shape[1]}) is not a "
                    f"multiple of num_limbs ({self.num_limbs}) × obs_per_limb "
                    f"({self.obs_per_limb}) = {usable}. "
                    f"Truncating {obs.shape[1] - usable} trailing feature(s).",
                    stacklevel=2,
                )
            x = obs[:, :usable].view(batch_size, self.num_limbs, self.obs_per_limb)
        else:
            x = obs
            batch_size = x.shape[0]

        if self.use_gcn and gcn_embeddings is not None:
            x = torch.cat([x, gcn_embeddings], dim=-1)

        x = self.limb_embed(x)

        # Build fixed attention mask from adjacency matrix if requested
        fixed_mask = None
        if self.use_fixed_attention and adj_mask is not None:
            fixed_mask = self._build_adj_attn_mask(adj_mask)

        for layer in self.encoder_layers:
            x = layer(x, fixed_mask)
        x = self.norm(x)

        x = x.contiguous().view(batch_size, -1)
        return self.output_proj(x)
