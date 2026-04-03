"""Transformer encoder layers for processing sequential robot limb observations.

Implements multi-head self-attention with pre-norm residual connections and
**fixed attention** driven by a structural context tensor (GCN embeddings).

Fixed attention pattern (matching the reference ``TransformerEncoderLayerResidual``
design):

* When a ``context`` tensor is supplied, each encoder layer performs
  **cross-attention**: Q and K are derived from ``norm_context(context)``
  (the GCN structural embedding), while V comes from ``norm1(src)`` (the
  actual limb observation sequence).  This "fixes" the attention routing to
  the robot's kinematic structure while leaving the attended values fully
  learned.
* When ``context`` is ``None`` the layer degrades to standard self-attention:
  Q = K = V = ``norm1(src)``.

An additive adjacency mask (0.0 for allowed, ``-inf`` for blocked) can also be
applied on top of either mode to restrict attention to kinematically adjacent
limb pairs.

Default hyperparameters (also reflected in ``config/g1/default.yaml``):

* ``d_model = 128``
* ``num_heads = 2``
* ``num_layers = 5``
* ``dim_feedforward = 1024``
* ``dropout = 0.0``
* ``use_fixed_attention = True``
"""

import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import ModuleList


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise ValueError(f"Unknown activation: '{activation}'")


class TransformerEncoderLayerResidual(nn.Module):
    """Single encoder layer with pre-norm residuals and optional fixed attention.

    Mirrors the ``TransformerEncoderLayerResidual`` reference implementation.

    When ``fix_attention=True`` a ``norm_context`` LayerNorm is added.  At
    forward time, if a ``context`` tensor is provided the layer performs
    cross-attention (Q=K from context, V from src).  If ``context`` is
    ``None`` the layer falls back to standard self-attention.

    Args:
        d_model (int): Model dimension.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Feed-forward hidden dimension (default: 1024).
        dropout (float): Dropout probability (default: 0.0).
        activation (str): Activation function – ``'relu'`` or ``'gelu'``
            (default: ``'relu'``).
        batch_first (bool): If ``True``, input/output shape is
            ``(batch, seq, d_model)``; otherwise ``(seq, batch, d_model)``
            (default: ``True``).
        fix_attention (bool): Whether to allocate ``norm_context`` and support
            context-based fixed attention (default: ``False``).
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        batch_first=True,
        fix_attention=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.fix_attention = fix_attention
        # norm_context is only created when fixed attention is enabled
        if fix_attention:
            self.norm_context = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        context=None,
        return_attention=False,
    ):
        """Apply one encoder layer.

        Args:
            src (torch.Tensor): Shape ``(batch, seq_len, d_model)`` when
                ``batch_first=True``.
            src_mask (torch.Tensor, optional): Additive float attention mask of
                shape ``(seq_len, seq_len)``.  ``0.0`` = attend,
                ``-inf`` = block.
            src_key_padding_mask (torch.Tensor, optional): Per-batch key
                padding mask of shape ``(batch, seq_len)``.
            context (torch.Tensor, optional): Fixed structural context tensor
                of shape ``(batch, seq_len, d_model)``.  When provided (and
                ``fix_attention=True``), Q and K are both derived from
                ``norm_context(context)`` (same tensor for both Q and K),
                while V comes from ``norm1(src)``.  This fixes the attention
                routing to the structural context (e.g. GCN embeddings).
                When ``None`` the layer falls back to standard self-attention.
            return_attention (bool): When ``True``, also returns the attention
                weight tensor.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                Updated ``src`` of the same shape, optionally with attention
                weights ``(batch, seq_len, seq_len)``.
        """
        src2 = self.norm1(src)

        if self.fix_attention and context is not None:
            # Fixed attention: Q=K=norm_context(context), V=norm1(src)
            context_normed = self.norm_context(context)
            src2, attn_weights = self.self_attn(
                context_normed,
                context_normed,
                src2,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=return_attention,
            )
        else:
            # Standard self-attention: Q=K=V=norm1(src)
            src2, attn_weights = self.self_attn(
                src2,
                src2,
                src2,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=return_attention,
            )

        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        if return_attention:
            return src, attn_weights
        return src


class TransformerEncoder(nn.Module):
    """Stack of :class:`TransformerEncoderLayerResidual` layers.

    Mirrors the ``TransformerEncoder`` reference implementation, including the
    ``get_attention_maps`` diagnostic method.

    Args:
        encoder_layer (TransformerEncoderLayerResidual): Prototype layer; deep
            copies are made for each stack entry.
        num_layers (int): Number of encoder layers.
        norm (nn.Module, optional): Final normalisation layer.
    """

    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, context=None):
        """Pass src through all encoder layers.

        Args:
            src (torch.Tensor): Input sequence.
            mask (torch.Tensor, optional): Additive attention mask.
            src_key_padding_mask (torch.Tensor, optional): Padding mask.
            context (torch.Tensor, optional): Fixed structural context passed
                to every layer (see :class:`TransformerEncoderLayerResidual`).

        Returns:
            torch.Tensor: Encoded sequence of the same shape as ``src``.
        """
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                context=context,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output

    def get_attention_maps(self, src, mask=None, src_key_padding_mask=None, context=None):
        """Run forward pass and collect per-layer attention weights.

        Args:
            src (torch.Tensor): Input sequence.
            mask (torch.Tensor, optional): Additive attention mask.
            src_key_padding_mask (torch.Tensor, optional): Padding mask.
            context (torch.Tensor, optional): Fixed structural context.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]:
                ``(output, attention_maps)`` where ``attention_maps[i]`` is
                the attention weight tensor from layer ``i``.
        """
        attention_maps = []
        output = src
        for layer in self.layers:
            output, attn_map = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                context=context,
                return_attention=True,
            )
            attention_maps.append(attn_map)
        if self.norm is not None:
            output = self.norm(output)
        return output, attention_maps


class TransformerModel(nn.Module):
    """Transformer-based model for processing per-limb observations.

    Processes per-limb observations through:

    1. Limb embedding: linear projection of ``obs_per_limb`` → ``d_model``.
    2. (If ``use_fixed_attention`` and GCN enabled) project GCN embeddings
       ``gcn_output_dim`` → ``d_model`` to form the fixed attention context.
    3. :class:`TransformerEncoder` with fixed attention context and optional
       adjacency mask restricting attention to adjacent limb pairs.
    4. Output projection from flattened hidden states.

    **Fixed attention** (``use_fixed_attention=True``):
    GCN embeddings are projected to ``d_model`` and passed as ``context`` to
    every encoder layer.  Inside each layer Q and K come from the GCN context
    (normalised via ``norm_context``), while V comes from the observation
    sequence.  This roots the *structure* of attention in the robot's kinematic
    graph.  An additive adjacency mask additionally zeros out non-adjacent
    attention weights.

    When ``use_fixed_attention=False`` (or no GCN) the model reduces to a
    standard pre-norm transformer with optional GCN concatenation.

    This class is used for both the actor and critic networks inside
    :class:`~metamorph.algos.ppo.model.ActorCritic`.
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
            output_dim (int, optional): Output dimension (default: ``d_model``).
            use_gcn (bool): Whether GCN embeddings are available.
            gcn_output_dim (int): GCN output dimension.
            use_fixed_attention (bool): When ``True``, GCN embeddings are used
                as the fixed attention context (Q/K source) inside each encoder
                layer.  When ``False``, GCN embeddings are concatenated to the
                per-limb observations before the limb embedding.
        """
        super().__init__()
        self.num_limbs = num_limbs
        self.use_gcn = use_gcn
        self.obs_per_limb = obs_per_limb
        self.use_fixed_attention = use_fixed_attention

        # When fixed attention is used, GCN embeddings are the context (not
        # concatenated to obs), so limb_embed only takes obs_per_limb.
        if use_gcn and not use_fixed_attention:
            limb_embed_input = obs_per_limb + gcn_output_dim
        else:
            limb_embed_input = obs_per_limb
        self.limb_embed = nn.Linear(limb_embed_input, d_model)

        # Project GCN output → d_model to serve as fixed attention context
        self._has_gcn_context_proj = use_gcn and use_fixed_attention
        if self._has_gcn_context_proj:
            self.gcn_context_proj = nn.Linear(gcn_output_dim, d_model)

        encoder_layer = TransformerEncoderLayerResidual(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            fix_attention=use_fixed_attention,
        )
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        out_dim = output_dim if output_dim is not None else d_model
        self.output_proj = nn.Linear(d_model * num_limbs, out_dim)

    def _build_adj_attn_mask(self, adj_normalized):
        """Build an additive float attention mask from a normalised adjacency.

        Returns a ``(num_limbs, num_limbs)`` float tensor with ``0.0`` where
        attention is allowed (adjacent or self) and ``-inf`` where it is
        blocked (non-adjacent).  This format is directly consumed by
        ``nn.MultiheadAttention`` as an additive ``attn_mask``.

        Self-loops are always kept (diagonal forced to 0.0).
        """
        mask = torch.full(
            adj_normalized.shape,
            float("-inf"),
            dtype=adj_normalized.dtype,
            device=adj_normalized.device,
        )
        # Allow adjacent nodes and self-connections
        mask[adj_normalized > 0] = 0.0
        n = adj_normalized.size(0)
        mask[torch.arange(n), torch.arange(n)] = 0.0
        return mask

    def forward(self, obs, gcn_embeddings=None, adj_mask=None):
        """Forward pass.

        Args:
            obs (torch.Tensor): Shape ``(batch, num_limbs * obs_per_limb)``
                or ``(batch, num_limbs, obs_per_limb)``.
            gcn_embeddings (torch.Tensor, optional): Shape
                ``(batch, num_limbs, gcn_output_dim)``.
            adj_mask (torch.Tensor, optional): Normalised adjacency matrix of
                shape ``(num_limbs, num_limbs)``.  Used to build the additive
                attention mask restricting attention to adjacent limb pairs.

        Returns:
            torch.Tensor: Shape ``(batch, output_dim)``.
        """
        if obs.dim() == 2:
            batch_size = obs.shape[0]
            usable = self.num_limbs * self.obs_per_limb
            if obs.shape[1] != usable:
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

        # Concatenate GCN embeddings only in non-fixed-attention mode
        if self.use_gcn and not self.use_fixed_attention and gcn_embeddings is not None:
            x = torch.cat([x, gcn_embeddings], dim=-1)

        x = self.limb_embed(x)

        # Build fixed attention context from GCN embeddings
        context = None
        if self.use_fixed_attention and self._has_gcn_context_proj and gcn_embeddings is not None:
            context = self.gcn_context_proj(gcn_embeddings)

        # Build additive adjacency attention mask
        fixed_mask = None
        if adj_mask is not None:
            fixed_mask = self._build_adj_attn_mask(adj_mask)

        x = self.encoder(x, mask=fixed_mask, context=context)

        x = x.contiguous().view(batch_size, -1)
        return self.output_proj(x)
