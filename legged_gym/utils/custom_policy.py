import torch
import torch.nn as nn
import math
from torch.distributions import Normal
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """Graph Convolution Layer with LayerNorm"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, num_nodes, in_features)
            adj: (batch, num_nodes, num_nodes)
        Returns:
            (batch, num_nodes, out_features)
        """
        out = torch.bmm(adj, x)
        out = self.linear(out)
        out = self.norm(out)
        out = torch.relu(out)
        return out


class GCN(nn.Module):
    """GCN with 4 layers (UNIMALS)"""
    def __init__(self, input_dim, hidden_dim=64, output_dim=23, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.gc_layers = nn.ModuleList([
            GraphConvolution(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x, adj):
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = torch.relu(x)
        
        for gc_layer in self.gc_layers:
            x_res = x
            x = gc_layer(x, adj)
            if x.shape[-1] == x_res.shape[-1]:
                x = x + x_res
        
        x = self.output_proj(x)
        x = self.output_norm(x)
        return x


class TransformerEncoderLayerWithFixedAttention(nn.Module):
    """
    Transformer encoder layer WITH FIXED ATTENTION support
    
    Features from your model.py:
    - Context-aware attention (privileged info)
    - Attention masking based on morphology
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 use_fixed_attention=False, context_embed_size=128):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.use_fixed_attention = use_fixed_attention
        if use_fixed_attention:
            # Context encoder for privileged info
            self.norm_context = nn.LayerNorm(d_model)
            self.context_embed = nn.Linear(context_embed_size, d_model)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None, 
                context=None, return_attention=False):
        """
        Args:
            src: (seq_len, batch, d_model) or (batch, seq_len, d_model)
            context: (seq_len, batch, d_model) - privileged observations
            return_attention: if True, return attention weights
        """
        src2 = self.norm1(src)
        
        if self.use_fixed_attention and context is not None:
            # Fixed attention: use context as keys/values
            context_proj = self.context_embed(context) if context.shape[-1] != src.shape[-1] else context
            context_normed = self.norm_context(context_proj)
            src2, attn_weights = self.self_attn(
                context_normed, context_normed, src2,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            )
        else:
            # Standard self-attention
            src2, attn_weights = self.self_attn(
                src2, src2, src2,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            )
        
        src = src + self.dropout1(src2)
        
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        if return_attention:
            return src, attn_weights
        return src


class TransformerEncoder(nn.Module):
    """Transformer encoder with optional fixed attention"""
    
    def __init__(self, d_model, nhead, num_layers=5, dim_feedforward=1024,
                 dropout=0.1, use_fixed_attention=False, context_embed_size=128):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithFixedAttention(
                d_model, nhead, dim_feedforward, dropout,
                use_fixed_attention=use_fixed_attention,
                context_embed_size=context_embed_size
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None, 
                context=None, return_attention=False):
        """
        Args:
            src: (batch, seq_len, d_model) with batch_first=True
            context: (batch, seq_len, context_embed_size) - privileged info
            return_attention: if True, return attention weights from all layers
        """
        output = src
        attention_maps = [] if return_attention else None
        
        for layer in self.layers:
            if return_attention:
                output, attn = layer(
                    output, src_mask=src_mask, 
                    src_key_padding_mask=src_key_padding_mask,
                    context=context, return_attention=True
                )
                attention_maps.append(attn)
            else:
                output = layer(
                    output, src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    context=context
                )
        
        output = self.norm(output)
        
        if return_attention:
            return output, attention_maps
        return output


class TransformerModel(nn.Module):
    """
    Your TransformerModel architecture from ModuMorph
    
    Features preserved:
    - Per-node embedding (PER_NODE_EMBED)
    - Fixed attention with context
    - Dropout masking (CONSISTENT_DROPOUT)
    - GCN integration
    """
    
    def __init__(self, num_obs, num_actions, num_limbs=6, cfg=None):
        super().__init__()
        
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.num_limbs = num_limbs
        self.cfg = cfg or {}
        
        print(f"[TransformerModel] num_obs={num_obs}, num_actions={num_actions}, num_limbs={num_limbs}")
        
        # Config parameters
        d_model = self.cfg.get('transformer_input_size', 128)
        transformer_heads = self.cfg.get('transformer_num_heads', 2)
        transformer_layers = self.cfg.get('transformer_num_layers', 5)
        transformer_ffn = self.cfg.get('transformer_feedforward_dim', 1024)
        gcn_layers = self.cfg.get('gcn_num_layers', 4)
        gcn_output = self.cfg.get('gcn_output_dim', 23)
        use_fixed_attention = self.cfg.get('use_fixed_attention', True)
        use_per_node_embed = self.cfg.get('use_per_node_embed', True)
        context_embed_size = self.cfg.get('context_embed_size', 128)
        
        self.d_model = d_model
        self.use_fixed_attention = use_fixed_attention
        self.use_per_node_embed = use_per_node_embed
        self.context_embed_size = context_embed_size
        
        # ============= INPUT PROJECTION (FLAT OBS → d_model) =============
        # G1 obs is 47 dims flat, project to d_model (128)
        self.input_proj = nn.Linear(num_obs, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # ============= RESHAPE TO PER-LIMB =============
        # After projection: (batch, 128) → (batch, num_limbs, 128//num_limbs)
        limb_feat_dim = d_model // num_limbs  # 128 // 6 ≈ 21
        
        # ============= PER-NODE EMBEDDING (YOUR FEATURE) =============
        if use_per_node_embed:
            # Per-limb embedding with node-specific weights
            initrange = 0.1
            self.limb_embed_weights = nn.Parameter(
                torch.zeros(num_limbs, limb_feat_dim, d_model).uniform_(-initrange, initrange)
            )
            self.limb_embed_bias = nn.Parameter(torch.zeros(num_limbs, d_model))
        else:
            # Shared embedding
            self.limb_embed = nn.Linear(limb_feat_dim, d_model)
        
        # ============= GCN (OPTIONAL) =============
        self.gcn = GCN(
            input_dim=limb_feat_dim,
            hidden_dim=64,
            output_dim=gcn_output,
            num_layers=gcn_layers
        )
        
        # ============= FIXED ATTENTION CONTEXT =============
        if use_fixed_attention:
            # Context encoder: (batch, num_obs) → (batch, context_embed_size)
            self.context_embed = nn.Linear(num_obs, context_embed_size)
            context_modules = [nn.ReLU()]
            for _ in range(2):
                context_modules.append(nn.Linear(context_embed_size, context_embed_size))
                context_modules.append(nn.ReLU())
            self.context_encoder = nn.Sequential(*context_modules)
        
        # ============= POSITIONAL ENCODING =============
        self.pos_embedding = nn.Parameter(torch.randn(1, num_limbs, d_model))
        
        # ============= TRANSFORMER =============
        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ffn,
            dropout=0.1,
            use_fixed_attention=use_fixed_attention,
            context_embed_size=context_embed_size
        )
        
        # ============= DECODER =============
        self.decoder = nn.Sequential(
            nn.Linear(d_model * num_limbs, 256),
            nn.ELU(),
            nn.Linear(256, num_actions)
        )
        
        # ============= DROPOUT MASK =============
        self.dropout = nn.Dropout(p=0.1)
        self.consistent_dropout = False  # YOUR FEATURE
        
        self.init_weights()
    
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, obs, context=None, return_attention=False, dropout_mask=None):
        """
        Args:
            obs: (batch, num_obs=47) - flat observations from IsaacGym
            context: (batch, num_obs=47) - privileged info (optional)
            return_attention: bool
            dropout_mask: precomputed dropout mask
        
        Returns:
            output: (batch, num_actions)
            attention_maps: (list of attention tensors) if return_attention=True
            dropout_mask: new dropout mask for next step
        """
        batch_size = obs.shape[0]
        
        # ============= PHASE 1: PROJECT FLAT OBS =============
        # (batch, 47) → (batch, 128)
        x = self.input_proj(obs)
        x = self.input_norm(x)
        x = torch.relu(x)
        
        # ============= PHASE 2: RESHAPE TO PER-LIMB =============
        # Use d_model // num_limbs as limb_feat_dim
        # Make d_model divisible: d_model=120 (6*20), so each limb gets 20 dims
        limb_feat_dim = self.d_model // self.num_limbs  # 128 // 6 = 21 (rounded down)
        # Pad to make divisible
        padded_dim = self.num_limbs * limb_feat_dim  # 6 * 21 = 126
        
        if x.shape[1] != padded_dim:
            # Pad x to make it divisible
            if x.shape[1] < padded_dim:
                padding = torch.zeros(batch_size, padded_dim - x.shape[1], device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, :padded_dim]
        
        # Now reshape safely
        x = x.reshape(batch_size, self.num_limbs, limb_feat_dim)  # (batch, 6, 21)
        
        # ============= PHASE 3: PER-NODE EMBEDDING =============
        if self.use_per_node_embed:
            # Apply per-node weights: (batch, 6, 21) @ (6, 21, 128) → (batch, 6, 128)
            obs_embed = (x.unsqueeze(-1) * self.limb_embed_weights.unsqueeze(0)).sum(dim=-2)
            obs_embed = obs_embed + self.limb_embed_bias.unsqueeze(0)
        else:
            obs_embed = self.limb_embed(x)
        
        # (batch, 6, 128)
        
        # Add positional embedding
        obs_embed = obs_embed + self.pos_embedding
        
        # ============= PHASE 4: FIXED ATTENTION CONTEXT =============
        context_to_attn = None
        if self.use_fixed_attention and context is not None:
            # (batch, 47) → (batch, context_embed_size)
            context_embed = self.context_embed(context)
            context_to_attn = self.context_encoder(context_embed)
        
        # ============= PHASE 5: DROPOUT WITH CONSISTENCY =============
        if self.training and self.consistent_dropout:
            if dropout_mask is None:
                obs_embed_dropped = self.dropout(obs_embed)
                dropout_mask = torch.where(obs_embed_dropped == 0., 0., 1.)
                obs_embed = obs_embed_dropped
            else:
                obs_embed = obs_embed * dropout_mask / 0.9
        else:
            obs_embed = self.dropout(obs_embed)
            dropout_mask = None
        
        # ============= PHASE 6: TRANSFORMER WITH FIXED ATTENTION =============
        if return_attention:
            obs_embed_t, attention_maps = self.transformer(
                obs_embed,
                context=context_to_attn,
                return_attention=True
            )
        else:
            obs_embed_t = self.transformer(
                obs_embed,
                context=context_to_attn
            )
            attention_maps = None
        
        # ============= PHASE 7: DECODER =============
        # (batch, 6, 128) → (batch, 768) → (batch, 256) → (batch, 12)
        output = obs_embed_t.reshape(batch_size, -1)
        output = self.decoder(output)
        
        return output, attention_maps, dropout_mask

class ActorCritic(nn.Module):
    """
    Actor-Critic with YOUR TransformerModel
    
    Preserved features:
    - Per-node embedding
    - Fixed attention with context
    - Consistent dropout mask
    """
    
    def __init__(self, num_obs, num_actions, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.is_recurrent = False
        self.num_obs = num_obs
        self.num_actions = num_actions
        
        # Determine number of limbs from action space
        if num_actions == 12:  # G1
            num_limbs = 6
        elif num_actions == 10:  # H1
            num_limbs = 5
        else:
            num_limbs = num_actions // 2
        
        # Convert cfg to dict if it's a config object
        cfg_dict = {}
        if hasattr(cfg, '__dict__'):
            cfg_dict = vars(cfg)
        else:
            cfg_dict = dict(cfg)
        
        # ============= ACTOR & CRITIC =============
        # Actor uses proprioceptive obs (47 dims)
        self.actor = TransformerModel(
            num_obs=47,
            num_actions=num_actions,
            num_limbs=num_limbs,
            cfg=cfg_dict
        )
        
        # Critic uses full obs (proprioceptive + privileged)
        critic_input_size = 50  # G1: 47 + 3 privileged
        
        self.critic = nn.Sequential(
            nn.Linear(critic_input_size, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )
        
        # ============= ACTION DISTRIBUTION =============
        self.log_std = nn.Parameter(torch.zeros(num_actions))
        
        # ============= BUFFERS FOR PPO =============
        self.register_buffer('action_mean', torch.zeros(1, num_actions))
        self.register_buffer('action_std', torch.ones(1, num_actions))
        self.register_buffer('entropy', torch.zeros(1))
        self.register_buffer('dropout_mask_a', None)
        self.register_buffer('dropout_mask_c', None)
        self.register_buffer('mu', torch.zeros(1, num_actions))
    
    # ============= PROPERTIES FOR LOGGING =============
    @property
    def std(self):
        """Return action std for logging"""
        return self.action_std
    
    def reset(self, dones):
        """Reset any recurrent state (not needed for non-recurrent)"""
        pass
    
    def forward(self, obs, context=None, deterministic=False, 
                dropout_mask_a=None, dropout_mask_c=None, **kwargs):
        """
        Args:
            obs: (batch, 47) - proprioceptive obs
            context: (batch, 50) - full obs with privileged info
            **kwargs: ignored (for compatibility with recurrent models)
        """
        # Actor
        action_mean, _, dropout_mask_a = self.actor(
            obs, context=context, dropout_mask=dropout_mask_a
        )
        
        # Compute action std and entropy
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)
        entropy = dist.entropy().mean()
        
        # Store for PPO
        self.action_mean = action_mean
        self.action_std = action_std
        self.entropy = entropy
        self.mu = action_mean
        
        # Critic needs full obs (proprioceptive + privileged)
        if context is not None:
            value = self.critic(context)
        else:
            padded_obs = torch.cat([obs, torch.zeros(obs.shape[0], 3, device=obs.device, dtype=obs.dtype)], dim=1)
            value = self.critic(padded_obs)
        
        return value, dist, dropout_mask_a, dropout_mask_c
    
    def act(self, obs, context=None, deterministic=False,
            dropout_mask_a=None, dropout_mask_c=None, **kwargs):
        """Inference pass - returns action only
        
        Accepts additional kwargs (masks, hidden_states, etc.) for compatibility with recurrent models
        """
        action_mean, _, dropout_mask_a = self.actor(
            obs, context=context, dropout_mask=dropout_mask_a
        )
        
        # Compute action std and entropy
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)
        entropy = dist.entropy().mean()
        
        # Store for PPO
        self.action_mean = action_mean
        self.action_std = action_std
        self.entropy = entropy
        self.mu = action_mean
        
        if deterministic:
            action = action_mean
        else:
            action = dist.rsample()
        
        return action
    
    def get_actions_log_prob(self, actions):
        """Compute log probabilities for given actions"""
        dist = Normal(self.action_mean, self.action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        return log_prob
    
    def evaluate(self, obs, actions=None, context=None, **kwargs):
        """PPO loss computation - obs here is critic_obs (50 dims)
        
        Accepts additional kwargs (masks, hidden_states, etc.) for compatibility with recurrent models
        """
        # Extract proprioceptive part for actor (first 47 dims)
        proprioceptive_obs = obs[:, :47]
        
        action_mean = self.actor(proprioceptive_obs, context=context)[0]
        
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)
        entropy = dist.entropy().mean()
        
        # Store for PPO
        self.action_mean = action_mean
        self.action_std = action_std
        self.entropy = entropy
        self.mu = action_mean
        
        # Critic uses full obs (50 dims)
        value = self.critic(obs)
        
        if actions is not None:
            log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            return log_prob, value, entropy
        else:
            return value