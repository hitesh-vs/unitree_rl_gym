import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from modular_policy.config import cfg
from modular_policy.utils import model as tu
from modular_policy.algos.ppo.transformer import (
    TransformerEncoder, TransformerEncoderLayerResidual)
from modular_policy.algos.ppo.gcn import build_gcn_from_cfg
# No gym import — obs_space is _DictSpace from obs_builder

class FiLMGenerator(nn.Module):
    """
    Per-limb FiLM: obs_context (L, B, ctx_dim) -> gamma, beta each (L, B, d_model).

    Architecture mirrors FIX_ATTENTION's context_embed_attention path exactly:
        Linear(ctx_dim, CONTEXT_EMBED_SIZE)
        -> ReLU -> [Linear(H,H) -> ReLU] x LINEAR_CONTEXT_LAYER
        -> Linear(CONTEXT_EMBED_SIZE, 2 * d_model)

    Identity init: gamma bias=1, beta bias=0.
    Network starts as a no-op; training sculpts from a stable baseline.
    """
    def __init__(self, ctx_dim_per_limb: int, d_model: int):
        super().__init__()
        H = cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE

        self.embed = nn.Linear(ctx_dim_per_limb, H)

        layers = [nn.ReLU()]
        for _ in range(cfg.MODEL.TRANSFORMER.LINEAR_CONTEXT_LAYER):
            layers += [nn.Linear(H, H), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

        self.out = nn.Linear(H, 2 * d_model)
        self._init_weights(d_model)

    def _init_weights(self, d_model):
        r = cfg.MODEL.TRANSFORMER.EMBED_INIT
        self.embed.weight.data.uniform_(-r, r)
        self.embed.bias.data.zero_()
        # Zero weights on out-projection so the bias fully controls init
        nn.init.zeros_(self.out.weight)
        self.out.bias.data[:d_model] = 1.0   # gamma -> 1
        self.out.bias.data[d_model:] = 0.0   # beta  -> 0

    def forward(self, obs_context):
        # obs_context: (L, B, ctx_dim_per_limb)
        h     = self.embed(obs_context)    # (L, B, H)
        h     = self.encoder(h)            # (L, B, H)
        out   = self.out(h)                # (L, B, 2*d_model)
        d     = out.shape[-1] // 2
        return out[..., :d], out[..., d:]  # gamma, beta

class TransformerModel(nn.Module):
    def __init__(self, obs_space, decoder_out_dim):
        super().__init__()
        self.decoder_out_dim = decoder_out_dim
        self.model_args      = cfg.MODEL.TRANSFORMER
        self.seq_len         = cfg.MODEL.MAX_LIMBS

        # obs_space["proprioceptive"].shape[0] gives prop_dim = max_limbs * limb_obs_size
        limb_obs_size = obs_space["proprioceptive"].shape[0] // self.seq_len
        self.d_model  = cfg.MODEL.LIMB_EMBED_SIZE

        if self.model_args.PER_NODE_EMBED:
            initrange = self.model_args.EMBED_INIT
            self.limb_embed_weights = nn.Parameter(
                torch.zeros(self.seq_len, len(cfg.ENV.WALKERS),
                            limb_obs_size, self.d_model).uniform_(-initrange, initrange))
            self.limb_embed_bias = nn.Parameter(
                torch.zeros(self.seq_len, len(cfg.ENV.WALKERS), self.d_model))
        else:
            self.limb_embed = nn.Linear(limb_obs_size, self.d_model)

        self.ext_feat_fusion = self.model_args.EXT_MIX

        if self.model_args.POS_EMBEDDING == "learnt":
            self.pos_embedding = PositionalEncoding(self.d_model, self.seq_len)
        elif self.model_args.POS_EMBEDDING == "abs":
            self.pos_embedding = PositionalEncoding1D(self.d_model, self.seq_len)

        encoder_layers = TransformerEncoderLayerResidual(
            cfg.MODEL.LIMB_EMBED_SIZE,
            self.model_args.NHEAD,
            self.model_args.DIM_FEEDFORWARD,
            self.model_args.DROPOUT,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, self.model_args.NLAYERS, norm=None)

        decoder_input_dim      = self.d_model
        self.decoder_input_dim = decoder_input_dim

        if self.model_args.PER_NODE_DECODER:
            initrange = self.model_args.DECODER_INIT
            w = torch.zeros(decoder_input_dim, decoder_out_dim).uniform_(
                -initrange, initrange).repeat(self.seq_len, len(cfg.ENV.WALKERS), 1, 1)
            b = torch.zeros(decoder_out_dim).uniform_(
                -initrange, initrange).repeat(self.seq_len, len(cfg.ENV.WALKERS), 1)
            self.decoder_weights = nn.Parameter(w)
            self.decoder_bias    = nn.Parameter(b)
        else:
            self.decoder = tu.make_mlp_default(
                [decoder_input_dim] + self.model_args.DECODER_DIMS + [decoder_out_dim],
                final_nonlinearity=False)

        if self.model_args.FIX_ATTENTION:
            context_obs_size = obs_space["context"].shape[0] // self.seq_len
            self.context_embed_attention = nn.Linear(
                context_obs_size, self.model_args.CONTEXT_EMBED_SIZE)

            if self.model_args.CONTEXT_ENCODER == "transformer":
                ctx_layers = TransformerEncoderLayerResidual(
                    self.model_args.CONTEXT_EMBED_SIZE,
                    self.model_args.NHEAD,
                    self.model_args.DIM_FEEDFORWARD,
                    self.model_args.DROPOUT,
                )
                self.context_encoder_attention = TransformerEncoder(
                    ctx_layers, self.model_args.CONTEXT_LAYER, norm=None)
            else:
                modules = [nn.ReLU()]
                for _ in range(self.model_args.LINEAR_CONTEXT_LAYER):
                    modules += [nn.Linear(self.model_args.CONTEXT_EMBED_SIZE,
                                          self.model_args.CONTEXT_EMBED_SIZE),
                                nn.ReLU()]
                self.context_encoder_attention = nn.Sequential(*modules)
        
        if self.model_args.USE_FILM:
            context_obs_size = obs_space["context"].shape[0] // self.seq_len
            self.film_generator = FiLMGenerator(
                ctx_dim_per_limb=context_obs_size,
                d_model=self.d_model,
            )
        else:
            self.film_generator = None

        if self.model_args.HYPERNET:
            context_obs_size = obs_space["context"].shape[0] // self.seq_len
            self.context_embed_HN = nn.Linear(
                context_obs_size, self.model_args.CONTEXT_EMBED_SIZE)

            if self.model_args.HN_CONTEXT_ENCODER == "linear":
                modules = [nn.ReLU()]
                for _ in range(self.model_args.HN_CONTEXT_LAYER_NUM):
                    modules += [nn.Linear(self.model_args.CONTEXT_EMBED_SIZE,
                                          self.model_args.CONTEXT_EMBED_SIZE),
                                nn.ReLU()]
                self.context_encoder_HN = nn.Sequential(*modules)
            else:
                ctx_layers = TransformerEncoderLayerResidual(
                    self.model_args.CONTEXT_EMBED_SIZE,
                    self.model_args.NHEAD,
                    self.model_args.DIM_FEEDFORWARD,
                    self.model_args.DROPOUT,
                )
                self.context_encoder_HN = TransformerEncoder(
                    ctx_layers, self.model_args.HN_CONTEXT_LAYER_NUM, norm=None)

            HN = self.model_args.CONTEXT_EMBED_SIZE
            self.hnet_embed_weight = nn.Linear(HN, limb_obs_size * self.d_model)
            self.hnet_embed_bias   = nn.Linear(HN, self.d_model)
            self.decoder_dims      = ([decoder_input_dim]
                                      + self.model_args.DECODER_DIMS
                                      + [decoder_out_dim])
            self.hnet_decoder_weight = nn.ModuleList([
                nn.Linear(HN, self.decoder_dims[i] * self.decoder_dims[i+1])
                for i in range(len(self.decoder_dims) - 1)])
            self.hnet_decoder_bias = nn.ModuleList([
                nn.Linear(HN, self.decoder_dims[i+1])
                for i in range(len(self.decoder_dims) - 1)])

        if self.model_args.USE_SWAT_PE:
            self.swat_PE_encoder = SWATPEEncoder(self.d_model, self.seq_len)

        # GCN wired externally by ActorCritic
        self.gcn      = None
        self.gcn_proj = None
        self.dropout  = nn.Dropout(p=0.1)
        # t-SNE embedding buffer — activated by runner during collection
        self._tsne_buffer = {"embeds": [], "active": False}
        self.init_weights()

    def init_weights(self):
        if not self.model_args.PER_NODE_EMBED:
            r = self.model_args.EMBED_INIT
            self.limb_embed.weight.data.uniform_(-r, r)
        if not self.model_args.PER_NODE_DECODER:
            r = self.model_args.DECODER_INIT
            self.decoder[-1].bias.data.zero_()
            self.decoder[-1].weight.data.uniform_(-r, r)
        if self.model_args.FIX_ATTENTION:
            r = self.model_args.EMBED_INIT
            self.context_embed_attention.weight.data.uniform_(-r, r)
        if self.model_args.HYPERNET:
            r = self.model_args.HN_EMBED_INIT
            self.context_embed_HN.weight.data.uniform_(-r, r)
            r = self.model_args.EMBED_INIT
            self.hnet_embed_weight.weight.data.zero_()
            self.hnet_embed_weight.bias.data.uniform_(-r, r)
            self.hnet_embed_bias.weight.data.zero_()
            self.hnet_embed_bias.bias.data.zero_()
            r = self.model_args.DECODER_INIT
            for i in range(len(self.hnet_decoder_weight)):
                self.hnet_decoder_weight[i].weight.data.zero_()
                self.hnet_decoder_weight[i].bias.data.uniform_(-r, r)
                self.hnet_decoder_bias[i].weight.data.zero_()
                self.hnet_decoder_bias[i].bias.data.zero_()

    def forward(self, obs, obs_mask, obs_env, obs_cm_mask, obs_context,
                morphology_info, return_attention=False,
                dropout_mask=None, unimal_ids=None):
        _, batch_size, limb_obs_size = obs.shape

        if self.model_args.FIX_ATTENTION:
            ctx_attn = self.context_embed_attention(obs_context)
            if self.model_args.CONTEXT_ENCODER == "transformer":
                ctx_attn = self.context_encoder_attention(
                    ctx_attn, src_key_padding_mask=obs_mask,
                    morphology_info=morphology_info)
            else:
                ctx_attn = self.context_encoder_attention(ctx_attn)

        if self.model_args.HYPERNET:
            ctx_hn = self.context_embed_HN(obs_context)
            ctx_hn = self.context_encoder_HN(ctx_hn)

        # Embedding
        if self.model_args.HYPERNET and self.model_args.HN_EMBED:
            ew = self.hnet_embed_weight(ctx_hn).reshape(
                self.seq_len, batch_size, limb_obs_size, self.d_model)
            eb = self.hnet_embed_bias(ctx_hn)
            obs_embed = (obs[:, :, :, None] * ew).sum(dim=-2) + eb
        else:
            if self.model_args.PER_NODE_EMBED:
                obs_embed = (
                    obs[:, :, :, None] *
                    self.limb_embed_weights[:, unimal_ids, :, :]
                ).sum(dim=-2) + self.limb_embed_bias[:, unimal_ids, :]
            else:
                obs_embed = self.limb_embed(obs)

        if self.model_args.EMBEDDING_SCALE:
            obs_embed *= math.sqrt(self.d_model)

        # GCN injection
        if self.gcn is not None and morphology_info is not None:
            X      = morphology_info.get("graph_node_features")
            A_norm = morphology_info.get("graph_A_norm")
            if X is not None and A_norm is not None:
                gcn_emb   = self.gcn(X.float(), A_norm.float())
                gcn_emb   = gcn_emb.permute(1, 0, 2)
                obs_embed = obs_embed + self.gcn_proj(gcn_emb)
        
        # NEW: FiLM modulation
        # Runs after GCN so it modulates the full (limb_embed + gcn) signal.
        # Runs before pos encoding so position is added on top, not modulated.
        # obs_context is (L, B, 12) — same tensor FIX_ATTENTION reads.
        if self.film_generator is not None:
            gamma, beta = self.film_generator(obs_context)
            obs_embed   = gamma * obs_embed + beta   # NEW

        
        # Capture embeddings for t-SNE if active
        if self._tsne_buffer["active"]:
            self._tsne_buffer["embeds"].append(obs_embed.detach().cpu())

        if self.model_args.POS_EMBEDDING in ["learnt", "abs"]:
            obs_embed = self.pos_embedding(obs_embed)
        if self.model_args.USE_SWAT_PE:
            obs_embed = self.swat_PE_encoder(obs_embed, morphology_info["traversals"])

        if self.model_args.EMBEDDING_DROPOUT:
            obs_embed    = self.dropout(obs_embed)
            dropout_mask = 0.
        else:
            dropout_mask = 0.

        ctx_to_base = ctx_attn if self.model_args.FIX_ATTENTION else None
        attn_mask   = morphology_info["SWAT_RE"] if self.model_args.USE_SWAT_RE else None

        if return_attention:
            obs_embed_t, attn_maps = self.transformer_encoder.get_attention_maps(
                obs_embed, mask=attn_mask,
                src_key_padding_mask=obs_mask,
                context=ctx_to_base, morphology_info=morphology_info)
        else:
            obs_embed_t = self.transformer_encoder(
                obs_embed, mask=attn_mask,
                src_key_padding_mask=obs_mask,
                context=ctx_to_base, morphology_info=morphology_info)
            attn_maps = None

        decoder_input = obs_embed_t

        if self.model_args.HYPERNET and self.model_args.HN_DECODER:
            output = decoder_input
            for i in range(len(self.hnet_decoder_weight)):
                lw = self.hnet_decoder_weight[i](ctx_hn).reshape(
                    self.seq_len, batch_size,
                    self.decoder_dims[i], self.decoder_dims[i+1])
                lb = self.hnet_decoder_bias[i](ctx_hn)
                output = (output[:, :, :, None] * lw).sum(dim=-2) + lb
                if i != len(self.hnet_decoder_weight) - 1:
                    output = F.relu(output)
        else:
            if self.model_args.PER_NODE_DECODER:
                output = (
                    decoder_input[:, :, :, None] *
                    self.decoder_weights[:, unimal_ids, :, :]
                ).sum(dim=-2) + self.decoder_bias[:, unimal_ids, :]
            else:
                output = self.decoder(decoder_input)

        output = output.permute(1, 0, 2).reshape(batch_size, -1)
        return output, attn_maps, dropout_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(seq_len, 1, d_model))

    def forward(self, x):
        return x + self.pe


class PositionalEncoding1D(nn.Module):
    """Sinusoidal 1-D positional encoding (POS_EMBEDDING == 'abs')."""
    def __init__(self, d_model, seq_len):
        super().__init__()
        pe       = torch.zeros(seq_len, 1, d_model)
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class SWATPEEncoder(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        travs       = cfg.MODEL.TRANSFORMER.TRAVERSALS
        self.pe_dim = [d_model // len(travs) for _ in travs]
        self.pe_dim[-1] = d_model - self.pe_dim[0] * (len(travs) - 1)
        self.swat_pe = nn.ModuleList(
            [nn.Embedding(seq_len, dim) for dim in self.pe_dim])

    def forward(self, x, indexes):
        parts = []
        for i in range(len(cfg.MODEL.TRANSFORMER.TRAVERSALS)):
            parts.append(self.swat_pe[i](indexes[:, :, i]))
        return x + torch.cat(parts, dim=-1)


class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.seq_len        = cfg.MODEL.MAX_LIMBS
        self.graph_encoding = cfg.MODEL.GRAPH_ENCODING

        self.gcn = build_gcn_from_cfg(cfg)
        if self.gcn is not None:
            print(f"[ActorCritic] GCN enabled: mode={self.graph_encoding}, "
                  f"out_dim={cfg.MODEL.GCN.OUT_DIM}")

        self.v_net       = TransformerModel(obs_space, 1)
        self.mu_net      = TransformerModel(obs_space, 1)
        self.num_actions = cfg.MODEL.MAX_LIMBS

        if self.gcn is not None:
            d_model  = cfg.MODEL.LIMB_EMBED_SIZE
            gcn_out  = cfg.MODEL.GCN.OUT_DIM
            self.gcn_proj        = nn.Linear(gcn_out, d_model)
            self.v_net.gcn       = self.gcn
            self.v_net.gcn_proj  = self.gcn_proj
            self.mu_net.gcn      = self.gcn
            self.mu_net.gcn_proj = self.gcn_proj
            print(f"[ActorCritic] GCN wired: gcn_out={gcn_out} → d_model={d_model}")

        if cfg.MODEL.ACTION_STD_FIXED:
            log_std = np.log(cfg.MODEL.ACTION_STD)
            self.log_std = nn.Parameter(
                log_std * torch.ones(1, self.num_actions), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, self.num_actions))

    def forward(self, obs, act=None, return_attention=False,
                dropout_mask_v=None, dropout_mask_mu=None,
                unimal_ids=None, compute_val=True):

        batch_size   = cfg.PPO.BATCH_SIZE if act is not None else cfg.PPO.NUM_ENVS
        obs_env      = {k: obs[k] for k in cfg.ENV.KEYS_TO_KEEP}
        obs_cm_mask  = obs.get("obs_padding_cm_mask", None)

        obs_raw      = obs["proprioceptive"]
        obs_mask     = obs["obs_padding_mask"].bool()
        act_mask     = obs["act_padding_mask"].bool()
        obs_context  = obs["context"]

        morphology_info = {}
        if cfg.MODEL.TRANSFORMER.USE_SWAT_PE:
            morphology_info["traversals"] = obs["traversals"].permute(1, 0, 2).long()
        if cfg.MODEL.TRANSFORMER.USE_SWAT_RE:
            morphology_info["SWAT_RE"] = obs["SWAT_RE"]
        if self.gcn is not None:
            morphology_info["graph_node_features"] = obs["graph_node_features"]
            morphology_info["graph_A_norm"]        = obs["graph_A_norm"]
        if not morphology_info:
            morphology_info = None

        obs_t   = obs_raw.reshape(batch_size, self.seq_len, -1).permute(1, 0, 2)
        obs_ctx = obs_context.reshape(batch_size, self.seq_len, -1).permute(1, 0, 2)

        if compute_val:
            limb_vals, v_attn, dmv = self.v_net(
                obs_t, obs_mask, obs_env, obs_cm_mask, obs_ctx,
                morphology_info, return_attention=return_attention,
                dropout_mask=dropout_mask_v, unimal_ids=unimal_ids)
            limb_vals = limb_vals * (1 - obs_mask.int())
            num_limbs = self.seq_len - torch.sum(obs_mask.int(), dim=1, keepdim=True)
            val       = torch.divide(torch.sum(limb_vals, dim=1, keepdim=True), num_limbs)
        else:
            val, v_attn, dmv = 0., None, 0.

        mu, mu_attn, dmmu = self.mu_net(
            obs_t, obs_mask, obs_env, obs_cm_mask, obs_ctx,
            morphology_info, return_attention=return_attention,
            dropout_mask=dropout_mask_mu, unimal_ids=unimal_ids)

        std = torch.exp(self.log_std)
        pi  = Normal(mu, std)

        if act is not None:
            logp          = pi.log_prob(act)
            logp[act_mask] = 0.0
            logp    = logp.sum(-1, keepdim=True)
            entropy = pi.entropy()
            entropy[act_mask] = 0.0
            entropy = entropy.mean()
            return val, pi, logp, entropy, dmv, dmmu
        else:
            if return_attention:
                return val, pi, v_attn, mu_attn, dmv, dmmu
            return val, pi, None, None, dmv, dmmu


class Agent:
    def __init__(self, actor_critic):
        self.ac = actor_critic

    @torch.no_grad()
    def act(self, obs, return_attention=False,
            dropout_mask_v=None, dropout_mask_mu=None,
            unimal_ids=None, compute_val=True):
        val, pi, _, _, dmv, dmmu = self.ac(
            obs, return_attention=return_attention,
            dropout_mask_v=dropout_mask_v, dropout_mask_mu=dropout_mask_mu,
            unimal_ids=unimal_ids, compute_val=compute_val)
        act  = pi.loc if cfg.DETERMINISTIC else pi.sample()
        logp = pi.log_prob(act)
        act_mask = obs["act_padding_mask"].bool()
        logp[act_mask] = 0.0
        logp = logp.sum(-1, keepdim=True)
        return val, act, logp, dmv, dmmu

    @torch.no_grad()
    def get_value(self, obs, dropout_mask_v=None,
                  dropout_mask_mu=None, unimal_ids=None):
        val, _, _, _, _, _ = self.ac(
            obs, dropout_mask_v=dropout_mask_v,
            dropout_mask_mu=dropout_mask_mu, unimal_ids=unimal_ids)
        return val