"""
modular_policy/config.py

Standalone config — no yacs, no metamorph dependency.
Plain Python namespace that mirrors every cfg.* field used by the model.
Edit the values at the bottom of this file or call cfg.update() at runtime.
"""


class _Namespace:
    """Simple dot-access namespace. Supports nested assignment."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def update(self, d):
        for k, v in d.items():
            parts = k.split(".")
            obj = self
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], v)

    def __repr__(self):
        return str(self.__dict__)


# ---------------------------------------------------------------------------
# Build cfg
# ---------------------------------------------------------------------------

cfg = _Namespace()

# ── ENV ──────────────────────────────────────────────────────────────────────
cfg.ENV = _Namespace(
    WALKERS        = ["g1"],          # agent names for TrainMeter
    KEYS_TO_KEEP   = [],              # no hfield for G1
    TASK           = "locomotion",
    ENV_NAME       = "Modular-v0",
)
cfg.ENV_NAME = "Modular-v0"

# ── PPO ───────────────────────────────────────────────────────────────────────
cfg.PPO = _Namespace(
    GAMMA            = 0.99,
    GAE_LAMBDA       = 0.95,
    CLIP_EPS         = 0.2,
    EPOCHS           = 4,
    BATCH_SIZE       = 4096,
    VALUE_COEF       = 0.2,
    KL_TARGET_COEF   = 20.0,
    USE_CLIP_VALUE_FUNC = True,
    ENTROPY_COEF     = 0.005,
    TIMESTEPS        = 2560,
    NUM_ENVS         = 512,
    BASE_LR          = 6e-4,
    MIN_LR           = 0.0,
    LR_POLICY        = "cos",
    WARMUP_FACTOR    = 0.1,
    WARMUP_ITERS     = 5,
    EPS              = 1e-5,
    WEIGHT_DECAY     = 0.0,
    MAX_GRAD_NORM    = 0.5,
    MAX_ITERS        = 3000,
    EARLY_EXIT       = False,
    EARLY_EXIT_MAX_ITERS = 3000,
    CHECKPOINT_PATH  = "",
)

# ── MODEL ─────────────────────────────────────────────────────────────────────
cfg.MODEL = _Namespace(
    TYPE             = "transformer",
    LIMB_EMBED_SIZE  = 128,
    JOINT_EMBED_SIZE = 128,
    MAX_JOINTS       = 12,
    MAX_LIMBS        = 13,
    ACTION_STD       = 0.9,
    MORPH_CTX_DIM    = 12,
    ACTION_STD_FIXED = False,
    GRAPH_ENCODING   = "topological",   # "none" | "onehot" | "topological"
    OBS_TO_NORM      = ["proprioceptive"],
    BASE_CONTEXT_NORM = "running",
    CONTEXT_OBS_TYPES = [
        "body_pos", "body_ipos", "body_iquat", "geom_quat",
        "body_mass", "body_shape",
        "jnt_pos",
        "joint_range", "joint_axis", "gear",
    ],
    OBS_TYPES = [
        "proprioceptive", "edges", "obs_padding_mask", "act_padding_mask",
        "context", "traversals", "SWAT_RE",
    ],
    DETERMINISTIC = False,
)

cfg.DETERMINISTIC = False

# ── MODEL.TRANSFORMER ─────────────────────────────────────────────────────────
cfg.MODEL.TRANSFORMER = _Namespace(
    NHEAD               = 2,
    DIM_FEEDFORWARD     = 1024,
    DROPOUT             = 0.0,
    NLAYERS             = 5,
    EMBED_INIT          = 0.1,
    DECODER_INIT        = 0.01,
    HN_EMBED_INIT       = 0.1,
    DECODER_DIMS        = [],
    EXT_HIDDEN_DIMS     = [],
    EXT_MIX             = "none",
    POS_EMBEDDING       = None,        # "learnt" | "abs" | None
    EMBEDDING_DROPOUT   = False,
    CONSISTENT_DROPOUT  = False,
    EMBEDDING_SCALE     = True,
    HYPERNET            = False,
    CONTEXT_EMBED_SIZE  = 128,
    HN_EMBED            = True,
    HN_DECODER          = True,
    HN_CONTEXT_ENCODER  = "linear",
    HN_CONTEXT_LAYER_NUM = 1,
    FIX_ATTENTION       = True,
    CONTEXT_LAYER       = 3,
    LINEAR_CONTEXT_LAYER = 2,
    CONTEXT_ENCODER     = "linear",
    HFIELD_IN_FIX_ATTENTION = False,
    USE_MORPHOLOGY_INFO_IN_ATTENTION = False,
    USE_SWAT_PE         = False,
    USE_SWAT_RE         = False,
    TRAVERSALS          = ["pre", "inlcrs", "postlcrs"],
    PER_NODE_EMBED      = False,
    PER_NODE_DECODER    = False,
)

# ── MODEL.GCN ─────────────────────────────────────────────────────────────────
cfg.MODEL.GCN = _Namespace(
    HIDDEN_DIM = 16,
    OUT_DIM    = 13,
    NUM_LAYERS = 4,
)

# ── MODEL.MLP ─────────────────────────────────────────────────────────────────
cfg.MODEL.MLP = _Namespace(
    HIDDEN_DIM       = 256,
    LAYER_NUM        = 3,
    CONSISTENT_PADDING = None,
)

# ── MODEL.FINETUNE ────────────────────────────────────────────────────────────
cfg.MODEL.FINETUNE = _Namespace(
    FULL_MODEL      = True,
    LAYER_SUBSTRING = [],
)

# ── TASK_SAMPLING ─────────────────────────────────────────────────────────────
cfg.TASK_SAMPLING = _Namespace(
    EMA_ALPHA  = 0.1,
    PROB_ALPHA = 1.0,
    AVG_TYPE   = "ema",
)

# RWSE params
cfg.MODEL.RWSE_K = 8

# ── MISC ──────────────────────────────────────────────────────────────────────
cfg.OUT_DIR        = "./output_walk_isaac"
cfg.LOG_PERIOD     = 10
cfg.CHECKPOINT_PERIOD = 100
cfg.DEVICE         = "cuda:0"
cfg.RNG_SEED       = 1409