# generate_inference_pkg.py
import torch
from modular_policy.config import cfg

ckpt_path = "output_film_wide/Mar31_18-18-17/model_400.pt"
out_path  = "output_film_wide/Mar31_18-18-17/model_400_inference.pt"

ckpt = torch.load(ckpt_path, map_location="cpu")

cfg.MODEL.GRAPH_ENCODING       = "rwse"
cfg.MODEL.RWSE_K               = 8
cfg.MODEL.TRANSFORMER.USE_FILM = True
cfg.MODEL.MAX_LIMBS            = 13
cfg.MODEL.MAX_JOINTS           = 12
cfg.MODEL.GCN.HIDDEN_DIM       = 16
cfg.MODEL.GCN.OUT_DIM          = 13
cfg.MODEL.GCN.NUM_LAYERS       = 4

ob_mean = ckpt["ob_mean"]  # (11, 338)
ob_var  = ckpt["ob_var"]   # (11, 338)

# variant order from your training log
variant_names = [
    "robot_variant_0", "robot_variant_1", "robot_variant_2",
    "robot_variant_3", "robot_variant_4", "robot_variant_5",
    "robot_variant_6", "robot_variant_7", "robot_variant_8",
    "robot_variant_9", "g1_12dof"
]

torch.save({
    "model_state_dict":    ckpt["model_state_dict"],
    "ob_mean_per_variant": ob_mean,
    "ob_var_per_variant":  ob_var,
    "ob_mean_ood":         ob_mean.mean(0),
    "ob_var_ood":          ob_var.mean(0),
    "variant_names":       variant_names,
    "graph_encoding":      "rwse",
    "rwse_k":              8,
    "use_film":            True,
    "max_limbs":           13,
    "max_joints":          12,
    "limb_obs_size":       26,
    "gcn_hidden":          16,
    "gcn_out":             13,
    "gcn_layers":          4,
}, out_path)

print(f"Saved → {out_path}")
print(f"ob_mean shape: {ob_mean.shape}")
print(f"variant_names: {variant_names}")