"""
legged_gym/scripts/train_modular.py  (also at modular_policy/train_modular.py)

Entry point for single-variant and multi-variant training.

Single variant:
    python modular_policy/train_modular.py \
        --xml_path /path/to/g1_12dof_stripped.xml \
        --num_envs 512 --headless

Multi-variant:
    python modular_policy/train_modular.py \
        --xml_path /path/to/g1_12dof_stripped.xml \
        --variants_metadata /path/to/variants_metadata.json \
        --num_envs 512 --headless
"""

# ── Isaac Gym MUST be first ───────────────────────────────────────────────────
import isaacgym
from isaacgym import gymapi, gymutil
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import sys
import torch
from datetime import datetime

print("=" * 60)
print("RUNNING train_modular.py — ModularRunner")
print("=" * 60, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xml_path",  type=str, required=True,
                   help="Base MuJoCo XML for graph structure")
    p.add_argument("--variants_metadata", type=str, default=None,
                   help="Path to variants_metadata.json for multi-variant training")
    p.add_argument("--num_envs",  type=int, default=512)
    p.add_argument("--out_dir",   type=str, default="./output_walk_isaac")
    p.add_argument("--sim_device",type=str, default="cuda:0")
    p.add_argument("--rl_device", type=str, default="cuda:0")
    p.add_argument("--headless",  action="store_true", default=True)
    p.add_argument("--resume",    type=str, default=None)
    p.add_argument("--seed",      type=int, default=1409)
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--graph_encoding", type=str, default="topological",
               choices=["none", "onehot", "topological", "rwse"])
    p.add_argument("--film", action="store_true", default=False,
                   help="Enable FiLM morphology conditioning on per-limb embeddings")
    p.add_argument("--context_noise", type=float, default=0.0,
               help="Fractional noise on context vector (e.g. 0.05 = 5%)")
    return p.parse_args()


def main():
    args = parse_args()

    # modular_policy must be importable from repo root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.abspath(os.path.join(script_dir, ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from modular_policy.config import cfg
    from modular_policy.algos.ppo.runner import ModularRunner

    cfg.PPO.NUM_ENVS             = args.num_envs
    cfg.PPO.MAX_ITERS            = args.max_iters
    cfg.PPO.EARLY_EXIT_MAX_ITERS = args.max_iters
    cfg.MODEL.GRAPH_ENCODING     = args.graph_encoding
    cfg.MODEL.TRANSFORMER.USE_FILM = args.film
    cfg.MODEL.CONTEXT_NOISE = args.context_noise
    cfg.DEVICE                   = args.rl_device
    cfg.OUT_DIR                  = args.out_dir

    multi_variant = (args.variants_metadata is not None)

    if multi_variant:
        import json
        with open(args.variants_metadata) as f:
            meta = json.load(f)
        cfg.ENV.WALKERS = list(meta.keys())
        print(f"Multi-variant training: {len(cfg.ENV.WALKERS)} variants")
    else:
        cfg.ENV.WALKERS = ["g1"]

    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    # ── Create Isaac Gym env ──────────────────────────────────────────────
    from legged_gym.envs.g1.g1_config import G1RoughCfg
    from legged_gym.utils.helpers import parse_sim_params, class_to_dict, set_seed

    set_seed(args.seed)
    env_cfg              = G1RoughCfg()
    env_cfg.env.num_envs = args.num_envs

    class _Args:
        physics_engine    = gymapi.SIM_PHYSX
        sim_device        = args.sim_device
        rl_device         = args.rl_device
        headless          = args.headless
        use_gpu           = True
        use_gpu_pipeline  = True
        subscenes         = 0
        num_threads       = 10
        num_envs          = None
        seed              = None
        max_iterations    = None
        resume            = False
        experiment_name   = None
        run_name          = None
        load_run          = None
        checkpoint        = None
        device            = args.rl_device
        sim_device_type   = "cuda"
        compute_device_id = 0
        num_subscenes     = 0

    sim_params = parse_sim_params(_Args(), {"sim": class_to_dict(env_cfg.sim)})

    if multi_variant:
        from legged_gym.envs.g1.multi_variant_env import MultiVariantG1Robot
        print(f"Creating MultiVariantG1Robot with {args.num_envs} envs ...")
        env = MultiVariantG1Robot(
            cfg                   = env_cfg,
            sim_params            = sim_params,
            physics_engine        = gymapi.SIM_PHYSX,
            sim_device            = args.sim_device,
            headless              = args.headless,
            variants_metadata_path = args.variants_metadata,
        )
    else:
        from legged_gym.envs.g1.g1_env import G1Robot
        print(f"Creating G1Robot with {args.num_envs} envs ...")
        env = G1Robot(
            cfg            = env_cfg,
            sim_params     = sim_params,
            physics_engine = gymapi.SIM_PHYSX,
            sim_device     = args.sim_device,
            headless       = args.headless,
        )

    print(f"Env ready — num_envs={env.num_envs} num_actions={env.num_actions}")

    log_dir = os.path.join(cfg.OUT_DIR, datetime.now().strftime("%b%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")

    runner = ModularRunner(
        env                    = env,
        xml_path               = os.path.abspath(args.xml_path),
        log_dir                = log_dir,
        device                 = args.rl_device,
        variants_metadata_path = args.variants_metadata,
    )

    if args.resume:
        runner.load(args.resume)

    runner.learn(
        num_learning_iterations=cfg.PPO.MAX_ITERS,
        init_at_random_ep_len=True)


if __name__ == "__main__":
    main()