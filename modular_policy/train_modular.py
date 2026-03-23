"""
legged_gym/scripts/train_modular.py

Entry point. Imports isaacgym FIRST, then creates G1Robot and runs your PPO.

Usage (from unitree_rl_gym root):
    python legged_gym/scripts/train_modular.py \
        --xml_path /path/to/g1_12dof_stripped.xml \
        --num_envs 4096 \
        --out_dir ./output_walk_isaac \
        --headless
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xml_path",  type=str, required=True,
                   help="MuJoCo XML for graph structure (g1_12dof_stripped.xml)")
    p.add_argument("--num_envs",  type=int, default=4096)
    p.add_argument("--out_dir",   type=str, default="./output_walk_isaac")
    p.add_argument("--sim_device",type=str, default="cuda:0")
    p.add_argument("--rl_device", type=str, default="cuda:0")
    p.add_argument("--headless",  action="store_true", default=True)
    p.add_argument("--resume",    type=str, default=None)
    p.add_argument("--seed",      type=int, default=1409)
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--graph_encoding", type=str, default="topological",
                   choices=["none", "onehot", "topological"])
    return p.parse_args()


def main():
    args = parse_args()

    # modular_policy lives at unitree_rl_gym/modular_policy/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.abspath(os.path.join(script_dir, "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from modular_policy.config import cfg
    from modular_policy.algos.ppo.runner import ModularRunner

    cfg.PPO.NUM_ENVS             = args.num_envs
    cfg.PPO.MAX_ITERS            = args.max_iters
    cfg.PPO.EARLY_EXIT_MAX_ITERS = args.max_iters
    cfg.MODEL.GRAPH_ENCODING     = args.graph_encoding
    cfg.ENV.WALKERS              = ["g1"]
    cfg.DEVICE                   = args.rl_device
    cfg.OUT_DIR                  = args.out_dir

    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    from legged_gym.envs.g1.g1_env    import G1Robot
    from legged_gym.envs.g1.g1_config import G1RoughCfg
    from legged_gym.utils.helpers     import parse_sim_params, class_to_dict, set_seed

    set_seed(args.seed)
    env_cfg              = G1RoughCfg()
    env_cfg.env.num_envs = args.num_envs

    class _Args:
        physics_engine    = gymapi.SIM_PHYSX
        sim_device        = args.sim_device       # e.g. "cuda:0"
        sim_device_type   = "cuda"                # used to rebuild sim_device
        compute_device_id = 0                     # GPU index
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

    sim_params = parse_sim_params(_Args(), {"sim": class_to_dict(env_cfg.sim)})

    print(f"Creating G1Robot with {args.num_envs} envs ...")
    env = G1Robot(
        cfg            = env_cfg,
        sim_params     = sim_params,
        physics_engine = gymapi.SIM_PHYSX,
        sim_device     = args.sim_device,
        headless       = args.headless,
    )
    print(f"G1Robot ready — num_envs={env.num_envs} num_actions={env.num_actions}")

    log_dir = os.path.join(cfg.OUT_DIR, datetime.now().strftime("%b%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    runner = ModularRunner(
        env      = env,
        xml_path = os.path.abspath(args.xml_path),
        log_dir  = log_dir,
        device   = args.rl_device,
    )

    if args.resume:
        runner.load(args.resume)

    runner.learn(num_learning_iterations=cfg.PPO.MAX_ITERS, init_at_random_ep_len=True)


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING train_modular.py — ModularRunner")
    print("=" * 60)
    main()