"""Training script for the G1 robot with a Transformer+GCN policy.

Integrates:
- :class:`~legged_gym.envs.g1.g1.G1GraphRobot` – IsaacGym vectorised
  environment (up to 4096 parallel envs).
- :class:`~metamorph.algos.ppo.model.ActorCritic` – Transformer+GCN policy.
- :class:`~metamorph.algos.ppo.ppo.PPOAgent` – PPO training algorithm.
- :class:`~legged_gym.algos.ppo.graph_obs_adapter.GraphObsAdapter` –
  converts flat IsaacGym observations to the policy dict format.

Usage::

    python legged_gym/scripts/train_g1_custom.py --cfg config/g1/default.yaml
"""

import os
import argparse
import time
from datetime import datetime

import isaacgym  # noqa: F401 - must be imported before torch
import isaacgym.gymapi as gymapi
import torch
from torch.utils.tensorboard import SummaryWriter

from legged_gym.envs import *  # noqa: F401,F403 - registers tasks
from legged_gym.envs.g1.g1 import G1GraphRobot
from legged_gym.envs.g1.g1_config import G1RoughCfg
from legged_gym.algos.ppo.graph_obs_adapter import GraphObsAdapter
from legged_gym.utils.helpers import get_args, parse_sim_params, set_seed
from metamorph.config import load_config
from metamorph.algos.ppo.model import ActorCritic
from metamorph.algos.ppo.ppo import PPOAgent
from metamorph.algos.ppo.buffer import RolloutBuffer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train G1 robot with Transformer+GCN policy'
    )
    parser.add_argument(
        '--cfg',
        type=str,
        default='config/g1/default.yaml',
        help='Path to the YAML config file',
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        default=True,
        help='Run without rendering',
    )
    parser.add_argument(
        '--sim_device',
        type=str,
        default='cuda:0',
        help='IsaacGym simulation device',
    )
    parser.add_argument(
        '--rl_device',
        type=str,
        default='cuda:0',
        help='RL training device',
    )
    parser.add_argument(
        '--num_envs',
        type=int,
        default=None,
        help='Override number of parallel environments',
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to a checkpoint to resume training from',
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=None,
        help='Override the log/checkpoint directory',
    )
    return parser.parse_args()


def setup_logging(log_dir):
    """Create *log_dir* and return a TensorBoard SummaryWriter."""
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[Train] Logging to: {log_dir}")
    return writer


def save_checkpoint(model, optimizer_state_dict, iteration, log_dir):
    """Save model and optimiser state to *log_dir*.

    Two files are written: a numbered snapshot and a ``checkpoint_latest.pt``
    alias for easy resumption.

    Args:
        model (ActorCritic): The model whose state to save.
        optimizer_state_dict (dict): Optimiser state from ``optimizer.state_dict()``.
        iteration (int): Current training iteration (used in the file name).
        log_dir (str): Directory to write checkpoints into.

    Returns:
        str: Path of the numbered checkpoint file.
    """
    payload = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_state_dict,
    }
    ckpt_path = os.path.join(log_dir, f'checkpoint_{iteration:06d}.pt')
    torch.save(payload, ckpt_path)
    torch.save(payload, os.path.join(log_dir, 'checkpoint_latest.pt'))
    return ckpt_path


def load_checkpoint(model, agent, ckpt_path, device='cpu'):
    """Restore model and optimiser state from a checkpoint.

    Args:
        model (ActorCritic): Model to restore weights into.
        agent (PPOAgent): Agent whose optimiser state to restore.
        ckpt_path (str): Path to the ``.pt`` checkpoint file.
        device (str): Device to map tensors to when loading.

    Returns:
        int: The iteration number stored in the checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    agent.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['iteration']


def train():
    args = parse_args()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    cfg = load_config(args.cfg)

    device = args.rl_device

    if args.log_dir is not None:
        log_dir = args.log_dir
    else:
        timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('logs', 'g1_custom', timestamp)

    writer = setup_logging(log_dir)

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    # Build IsaacGym arguments expected by the helpers module.
    isaac_args = get_args()
    isaac_args.headless = args.headless
    isaac_args.sim_device = args.sim_device
    isaac_args.rl_device = args.rl_device

    env_cfg = G1RoughCfg()
    if args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs

    seed = int(getattr(cfg, 'seed', 42))
    set_seed(seed)
    sim_params = parse_sim_params(isaac_args, {'sim': {}})

    print("[Train] Creating G1GraphRobot environment ...")
    env = G1GraphRobot(
        cfg=env_cfg,
        sim_params=sim_params,
        physics_engine=gymapi.SIM_PHYSX,
        sim_device=args.sim_device,
        headless=args.headless,
    )

    num_envs = env_cfg.env.num_envs
    obs_dim = env_cfg.env.num_observations
    act_dim = env_cfg.env.num_actions

    print(f"[Train] Environment: {num_envs} envs, obs_dim={obs_dim}, act_dim={act_dim}")

    # ------------------------------------------------------------------
    # Graph observation adapter
    # ------------------------------------------------------------------
    graph_tensors = env.get_graph_tensors()
    adapter = GraphObsAdapter(graph_tensors=graph_tensors, device=device)

    # Static graph data dict injected into obs_dict during training
    graph_data = None
    if graph_tensors is not None:
        graph_data = {
            'graph_node_features': graph_tensors['node_features'].to(device),
            'graph_adj_normalized': graph_tensors['adj_normalized'].to(device),
        }

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cfg.model.obs_dim = obs_dim
    cfg.model.act_dim = act_dim
    if graph_tensors is not None:
        cfg.model.num_limbs = graph_tensors['num_nodes']
        cfg.model.node_feature_dim = graph_tensors['feature_dim']

    model = ActorCritic(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Model parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # PPO agent & rollout buffer
    # ------------------------------------------------------------------
    agent = PPOAgent(model, cfg, device)

    start_iteration = 0
    if args.resume is not None:
        print(f"[Train] Resuming from: {args.resume}")
        start_iteration = load_checkpoint(model, agent, args.resume, device=device)
        print(f"[Train] Resumed at iteration {start_iteration}")

    num_steps = int(getattr(cfg.ppo, 'num_steps', 24))
    buffer = RolloutBuffer(
        num_envs=num_envs,
        num_steps=num_steps,
        obs_shape=(obs_dim,),
        act_dim=act_dim,
        device=device,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    max_iterations = int(getattr(cfg.training, 'max_iterations', 10000))
    save_interval = int(getattr(cfg.training, 'save_interval', 500))
    log_interval = int(getattr(cfg.training, 'log_interval', 50))

    print(f"[Train] Starting training for {max_iterations} iterations …")

    obs, _, _, _, _ = env.reset()
    obs_dict = adapter.convert(obs)
    if graph_data is not None:
        obs_dict.update(graph_data)

    total_steps = 0

    for iteration in range(start_iteration, max_iterations):
        iter_start = time.time()

        # Collect rollout
        obs_dict = agent.collect_rollout(env, buffer, obs_dict, num_steps)
        if graph_data is not None:
            obs_dict.update(graph_data)

        total_steps += num_steps * num_envs

        # Update policy
        metrics = agent.update(buffer, graph_data=graph_data)
        buffer.reset()

        iter_time = time.time() - iter_start
        fps = (num_steps * num_envs) / iter_time

        # Logging
        if (iteration + 1) % log_interval == 0:
            print(
                f"[Iter {iteration + 1:6d}] "
                f"policy_loss={metrics['policy_loss']:.4f}  "
                f"value_loss={metrics['value_loss']:.4f}  "
                f"entropy={metrics['entropy']:.4f}  "
                f"fps={fps:.0f}"
            )
            writer.add_scalar('train/policy_loss', metrics['policy_loss'], iteration)
            writer.add_scalar('train/value_loss', metrics['value_loss'], iteration)
            writer.add_scalar('train/entropy', metrics['entropy'], iteration)
            writer.add_scalar('train/fps', fps, iteration)
            writer.add_scalar('train/total_steps', total_steps, iteration)

        # Checkpointing
        if (iteration + 1) % save_interval == 0:
            ckpt_path = save_checkpoint(
                model, agent.optimizer.state_dict(), iteration + 1, log_dir
            )
            print(f"[Train] Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    save_checkpoint(model, agent.optimizer.state_dict(), max_iterations, log_dir)
    writer.close()
    print("[Train] Training complete!")


if __name__ == '__main__':
    train()
