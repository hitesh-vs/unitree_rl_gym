import isaacgym
from isaacgym import gymapi
import argparse
import os
import torch
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Legged Gym / RSL_RL Imports
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.multi_variant_env import MultiVariantG1Robot
from legged_gym.utils.helpers import parse_sim_params, class_to_dict, set_seed
from rsl_rl.runners import OnPolicyRunner

# ── Wrapper to bridge Multi-Variant Env with MLP Runner ─────────────────────
class MLPLoggingWrapper:
    def __init__(self, env):
        self.env = env
        self.num_obs = 47  
        self.num_privileged_obs = self.env.num_privileged_obs
        self.num_actions = self.env.num_actions
        self.num_envs = self.env.num_envs
        self.device = self.env.device

    def get_observations(self):
        obs = self.env.get_observations()
        return obs["proprioception"] if isinstance(obs, dict) else obs

    def get_privileged_observations(self):
        return self.env.get_privileged_observations()

    def step(self, actions):
        # Unpack 5 values (standard for Unitree/Legged Gym)
        obs, priv_obs, rewards, dones, infos = self.env.step(actions)
        obs_out = obs["proprioception"] if isinstance(obs, dict) else obs
        return obs_out, priv_obs, rewards, dones, infos

    def reset(self):
        # Env reset returns (obs, priv_obs)
        obs, priv_obs = self.env.reset()
        obs_out = obs["proprioception"] if isinstance(obs, dict) else obs
        return obs_out, priv_obs

    def __getattr__(self, name):
        return getattr(self.env, name)

# ── Main Training Function ──────────────────────────────────────────────────
def train_baseline():
    p = argparse.ArgumentParser(description="Multi-variant MLP Baseline")
    p.add_argument("--variants_metadata", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--out_dir", type=str, default="./output_baseline")
    p.add_argument("--seed", type=int, default=1409)
    args = p.parse_args()
    
    set_seed(args.seed)
    
    env_cfg = G1RoughCfg()
    train_cfg = G1RoughCfgPPO()
    env_cfg.env.num_envs = args.num_envs
    
    sim_args = argparse.Namespace(
        physics_engine=gymapi.SIM_PHYSX,
        sim_device="cuda:0", rl_device="cuda:0",
        headless=True, use_gpu=True, use_gpu_pipeline=True,
        subscenes=0, num_threads=10, slices=0, num_subscenes=0,
        compute_device_id=0, sim_device_type="cuda"
    )

    sim_params = parse_sim_params(sim_args, {"sim": class_to_dict(env_cfg.sim)})
    
    raw_env = MultiVariantG1Robot(
        cfg=env_cfg, sim_params=sim_params, physics_engine=gymapi.SIM_PHYSX,
        sim_device="cuda:0", headless=True, variants_metadata_path=args.variants_metadata
    )

    env = MLPLoggingWrapper(raw_env)
    
    train_cfg_dict = class_to_dict(train_cfg)
    train_cfg_dict["runner"]["max_iterations"] = args.max_iters
    
    log_dir = os.path.join(args.out_dir, datetime.now().strftime("%b%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device="cuda:0")
    
    variant_names = raw_env.variant_names
    variant_ids = raw_env.env_variant_ids
    
    print("\n" + "="*50)
    print("RUNNING BASELINE — MLP (No Morphology Context)")
    print("="*50 + "\n")

    start_time = time.time()
    for it in range(args.max_iters):
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=(it == 0))
        
        fps = int(env.num_envs / (time.time() - start_time))
        print(f"\nIter {it}/{args.max_iters} | FPS {fps}")
        
        # Calculate per-variant stats
        for i, name in enumerate(variant_names):
            mask = (variant_ids == i)
            
            if hasattr(raw_env, 'episode_sums'):
                # Sum all reward components manually
                total_reward_batch = torch.zeros(env.num_envs, device=env.device)
                for reward_tensor in raw_env.episode_sums.values():
                    total_reward_batch += reward_tensor
                
                avg_reward = torch.mean(total_reward_batch[mask]).item()
                
                # Correct attribute: episode_length_buf instead of episode_lengths
                if hasattr(raw_env, 'episode_length_buf'):
                    avg_len = torch.mean(raw_env.episode_length_buf[mask].float()).item()
                else:
                    avg_len = 0.0 # Fallback
                
                print(f"Agent {name:20}: reward {avg_reward:7.2f}, ep_len {avg_len:5.1f}")
                writer.add_scalar(f"Agent_Reward/{name}", avg_reward, it)

        start_time = time.time()

if __name__ == "__main__":
    train_baseline()