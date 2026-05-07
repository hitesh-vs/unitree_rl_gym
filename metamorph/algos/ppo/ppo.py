"""Proximal Policy Optimisation (PPO) agent for Transformer+GCN policy.

Manages collecting rollouts from a vectorised IsaacGym environment and
updating the :class:`~metamorph.algos.ppo.model.ActorCritic` policy using
the clipped PPO surrogate objective.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class PPOAgent:
    """PPO agent with clipped surrogate objective.

    Wraps an :class:`~metamorph.algos.ppo.model.ActorCritic` model and
    exposes :meth:`collect_rollout` (environment interaction) and
    :meth:`update` (policy gradient step) as the main API used by the
    training loop in ``train_g1_custom.py``.
    """

    def __init__(self, model, cfg, device):
        """Initialise the PPO agent.

        Args:
            model (ActorCritic): The actor-critic model to train.
            cfg (ConfigNode): Configuration object.  Required sub-node:

                ``cfg.ppo``:
                    - ``lr`` (float): Learning rate (default 3e-4).
                    - ``clip_ratio`` (float): PPO clip ratio ε (default 0.2).
                    - ``value_coef`` (float): Value loss coefficient (default 0.5).
                    - ``entropy_coef`` (float): Entropy bonus coefficient (default 0.01).
                    - ``max_grad_norm`` (float): Gradient clipping norm (default 0.5).
                    - ``num_epochs`` (int): PPO optimisation epochs per rollout (default 10).
                    - ``batch_size`` (int): Mini-batch size (default 256).
                    - ``gamma`` (float): Discount factor (default 0.99).
                    - ``lam`` (float): GAE λ (default 0.95).
            device (str or torch.device): Training device.
        """
        self.model = model
        self.cfg = cfg
        self.device = device

        lr = getattr(cfg.ppo, 'lr', 3e-4)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        self.clip_ratio = getattr(cfg.ppo, 'clip_ratio', 0.2)
        self.value_coef = getattr(cfg.ppo, 'value_coef', 0.5)
        self.entropy_coef = getattr(cfg.ppo, 'entropy_coef', 0.01)
        self.max_grad_norm = getattr(cfg.ppo, 'max_grad_norm', 0.5)
        self.num_epochs = getattr(cfg.ppo, 'num_epochs', 10)
        self.batch_size = getattr(cfg.ppo, 'batch_size', 256)
        self.gamma = getattr(cfg.ppo, 'gamma', 0.99)
        self.lam = getattr(cfg.ppo, 'lam', 0.95)

    def update(self, buffer, graph_data=None):
        """Update the policy using data stored in *buffer*.

        Args:
            buffer (RolloutBuffer): Rollout buffer populated by
                :meth:`collect_rollout`.
            graph_data (dict, optional): Static graph tensors to inject into
                every obs_dict during the update (keys
                ``'graph_node_features'`` and ``'graph_adj_normalized'``).

        Returns:
            dict: Mean training metrics over all mini-batches and epochs:
                ``'policy_loss'``, ``'value_loss'``, ``'entropy'``,
                ``'total_loss'``.
        """
        metrics = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'total_loss': 0.0}
        n_updates = 0

        for _ in range(self.num_epochs):
            for batch in buffer.get_batches(self.batch_size):
                obs_dict = {'proprioceptive': batch['obs']}
                if graph_data is not None:
                    obs_dict.update(graph_data)

                log_prob, entropy, value = self.model.evaluate(obs_dict, batch['actions'])

                # Normalise advantages per mini-batch
                adv = batch['advantages']
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Clipped policy loss
                ratio = torch.exp(log_prob - batch['old_log_probs'])
                clip_adv = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
                policy_loss = -torch.min(ratio * adv, clip_adv).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(value, batch['returns'])

                # Combined loss
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.mean().item()
                metrics['total_loss'] += total_loss.item()
                n_updates += 1

        if n_updates > 0:
            for k in metrics:
                metrics[k] /= n_updates
        return metrics

    def collect_rollout(self, env, buffer, obs_dict, num_steps):
        """Collect *num_steps* transitions from *env* and store in *buffer*.

        Args:
            env: IsaacGym vectorised environment (implements ``step``).
            buffer (RolloutBuffer): Buffer to write transitions into.
            obs_dict (dict): Current observation dictionary.
            num_steps (int): Number of steps to collect per environment.

        Returns:
            dict: Updated observation dictionary after the last step, with
                static graph data keys preserved if they were present.
        """
        self.model.eval()
        # Remember static graph data so we can re-attach it after each step
        graph_keys = {k: v for k, v in obs_dict.items() if k != 'proprioceptive'}

        with torch.no_grad():
            for _ in range(num_steps):
                action, log_prob, value = self.model.act(obs_dict)
                obs, _privileged_obs, reward, done, _info = env.step(action)

                buffer.insert(
                    obs=obs_dict['proprioceptive'],
                    action=action,
                    log_prob=log_prob,
                    reward=reward,
                    done=done.float(),
                    value=value,
                )

                obs_dict = {'proprioceptive': obs}
                obs_dict.update(graph_keys)

            # Bootstrap value for GAE computation
            last_value = self.model.get_value(obs_dict)

        buffer.compute_returns(last_value, gamma=self.gamma, lam=self.lam)
        self.model.train()
        return obs_dict
