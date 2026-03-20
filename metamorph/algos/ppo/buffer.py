"""Experience rollout buffer for PPO training.

Stores transitions collected from vectorised environments and supports
mini-batch sampling with Generalised Advantage Estimation (GAE).
"""

import torch


class RolloutBuffer:
    """Fixed-size rollout buffer for PPO.

    Stores ``num_steps`` transitions across ``num_envs`` parallel environments
    and computes GAE-λ advantages before yielding mini-batches for training.
    """

    def __init__(self, num_envs, num_steps, obs_shape, act_dim, device):
        """Initialise the buffer.

        Args:
            num_envs (int): Number of parallel environments.
            num_steps (int): Number of steps per rollout.
            obs_shape (tuple): Shape of a single observation (excluding batch).
            act_dim (int): Action dimension.
            device (str or torch.device): Target device for stored tensors.
        """
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device

        self.obs = torch.zeros(num_steps, num_envs, *obs_shape, device=device)
        self.actions = torch.zeros(num_steps, num_envs, act_dim, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, device=device)

        self.step = 0
        self.full = False

    def insert(self, obs, action, log_prob, reward, done, value):
        """Store one transition.

        Args:
            obs (torch.Tensor): Observations of shape (num_envs, *obs_shape).
            action (torch.Tensor): Actions of shape (num_envs, act_dim).
            log_prob (torch.Tensor): Log-probabilities of shape (num_envs,).
            reward (torch.Tensor): Rewards of shape (num_envs,).
            done (torch.Tensor): Done flags of shape (num_envs,).
            value (torch.Tensor): Value estimates of shape (num_envs,).
        """
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.step = (self.step + 1) % self.num_steps
        if self.step == 0:
            self.full = True

    def compute_returns(self, last_value, gamma=0.99, lam=0.95):
        """Compute GAE-λ advantages and discounted returns.

        Args:
            last_value (torch.Tensor): Bootstrap value of shape (num_envs,).
            gamma (float): Discount factor.
            lam (float): GAE λ parameter.
        """
        gae = torch.zeros_like(last_value)
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            )
            gae = delta + gamma * lam * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size):
        """Yield shuffled mini-batches over the stored rollout.

        Args:
            batch_size (int): Mini-batch size.

        Yields:
            dict: Mini-batch with keys ``'obs'``, ``'actions'``,
                ``'old_log_probs'``, ``'returns'``, and ``'advantages'``.
        """
        total = self.num_steps * self.num_envs
        indices = torch.randperm(total, device=self.device)

        obs_flat = self.obs.view(total, *self.obs.shape[2:])
        actions_flat = self.actions.view(total, -1)
        log_probs_flat = self.log_probs.view(total)
        returns_flat = self.returns.view(total)
        advantages_flat = self.advantages.view(total)

        for start in range(0, total, batch_size):
            batch_idx = indices[start:start + batch_size]
            yield {
                'obs': obs_flat[batch_idx],
                'actions': actions_flat[batch_idx],
                'old_log_probs': log_probs_flat[batch_idx],
                'returns': returns_flat[batch_idx],
                'advantages': advantages_flat[batch_idx],
            }

    def reset(self):
        """Reset internal step counter and full flag."""
        self.step = 0
        self.full = False
