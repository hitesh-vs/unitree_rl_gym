import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from modular_policy.config import cfg
# No gym import — obs_space is _DictSpace from obs_builder (has .spaces attribute)


class Buffer:
    def __init__(self, obs_space, act_shape):
        T, P = cfg.PPO.TIMESTEPS, cfg.PPO.NUM_ENVS

        # _DictSpace has .spaces dict; _BoxSpace has .shape tuple
        if hasattr(obs_space, "spaces"):
            self.obs = {
                k: torch.zeros(T, P, *v.shape)
                for k, v in obs_space.spaces.items()
            }
        else:
            self.obs = torch.zeros(T, P, *obs_space.shape)

        self.act             = torch.zeros(T, P, *act_shape)
        self.val             = torch.zeros(T, P, 1)
        self.rew             = torch.zeros(T, P, 1)
        self.ret             = torch.zeros(T, P, 1)
        self.logp            = torch.zeros(T, P, 1)
        self.masks           = torch.ones(T, P, 1)
        self.timeout         = torch.ones(T, P, 1)
        self.dropout_mask_v  = torch.ones(T, P, 12, 128)
        self.dropout_mask_mu = torch.ones(T, P, 12, 128)
        self.unimal_ids      = torch.zeros(T, P).long()
        self.step            = 0

    def to(self, device):
        if isinstance(self.obs, dict):
            for k in self.obs:
                self.obs[k] = self.obs[k].to(device)
        else:
            self.obs = self.obs.to(device)
        self.act             = self.act.to(device)
        self.val             = self.val.to(device)
        self.rew             = self.rew.to(device)
        self.ret             = self.ret.to(device)
        self.logp            = self.logp.to(device)
        self.masks           = self.masks.to(device)
        self.timeout         = self.timeout.to(device)
        self.dropout_mask_v  = self.dropout_mask_v.to(device)
        self.dropout_mask_mu = self.dropout_mask_mu.to(device)
        self.unimal_ids      = self.unimal_ids.to(device)

    def insert(self, obs, act, logp, val, rew, masks, timeouts,
               dropout_mask_v, dropout_mask_mu, unimal_ids):
        if isinstance(obs, dict):
            for k, v in obs.items():
                self.obs[k][self.step] = v
        else:
            self.obs[self.step] = obs
        self.act[self.step]             = act
        self.val[self.step]             = val
        self.rew[self.step]             = rew
        self.logp[self.step]            = logp
        self.masks[self.step]           = masks
        self.timeout[self.step]         = timeouts
        self.dropout_mask_v[self.step]  = dropout_mask_v
        self.dropout_mask_mu[self.step] = dropout_mask_mu
        self.unimal_ids[self.step]      = torch.LongTensor(unimal_ids)
        self.step = (self.step + 1) % cfg.PPO.TIMESTEPS

    def compute_returns(self, next_value):
        gamma, gae_lambda = cfg.PPO.GAMMA, cfg.PPO.GAE_LAMBDA
        val = torch.cat((self.val.squeeze(), next_value.t())).unsqueeze(2)
        gae = 0
        for step in reversed(range(cfg.PPO.TIMESTEPS)):
            delta = (self.rew[step]
                     + gamma * val[step + 1] * self.masks[step]
                     - val[step]) * self.timeout[step]
            gae            = delta + gamma * gae_lambda * self.masks[step] * gae
            self.ret[step] = gae + val[step]

    def get_sampler(self, adv):
        dset_size = cfg.PPO.TIMESTEPS * cfg.PPO.NUM_ENVS
        assert dset_size >= cfg.PPO.BATCH_SIZE
        sampler = BatchSampler(
            SubsetRandomSampler(range(dset_size)),
            cfg.PPO.BATCH_SIZE, drop_last=True)

        for idxs in sampler:
            batch = {"ret": self.ret.view(-1, 1)[idxs]}
            if isinstance(self.obs, dict):
                batch["obs"] = {
                    k: v.view(-1, *v.size()[2:])[idxs]
                    for k, v in self.obs.items()}
            else:
                batch["obs"] = self.obs.view(-1, *self.obs.size()[2:])[idxs]
            batch["val"]             = self.val.view(-1, 1)[idxs]
            batch["act"]             = self.act.view(-1, self.act.size(-1))[idxs]
            batch["adv"]             = adv.view(-1, 1)[idxs]
            batch["logp_old"]        = self.logp.view(-1, 1)[idxs]
            batch["dropout_mask_v"]  = self.dropout_mask_v.view(-1, 12, 128)[idxs]
            batch["dropout_mask_mu"] = self.dropout_mask_mu.view(-1, 12, 128)[idxs]
            batch["unimal_ids"]      = self.unimal_ids.view(-1)[idxs]
            yield batch