from collections import defaultdict, deque
import numpy as np
from modular_policy.config import cfg


class AgentMeter:
    def __init__(self, name):
        self.name = name
        self.mean_ep_rews = defaultdict(list)
        self.mean_pos, self.mean_vel, self.mean_metric, self.mean_ep_len = [], [], [], []
        self.ep_rew    = defaultdict(lambda: deque(maxlen=10))
        self.ep_pos    = deque(maxlen=10)
        self.ep_vel    = deque(maxlen=10)
        self.ep_metric = deque(maxlen=10)
        self.ep_count  = 0
        self.ep_len    = deque(maxlen=10)
        self.ep_len_ema = -1

    def add_ep_info(self, infos):
        for info in infos:
            if info["name"] != self.name:
                continue
            if "episode" in info:
                self.ep_rew["reward"].append(info["episode"]["r"])
                self.ep_count += 1
                self.ep_len.append(info["episode"]["l"])
                if self.ep_count == 10:
                    self.ep_len_ema = np.mean(self.ep_len)
                elif self.ep_count >= 10:
                    alpha = cfg.TASK_SAMPLING.EMA_ALPHA
                    self.ep_len_ema = alpha * self.ep_len[-1] + (1 - alpha) * self.ep_len_ema
                for rew_type, rew_ in info["episode"].items():
                    if "__reward__" in rew_type:
                        self.ep_rew[rew_type].append(rew_)

    def update_mean(self):
        if len(self.ep_rew["reward"]) == 0:
            return False
        for rew_type, rews_ in self.ep_rew.items():
            self.mean_ep_rews[rew_type].append(round(np.mean(rews_), 2))
        self.mean_ep_len.append(round(np.mean(self.ep_len), 2))
        return True

    def log_stats(self, max_name_len):
        if len(self.ep_rew["reward"]) == 0:
            return
        ep_rew = self.ep_rew["reward"]
        print(
            "Agent {:>{size}}: mean/median reward {:>4.0f}/{:<4.0f}, "
            "min/max reward {:>4.0f}/{:<4.0f}, "
            "#Ep: {:>7.0f}, avg/ema Ep len: {:>4.0f}/{:>4.0f}".format(
                self.name,
                np.mean(ep_rew), np.median(ep_rew),
                np.min(ep_rew),  np.max(ep_rew),
                self.ep_count,
                np.mean(self.ep_len), self.ep_len_ema,
                size=max_name_len,
            )
        )


class TrainMeter:
    def __init__(self, agent_names):
        self.agents       = agent_names
        self.max_name_len = max(len(a) for a in agent_names)
        self.agent_meters = {a: AgentMeter(a) for a in agent_names}
        self.train_stats  = defaultdict(list)
        self.mean_ep_rews = defaultdict(list)
        self.mean_ep_len  = []

    def add_train_stat(self, stat_type, stat_value):
        self.train_stats[stat_type].append(stat_value)

    def add_ep_info(self, infos):
        for _, meter in self.agent_meters.items():
            meter.add_ep_info(infos)

    def update_mean(self):
        for _, meter in self.agent_meters.items():
            if not meter.update_mean():
                return
        self.mean_ep_len.append(
            round(np.mean([m.mean_ep_len[-1] for m in self.agent_meters.values()]), 2))
        rew_types = self.agent_meters[self.agents[0]].mean_ep_rews.keys()
        for rew_type in rew_types:
            self.mean_ep_rews[rew_type].append(
                round(np.mean([m.mean_ep_rews[rew_type][-1]
                               for m in self.agent_meters.values()]), 2))

    def log_stats(self):
        for _, meter in self.agent_meters.items():
            meter.log_stats(self.max_name_len)
        if len(self.mean_ep_rews["reward"]) > 0:
            print("Agent {:>{size}}: mean/------ reward {:>4.0f}, ".format(
                "__env__", self.mean_ep_rews["reward"][-1],
                size=self.max_name_len))