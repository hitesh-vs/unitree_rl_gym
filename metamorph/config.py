"""Configuration system for metamorph training.

Provides a lightweight YAML-backed configuration object that supports
nested attribute access.
"""

import yaml


class ConfigNode:
    """Nested configuration node supporting attribute-style access.

    Recursively converts nested dicts into ConfigNode instances so that
    ``cfg.ppo.lr`` works instead of ``cfg['ppo']['lr']``.
    """

    def __init__(self, d=None):
        if d is not None:
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, ConfigNode(v))
                else:
                    setattr(self, k, v)

    def get(self, key, default=None):
        """Return attribute *key*, or *default* if not present."""
        return getattr(self, key, default)

    def __repr__(self):
        return f"ConfigNode({self.__dict__})"


def load_config(cfg_path):
    """Load configuration from a YAML file.

    Args:
        cfg_path (str): Path to YAML config file.

    Returns:
        ConfigNode: Nested configuration object.
    """
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return ConfigNode(cfg_dict)


def merge_configs(base_cfg, override_dict):
    """Override config values from a plain dictionary.

    Recursively merges *override_dict* into *base_cfg*.  Nested dicts
    are merged depth-first; scalar values are overwritten.

    Args:
        base_cfg (ConfigNode): Base configuration to update in-place.
        override_dict (dict): Values to override.

    Returns:
        ConfigNode: The updated *base_cfg*.
    """
    for k, v in override_dict.items():
        if isinstance(v, dict) and hasattr(base_cfg, k):
            merge_configs(getattr(base_cfg, k), v)
        else:
            setattr(base_cfg, k, v)
    return base_cfg
