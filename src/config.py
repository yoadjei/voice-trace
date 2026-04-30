"""
Centralized configuration loader for VoiceTrace.

Usage:
    from src.config import cfg
    
    # Access config values
    model = cfg.api.anthropic.model
    seed = cfg.seeds.numpy
    languages = cfg.languages.evaluated
"""
import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Any


class ConfigDict(dict):
    """Dict that allows attribute access (cfg.key instead of cfg['key'])."""
    
    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
            if isinstance(value, dict):
                return ConfigDict(value)
            return value
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
    
    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(config_path: str | Path | None = None) -> ConfigDict:
    """Load configuration from YAML file."""
    if config_path is None:
        # Default to configs/default.yaml relative to repo root
        repo_root = Path(__file__).parent.parent
        config_path = repo_root / "configs" / "default.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return ConfigDict(config)


# Global config instance (lazy loaded)
_cfg: ConfigDict | None = None


def get_config() -> ConfigDict:
    """Get the global config instance."""
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


# Convenience alias
cfg = property(lambda self: get_config())


def set_seeds(config: ConfigDict | None = None) -> None:
    """Set random seeds from config for reproducibility."""
    import random
    import numpy as np
    
    if config is None:
        config = get_config()
    
    seeds = config.seeds
    random.seed(seeds.python)
    np.random.seed(seeds.numpy)
    
    # Note: Claude API is not deterministic even with temperature=0
    # We use temperature=0.2 to reduce variance but cannot guarantee
    # exact reproducibility across runs


if __name__ == "__main__":
    # Test config loading
    cfg = load_config()
    print(f"Loaded config with {len(cfg)} top-level keys")
    print(f"Languages: {cfg.languages.evaluated}")
    print(f"Model: {cfg.api.anthropic.model}")
    print(f"Seeds: numpy={cfg.seeds.numpy}, python={cfg.seeds.python}")
