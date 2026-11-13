"""Load config.yaml once and hand it around. Nothing else in the project hard-codes settings."""
import os
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path=None) -> dict:
    """Read the YAML config. Override the location with the HELPDESK_CONFIG env var."""
    path = Path(path or os.environ.get("HELPDESK_CONFIG") or DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} is empty or not a mapping")
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """Turn a relative path from the config (e.g. paths.corpus) into an absolute one."""
    return PROJECT_ROOT / cfg["paths"][key]
