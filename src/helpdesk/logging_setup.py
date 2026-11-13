"""One place to configure logging so every module logs the same way."""
import logging
from .config import PROJECT_ROOT


def setup_logging(cfg: dict) -> logging.Logger:
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = PROJECT_ROOT / log_cfg.get("file", "logs/helpdesk.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("helpdesk")
    if not root.handlers:                      # avoid duplicate handlers on re-run
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        for h in (logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()):
            h.setFormatter(fmt)
            root.addHandler(h)
    root.setLevel(level)
    return root
