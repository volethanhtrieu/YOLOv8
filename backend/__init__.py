"""Backend components for the PPE ablation project.

Heavy runtime modules (OpenCV/Ultralytics) are intentionally not imported here,
so configuration and unit tests remain usable before installing GPU packages.
"""

from .config import AppConfig, load_config

__all__ = ["AppConfig", "load_config"]
