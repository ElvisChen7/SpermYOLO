"""SpermYOLO: a minimal, detection-only Ultralytics derivative."""

import os

__version__ = "8.3.9-spermyolo"

if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"

from ultralytics.models import YOLO
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks

settings = SETTINGS
__all__ = ("__version__", "ASSETS", "YOLO", "checks", "settings")
