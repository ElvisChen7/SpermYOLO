"""Detection-only YOLO model family."""

from . import detect
from .model import YOLO

__all__ = ("detect", "YOLO")
