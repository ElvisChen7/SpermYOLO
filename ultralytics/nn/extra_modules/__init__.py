"""Minimal SpermYOLO extension registry."""

from .block import C3k2_IDB
from .DCMPNet import MFM
from .head import DESD_Head
from .transformer import D2SEM

__all__ = ("C3k2_IDB", "D2SEM", "MFM", "DESD_Head")
