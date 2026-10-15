"""CAGE toolkit API."""

from .recording import CAGEDataset, Recording
from .spatial import MapToOptiTrackTransform

__all__ = ["CAGEDataset", "Recording", "MapToOptiTrackTransform"]
__version__ = "0.1.0"
