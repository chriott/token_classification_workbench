"""Tools for training token-classification models from span annotations."""

from .config import TrainingConfig
from .training import run_pipeline

__all__ = ["TrainingConfig", "run_pipeline"]
