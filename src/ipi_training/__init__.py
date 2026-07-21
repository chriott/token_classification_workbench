"""Training package for IPI token classification models."""

from .config import TrainingConfig
from .training import run_pipeline

__all__ = ["TrainingConfig", "run_pipeline"]
