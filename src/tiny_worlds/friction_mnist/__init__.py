"""Friction Moving MNIST: bouncing digits with damping and energy loss."""

from .dataset import FrictionMNIST, FrictionMNISTConfig, generate_dataset
from .viz import visualize_sample

__all__ = [
    "FrictionMNIST",
    "FrictionMNISTConfig",
    "generate_dataset",
    "visualize_sample",
]
