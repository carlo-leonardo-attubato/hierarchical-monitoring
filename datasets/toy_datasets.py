#!/usr/bin/env python3
"""
Toy dataset generation for hierarchical monitoring experiments.

This module contains functions to generate synthetic datasets with controlled
probe-monitor relationships and causal structures for testing curriculum learning
approaches.
"""

import os
import torch
import numpy as np
from typing import Tuple, Optional
import json


def generate_hierarchical_monitoring_dataset(
    n_samples: int = 2000,
    n_features: int = 32,
    harmful_ratio: float = 0.2,
    shared_noise_std: float = 0.8,
    probe_noise_std: float = 1.5,
    monitor_noise_std: float = 0.3,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate hierarchical monitoring dataset with controlled probe-monitor relationship.

    This implements a causal model where:
    - y = +1 for harmful, -1 for benign
    - shared_noise ~ N(0, shared_noise_std) creates correlation between probe and monitor
    - probe_suspiciousness = signal + shared_noise + independent_noise(probe_noise_std)
    - monitor_suspiciousness = signal + shared_noise + independent_noise(monitor_noise_std)
    - features are based on noisy probe scores (realistic scenario where probe learns from features)

    Args:
        n_samples: Number of samples to generate
        n_features: Dimensionality of feature vectors
        harmful_ratio: Fraction of samples that are harmful
        shared_noise_std: Standard deviation of shared noise component (controls correlation)
        probe_noise_std: Standard deviation of probe-specific noise (higher = worse probe)
        monitor_noise_std: Standard deviation of monitor-specific noise (lower = better monitor)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (features, labels, monitor_scores) where:
        - features: [n_samples, n_features] input features for probe training
        - labels: [n_samples] binary labels (0=benign, 1=harmful)
        - monitor_scores: [n_samples] monitor suspiciousness scores
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_harmful = int(n_samples * harmful_ratio)
    n_benign = n_samples - n_harmful

    # Generate binary signal: +1 for harmful, -1 for benign
    y = -torch.ones(n_samples)  # Start with -1 for benign
    y[:n_harmful] = 1  # First n_harmful samples are harmful

    # Convert to 0/1 labels for compatibility with existing code
    labels = torch.zeros(n_samples)
    labels[:n_harmful] = 1  # 1 for harmful, 0 for benign

    # Generate shared noise component (creates probe-monitor correlation)
    shared_noise = torch.randn(n_samples) * shared_noise_std

    # Generate probe suspiciousness (worse performance due to higher noise)
    probe_suspiciousness = 1.0 * y + shared_noise + torch.randn(n_samples) * probe_noise_std

    # Generate monitor suspiciousness (better performance due to lower noise)
    monitor_suspiciousness = 1.0 * y + shared_noise + torch.randn(n_samples) * monitor_noise_std

    # Monitor scores are the monitor suspiciousness values
    monitor_scores = monitor_suspiciousness

    # Generate fixed random direction vector (same across all samples)
    signal_direction = torch.randn(n_features)
    signal_direction = signal_direction / torch.norm(signal_direction)  # normalize to unit vector

    # Generate feature vectors based on noisy probe scores
    # This is realistic: probe learns from features, features contain probe-level signal
    features = torch.zeros(n_samples, n_features)
    for i in range(n_samples):
        # Random unit vector for baseline features
        random_unit = torch.randn(n_features)
        random_unit = random_unit / torch.norm(random_unit)

        # Feature vector = random baseline + probe_signal * direction
        features[i] = random_unit + probe_suspiciousness[i] * signal_direction

    # Shuffle the dataset to remove ordering effects
    indices = torch.randperm(n_samples)
    features = features[indices]
    labels = labels[indices]
    monitor_scores = monitor_scores[indices]

    return features, labels, monitor_scores


def save_dataset(
    features: torch.Tensor,
    labels: torch.Tensor,
    monitor_scores: torch.Tensor,
    save_dir: str = "datasets/toy",
    dataset_name: str = "hierarchical_monitoring",
    config: Optional[dict] = None
) -> str:
    """
    Save a generated dataset to disk with metadata.

    Args:
        features: Feature tensor [n_samples, n_features]
        labels: Label tensor [n_samples]
        monitor_scores: Monitor score tensor [n_samples]
        save_dir: Directory to save dataset
        dataset_name: Base name for dataset files
        config: Configuration dictionary to save as metadata

    Returns:
        Path to saved dataset directory
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create timestamped dataset directory
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = os.path.join(save_dir, f"{dataset_name}_{timestamp}")
    os.makedirs(dataset_dir, exist_ok=True)

    # Save tensors
    torch.save(features, os.path.join(dataset_dir, "features.pt"))
    torch.save(labels, os.path.join(dataset_dir, "labels.pt"))
    torch.save(monitor_scores, os.path.join(dataset_dir, "monitor_scores.pt"))

    # Save metadata
    metadata = {
        "dataset_name": dataset_name,
        "timestamp": timestamp,
        "n_samples": features.shape[0],
        "n_features": features.shape[1],
        "harmful_ratio": float(labels.mean()),
        "config": config or {}
    }

    with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset saved to: {dataset_dir}")
    return dataset_dir


def load_dataset(dataset_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Load a saved dataset from disk.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Tuple of (features, labels, monitor_scores, metadata)
    """
    features = torch.load(os.path.join(dataset_path, "features.pt"))
    labels = torch.load(os.path.join(dataset_path, "labels.pt"))
    monitor_scores = torch.load(os.path.join(dataset_path, "monitor_scores.pt"))

    with open(os.path.join(dataset_path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    return features, labels, monitor_scores, metadata


def generate_and_save_dataset(
    save_dir: str = "datasets/toy",
    dataset_name: str = "hierarchical_monitoring",
    **kwargs
) -> str:
    """
    Generate and save a hierarchical monitoring dataset in one step.

    Args:
        save_dir: Directory to save dataset
        dataset_name: Base name for dataset files
        **kwargs: Arguments to pass to generate_hierarchical_monitoring_dataset

    Returns:
        Path to saved dataset directory
    """
    # Generate dataset
    features, labels, monitor_scores = generate_hierarchical_monitoring_dataset(**kwargs)

    # Save with configuration
    config = kwargs.copy()
    dataset_path = save_dataset(
        features, labels, monitor_scores,
        save_dir=save_dir,
        dataset_name=dataset_name,
        config=config
    )

    return dataset_path


# Backward compatibility alias
def generate_hard_dataset(n_samples=800, n_features=32, harmful_ratio=0.2, signal_strength=0.3, seed=42):
    """
    Backward compatibility wrapper for existing code.

    This maintains the same interface as the original function in training_comparison.py
    but uses the new parameter names internally.
    """
    return generate_hierarchical_monitoring_dataset(
        n_samples=n_samples,
        n_features=n_features,
        harmful_ratio=harmful_ratio,
        shared_noise_std=0.8,
        probe_noise_std=1.5,
        monitor_noise_std=0.3,
        seed=seed
    )