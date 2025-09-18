#!/usr/bin/env python3
"""
Generate and save toy dataset for hierarchical monitoring experiments.

This creates the basic dataset with realistic probe-monitor correlations
that we developed for testing the modular curriculum learning system.
"""

import torch
import numpy as np
import json
import sys
import os
from sklearn.model_selection import train_test_split

# Add src to path to import training_comparison
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from training_comparison import generate_hard_dataset

def save_toy_dataset(n_samples=2000, n_features=500, harmful_ratio=0.2,
                     signal_strength=0.3, seed=42, dataset_name="toy_data"):
    """Generate and save toy dataset with realistic probe-monitor correlations using best practices."""

    print(f"Generating toy dataset with {n_samples} samples, {n_features} features...")

    # Generate the dataset
    features, labels, monitor_scores = generate_hard_dataset(
        n_samples=n_samples,
        n_features=n_features,
        harmful_ratio=harmful_ratio,
        signal_strength=signal_strength,
        seed=seed
    )

    # Create dataset directory
    dataset_dir = os.path.join(os.path.dirname(__file__), dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Split into train/test (70/30 split, stratified)
    X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
        features, labels, monitor_scores,
        test_size=0.3, random_state=42, stratify=labels
    )

    # Save data splits as PyTorch tensors (.pt format)
    torch.save(X_train, os.path.join(dataset_dir, 'train_features.pt'))
    torch.save(y_train, os.path.join(dataset_dir, 'train_labels.pt'))
    torch.save(m_train, os.path.join(dataset_dir, 'train_monitor_scores.pt'))

    torch.save(X_test, os.path.join(dataset_dir, 'test_features.pt'))
    torch.save(y_test, os.path.join(dataset_dir, 'test_labels.pt'))
    torch.save(m_test, os.path.join(dataset_dir, 'test_monitor_scores.pt'))

    # Save metadata as JSON
    metadata = {
        'dataset_name': dataset_name,
        'description': 'Toy dataset with realistic probe-monitor correlations for hierarchical monitoring',
        'version': '1.0',
        'creation_date': str(torch.get_default_dtype()),
        'data_splits': {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_harmful': int(y_train.sum().item()),
            'test_harmful': int(y_test.sum().item())
        },
        'parameters': {
            'n_samples': n_samples,
            'n_features': n_features,
            'harmful_ratio': harmful_ratio,
            'signal_strength': signal_strength,
            'seed': seed
        },
        'correlation_categories': {
            '0': 'Poor probe, good monitor (10%)',
            '1': 'Poor monitor, good probe (10%)',
            '2': 'Low correlation (30%)',
            '3': 'High correlation (50%)'
        },
        'file_formats': {
            'features': 'PyTorch tensor (.pt) - shape: [n_samples, n_features]',
            'labels': 'PyTorch tensor (.pt) - shape: [n_samples] - 0=benign, 1=harmful',
            'monitor_scores': 'PyTorch tensor (.pt) - shape: [n_samples] - continuous scores'
        }
    }

    with open(os.path.join(dataset_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create README
    readme_content = f"""# {dataset_name.title().replace('_', ' ')} Dataset

## Description
{metadata['description']}

## Dataset Statistics
- **Total samples**: {n_samples:,}
- **Features**: {n_features:,}
- **Harmful ratio**: {harmful_ratio:.1%}
- **Train/Test split**: 70/30

### Split Details
- **Training**: {len(X_train):,} samples ({int(y_train.sum().item())} harmful, {len(X_train) - int(y_train.sum().item())} benign)
- **Testing**: {len(X_test):,} samples ({int(y_test.sum().item())} harmful, {len(X_test) - int(y_test.sum().item())} benign)

## Correlation Structure
The dataset includes four categories of probe-monitor correlation patterns:
- **10% Poor probe, good monitor**: Monitor detects well, probe struggles
- **10% Poor monitor, good probe**: Probe detects well, monitor struggles
- **30% Low correlation**: Both systems see different aspects
- **50% High correlation**: Both systems correlated with underlying harmfulness

## Files
- `train_features.pt`: Training features [n_train, {n_features}]
- `train_labels.pt`: Training labels [n_train]
- `train_monitor_scores.pt`: Training monitor scores [n_train]
- `test_features.pt`: Test features [n_test, {n_features}]
- `test_labels.pt`: Test labels [n_test]
- `test_monitor_scores.pt`: Test monitor scores [n_test]
- `metadata.json`: Dataset metadata and parameters

## Usage
```python
import torch

# Load training data
X_train = torch.load('train_features.pt')
y_train = torch.load('train_labels.pt')
m_train = torch.load('train_monitor_scores.pt')

# Load test data
X_test = torch.load('test_features.pt')
y_test = torch.load('test_labels.pt')
m_test = torch.load('test_monitor_scores.pt')
```
"""

    with open(os.path.join(dataset_dir, 'README.md'), 'w') as f:
        f.write(readme_content)

    print(f"Dataset saved to {dataset_dir}/")
    print(f"- Train: {len(X_train)} samples ({int(y_train.sum().item())} harmful)")
    print(f"- Test: {len(X_test)} samples ({int(y_test.sum().item())} harmful)")
    print(f"- Features: {n_features}")
    print(f"- Files: train/test splits + metadata.json + README.md")

    return dataset_dir

if __name__ == "__main__":
    save_toy_dataset()