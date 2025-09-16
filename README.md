# Hierarchical Monitoring with Soft Gates

This project implements the soft-gating architecture for hierarchical monitoring systems as described in the paper "Training Probes for Hierarchical Monitoring".

## Overview

The system uses linear probes on internal activations as first-stage monitors, with probabilistic gates that decide when to escalate to expensive monitors (LLMs) or directly audit outputs. All components are differentiable, enabling end-to-end optimization.

## Key Features

- **Soft-gating architecture**: Probabilistic decisions instead of hard thresholds
- **End-to-end optimization**: Joint training of probe and gating parameters
- **Budget-aware loss**: Balances recall on harmful outputs with cost constraints
- **Differentiable components**: Gradient-based optimization throughout

## Project Structure

```
hierarchical_monitoring/
├── src/
│   ├── models/
│   │   ├── probe.py          # Linear probe implementation
│   │   ├── soft_gates.py     # Soft-gating architecture
│   │   └── hierarchical.py   # Complete hierarchical system
│   ├── training/
│   │   ├── loss.py           # Loss functions
│   │   ├── trainer.py        # Training loop
│   │   └── optimizer.py      # Optimization utilities
│   ├── data/
│   │   ├── datasets.py       # Data loading utilities
│   │   └── preprocessing.py  # Data preprocessing
│   ├── evaluation/
│   │   ├── metrics.py        # Evaluation metrics
│   │   └── visualization.py  # Plotting utilities
│   └── utils/
│       ├── config.py         # Configuration management
│       └── logging.py         # Logging utilities
├── experiments/
│   ├── deception_detection/  # Deception detection experiments
│   └── synthetic/           # Synthetic data experiments
├── notebooks/
│   └── demo.ipynb           # Interactive demo
├── tests/
│   └── test_*.py            # Unit tests
├── requirements.txt
└── setup.py
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from src.models.hierarchical import HierarchicalMonitor
from src.training.trainer import Trainer
from src.data.datasets import DeceptionDataset

# Load data
dataset = DeceptionDataset('path/to/data')

# Initialize model
model = HierarchicalMonitor(
    input_dim=128,
    monitor_cost=1.0,
    audit_cost=20.0,
    budget=2.0
)

# Train
trainer = Trainer(model, dataset)
trainer.train(epochs=100, lr=0.001)
```

## Experiments

Run deception detection experiments:
```bash
python experiments/deception_detection/train.py --config configs/deception.yaml
```

## Citation

If you use this code, please cite:
```bibtex
@article{attubato2025hierarchical,
  title={Training Probes for Hierarchical Monitoring},
  author={Attubato, Carlo Leonardo},
  journal={arXiv preprint},
  year={2025}
}
```
