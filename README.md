# Hierarchical Monitoring with Curriculum Learning

This repository implements hierarchical monitoring systems with three-phase curriculum learning for AI safety applications. The approach trains models to make optimal decisions about when to escalate from cheaper probe-level monitoring to more expensive monitor-level auditing.

## Key Features

- **Three-Phase Curriculum Learning**: Progressive training from probe-only → joint training → gate optimization
- **Modular Loss Functions**: 29 combinations of 5 recall × 4 cost loss functions + exponential variants
- **Comprehensive Parameter Sweep**: Tests across 6 feature dimensions (20, 50, 100, 200, 500, 1000)
- **Enforced Budget Evaluation**: Focus on recall performance at fixed $1.00 budget per benign example
- **Realistic Probe-Monitor Relationships**: Configurable correlation structures for experimental control

## Quick Start

The comprehensive sweep has been completed. View results:
```bash
# View final heatmap comparison
open plots/heatmaps_budget_recall_1758163294.png

# Run individual experiments (scripts in graveyard/)
python graveyard/modular_curriculum.py
```

## Comprehensive Sweep Results

**Peak Performance:**
- **Joint (Three-Phase)**: 27.0% recall at $1 budget (focal + absolute, 100 features)
- **Baseline (Probe-Only)**: 3.7% recall at $1 budget
- **Curriculum Advantage**: +21.2% absolute improvement

**Total Experiments**: 174 configurations (29 loss combinations × 6 feature dimensions)

## Repository Structure

```
hierarchical_monitoring/
├── src/                    # Core implementation modules
│   ├── models/            # Neural network architectures (HierarchicalMonitoring, etc.)
│   ├── training/          # Loss functions and training utilities
│   ├── evaluation/        # Metrics and evaluation tools
│   └── utils/             # Experiment utilities and helpers
├── datasets/              # Dataset generation and management
├── experiments/           # Previous experiment configurations and results
├── plots/                 # Final visualizations and results
│   └── heatmaps_budget_recall_1758163294.png  # Main result: comprehensive sweep
├── graveyard/             # Complete experimental codebase
│   ├── modular_curriculum.py      # Main 29-loss implementation
│   ├── comprehensive_sweep.py     # Full 174-experiment sweep
│   ├── evaluate_budget_recall.py  # Budget recall evaluation
│   ├── heatmaps_only.py          # Final visualization script
│   └── [50+ other experimental scripts]
└── venv/                  # Virtual environment
```

## Core Concepts

### Three-Phase Training
1. **Phase 1**: Train shared probe network on classification task
2. **Phase 2**: Joint training of probe + hierarchical gates with recall-cost loss
3. **Phase 3**: Fine-tune gates while freezing probe weights

### Modular Loss Functions
- **Recall losses**: log_likelihood, linear, quadratic, exponential, focal (5 total)
- **Cost losses**: quadratic, absolute, exponential_quadratic, huber (4 total)
- **Base combinations**: 5×4 = 20 configurations
- **Exponential variants**: 9 additional combinations with varying decay rates
- **Total**: 29 loss function combinations tested
- **Focus**: Recall performance at enforced $1.00 budget constraint

### Realistic Data Model
```python
# Causal structure ensures realistic probe-monitor relationships
y = +1 for harmful, -1 for benign
shared_noise = N(0, 0.8)  # Creates correlation
probe_suspiciousness = y + shared_noise + N(0, 1.5)  # Noisier (worse performance)
monitor_suspiciousness = y + shared_noise + N(0, 0.3)  # Better performance
features = random_baseline + probe_suspiciousness * signal_direction
```

## Key Results

The comprehensive 174-experiment sweep demonstrates:
- **Best Joint Performance**: focal + absolute at 100 features (27.0% recall at $1 budget)
- **Best Baseline Performance**: quadratic + absolute at 1000 features (5.8% recall at $1 budget)
- **Curriculum Learning Advantage**: Consistent +15-25% absolute improvement across all configurations
- **Optimal Feature Dimension**: 100 features provides best curriculum learning gains
- **Top Loss Combinations**: focal + absolute, log_likelihood + absolute show strongest performance

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

**Comprehensive sweep completed** - view results in `plots/heatmaps_budget_recall_1758163294.png`

- **Run experiments**: Scripts available in `graveyard/` directory
- **Core implementation**: Modular components in `src/` for extending research
- **Previous work**: Historical experiments preserved in `experiments/`

The comprehensive 174-experiment parameter sweep demonstrates that three-phase curriculum learning consistently outperforms probe-only baselines, with peak performance achieving 27.0% recall at enforced budget constraints.

## Core Architecture

### Soft-Gating System
- **Differentiable gates**: Probabilistic decisions instead of hard thresholds
- **End-to-end optimization**: Joint training of probe and gating parameters
- **Budget-aware loss**: Balances recall on harmful outputs with cost constraints

### Models
- `HierarchicalMonitoring`: Complete hierarchical system with soft gates
- Modular loss functions with configurable recall and cost penalties
- Realistic causal data generation with controlled probe-monitor correlations

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