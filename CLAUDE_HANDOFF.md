# Claude Handoff Instructions

## Context
This is a hierarchical monitoring project implementing soft-gating architecture for AI safety monitoring. The goal is to train probes that optimize for hierarchical monitoring performance rather than standalone classification accuracy.

## Current State
- **Codebase**: Complete implementation of hierarchical monitoring with soft gates
- **Key files**:
  - `src/models/hierarchical.py` - Main hierarchical monitoring system
  - `src/models/soft_gates.py` - Soft-gating architecture
  - `src/training/loss.py` - Loss functions for hierarchical training
  - `toy_experiment.py` - Complete experiment script (READY TO RUN)

## Immediate Task
Run the toy experiment to validate the hierarchical monitoring approach:

1. **Execute**: `python toy_experiment.py`
2. **Expected outcome**: Comparison showing hierarchical training achieves higher recall than baseline approach at same budget
3. **Key metrics**: Recall on harmful outputs at different budget levels ($0.5, $1.0, $2.0, $3.0, $5.0)

## Experiment Details
- **Dataset**: 100 samples, 8192-dimensional features, 10% harmful rate
- **Comparison**:
  - Hierarchical training (end-to-end with soft gates)
  - Baseline probe (trained for standalone accuracy) + hierarchical deployment
- **Costs**: Monitor=$1, Audit=$20
- **Output**: Results table + plots saved as `toy_experiment_results.png`

## Expected Results
The hierarchical approach should show improved recall (e.g., 5-15% better) at the same budget, validating that training for the full system is better than training for standalone accuracy.

## Dependencies
All required packages are in `requirements.txt`. Virtual environment in `venv/` folder.

## Next Steps After Experiment
1. Analyze results and identify any issues
2. Tune hyperparameters if needed
3. Scale to larger/realistic datasets
4. Compare different loss function combinations

## Project Structure
```
src/
├── models/          # Core models (probe, gates, hierarchical)
├── training/        # Loss functions and training utilities
├── data/           # Data loading (unused for toy experiment)
└── evaluation/     # Metrics and visualization
```

## Key Insight
Traditional probe training optimizes for classification accuracy, but in hierarchical monitoring, the probe's job is to identify cases where expensive monitoring adds value. Our approach optimizes for this directly.