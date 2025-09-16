# training_comparison

**Timestamp**: 2025-09-16 20:43:49
**Directory**: experiments/training_comparison_20250916_204349

## Description

    Side-by-side comparison of two training regimes for hierarchical monitoring:

    1. Curriculum Learning:
       - Stage 1: Train probe standalone (BCE loss)
       - Stage 2: Train gates with frozen probe (recall-cost loss)
       - Stage 3: Joint fine-tuning

    2. Joint Training:
       - Train probe and gates together from start (recall-cost loss)

    Dataset is much harder: 20% adversarial examples, weak signals, multiple patterns.
    

## Files Generated
- `config.json`: Experiment configuration parameters
- `results.json`: Final experiment results and metrics
- `*.png`: Generated plots and visualizations
- `*.pth`: Saved model weights
- `*.pkl`: Saved data objects

## Notes

    Side-by-side comparison of two training regimes for hierarchical monitoring:

    1. Curriculum Learning:
       - Stage 1: Train probe standalone (BCE loss)
       - Stage 2: Train gates with frozen probe (recall-cost loss)
       - Stage 3: Joint fine-tuning

    2. Joint Training:
       - Train probe and gates together from start (recall-cost loss)

    Dataset is much harder: 20% adversarial examples, weak signals, multiple patterns.
    
