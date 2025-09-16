# curriculum_learning

**Timestamp**: 2025-09-16 20:15:22
**Directory**: experiments/curriculum_learning_20250916_201522

## Description

    Curriculum learning experiment for hierarchical monitoring.

    Three-stage training process:
    1. Train probe for standalone classification (BCE loss)
    2. Train gate parameters with probe frozen (recall-cost loss)
    3. Joint fine-tuning of both probe and gate parameters

    Loss function: Recall = -P(audit|harmful), Cost = exp((cost/budget - 1)^2)
    

## Files Generated
- `config.json`: Experiment configuration parameters
- `results.json`: Final experiment results and metrics
- `*.png`: Generated plots and visualizations
- `*.pth`: Saved model weights
- `*.pkl`: Saved data objects

## Notes

    Curriculum learning experiment for hierarchical monitoring.

    Three-stage training process:
    1. Train probe for standalone classification (BCE loss)
    2. Train gate parameters with probe frozen (recall-cost loss)
    3. Joint fine-tuning of both probe and gate parameters

    Loss function: Recall = -P(audit|harmful), Cost = exp((cost/budget - 1)^2)
    
