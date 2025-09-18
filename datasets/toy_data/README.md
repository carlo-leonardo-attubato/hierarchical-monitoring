# Toy Data Dataset

## Description
Toy dataset with realistic probe-monitor correlations for hierarchical monitoring

## Dataset Statistics
- **Total samples**: 1,000
- **Features**: 500
- **Harmful ratio**: 20.0%
- **Train/Test split**: 70/30

### Split Details
- **Training**: 700 samples (140 harmful, 560 benign)
- **Testing**: 300 samples (60 harmful, 240 benign)

## Correlation Structure
The dataset includes four categories of probe-monitor correlation patterns:
- **10% Poor probe, good monitor**: Monitor detects well, probe struggles
- **10% Poor monitor, good probe**: Probe detects well, monitor struggles
- **30% Low correlation**: Both systems see different aspects
- **50% High correlation**: Both systems correlated with underlying harmfulness

## Files
- `train_features.pt`: Training features [n_train, 500]
- `train_labels.pt`: Training labels [n_train]
- `train_monitor_scores.pt`: Training monitor scores [n_train]
- `test_features.pt`: Test features [n_test, 500]
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
