"""
Complete example demonstrating hierarchical monitoring.

This script shows how to:
1. Create synthetic data
2. Train a hierarchical monitoring system
3. Evaluate performance
4. Deploy the system
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
import os

# Import from the installed package
from src.models.hierarchical import HierarchicalMonitor
from src.models.probe import LinearProbe
from src.training.trainer import HierarchicalTrainer, BaselineTrainer
from src.training.loss import HierarchicalLoss, BaselineLoss
from src.data.datasets import SyntheticDataset, create_data_loaders, split_dataset
from src.evaluation.metrics import HierarchicalMetrics, DeploymentMetrics


def create_synthetic_data(n_samples=1000, n_features=128, harmful_ratio=0.1, seed=42):
    """Create synthetic dataset for demonstration."""
    print(f"Creating synthetic dataset with {n_samples} samples...")
    
    dataset = SyntheticDataset(
        n_samples=n_samples,
        n_features=n_features,
        harmful_ratio=harmful_ratio,
        seed=seed
    )
    
    print(f"Dataset stats: {dataset.get_stats()}")
    return dataset


def train_hierarchical_model(train_loader, val_loader, device="cpu"):
    """Train hierarchical monitoring model."""
    print("\nTraining hierarchical monitoring model...")
    
    # Create model
    model = HierarchicalMonitor(
        input_dim=128,
        monitor_cost=1.0,
        audit_cost=20.0,
        budget=2.0
    )
    
    # Create loss function
    loss_fn = HierarchicalLoss(
        recall_loss_type="log_likelihood",
        cost_penalty_type="quadratic",
        lambda_cost=1.0
    )
    
    # Create trainer
    trainer = HierarchicalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device
    )
    
    # Train
    history = trainer.train(epochs=50, save_best=False)
    
    print(f"Training completed. Final train loss: {history['train_losses'][-1]:.4f}")
    if history['val_losses']:
        print(f"Final validation loss: {history['val_losses'][-1]:.4f}")
    
    return model, trainer


def train_baseline_model(train_loader, val_loader, device="cpu"):
    """Train baseline model for comparison."""
    print("\nTraining baseline model...")
    
    # Create baseline model (just the probe)
    model = LinearProbe(input_dim=128, output_dim=1)
    
    # Create baseline trainer
    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Train
    history = trainer.train(epochs=50)
    
    print(f"Baseline training completed. Final loss: {history[-1]:.4f}")
    
    return model, trainer


def evaluate_model(model, test_loader, device="cpu"):
    """Evaluate model performance."""
    print("\nEvaluating model performance...")
    
    model.eval()
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            monitor_scores = batch.get('monitor_scores', None)
            if monitor_scores is not None:
                monitor_scores = monitor_scores.to(device)
            
            # Forward pass
            if hasattr(model, 'deploy'):  # Hierarchical model
                outputs = model(features, monitor_scores, training=True)
                all_outputs.append(outputs)
            else:  # Baseline model
                probe_scores = model(features).squeeze(-1)
                outputs = {'p_audit': torch.sigmoid(probe_scores)}
                all_outputs.append(outputs)
            
            all_labels.append(labels)
    
    # Concatenate results
    labels = torch.cat(all_labels)
    outputs = {
        'p_audit': torch.cat([o['p_audit'] for o in all_outputs]),
        'p_gate': torch.cat([o.get('p_gate', torch.zeros_like(o['p_audit'])) for o in all_outputs]),
        'expected_total_cost': torch.cat([o.get('expected_total_cost', torch.zeros_like(o['p_audit'])) for o in all_outputs]),
        'p1': torch.cat([o.get('p1', torch.zeros_like(o['p_audit'])) for o in all_outputs]),
        'p2': torch.cat([o.get('p2', torch.zeros_like(o['p_audit'])) for o in all_outputs])
    }
    
    # Compute metrics
    metrics = HierarchicalMetrics().compute_metrics(outputs, labels, budget=2.0)
    
    print("Evaluation metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return metrics


def deploy_model(model, test_loader, device="cpu"):
    """Deploy model and evaluate real-world performance."""
    print("\nDeploying model...")
    
    model.eval()
    all_results = []
    all_labels = []
    
    # Mock monitor function
    def mock_monitor_fn(features):
        # Simulate expensive monitor that returns random scores
        return torch.randn(features.shape[0])
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            # Deploy model
            if hasattr(model, 'deploy'):
                # Hierarchical model
                results = model.deploy(features, mock_monitor_fn, sample_decisions=False)
            else:
                # Baseline model - simulate deployment
                probe_scores = model(features).squeeze(-1)
                audit_decisions = (torch.sigmoid(probe_scores) > 0.5).float()
                results = {
                    'audit_decisions': audit_decisions,
                    'monitor_calls': torch.zeros_like(audit_decisions),
                    'total_costs': audit_decisions * 20.0,  # Only audit cost
                    'immediate_audits': audit_decisions.sum().item(),
                    'escalations': 0,
                    'final_audits': 0
                }
            
            all_results.append(results)
            all_labels.append(labels)
    
    # Concatenate results
    labels = torch.cat(all_labels)
    combined_results = {
        'audit_decisions': torch.cat([r['audit_decisions'] for r in all_results]),
        'monitor_calls': torch.cat([r['monitor_calls'] for r in all_results]),
        'total_costs': torch.cat([r['total_costs'] for r in all_results]),
        'immediate_audits': sum(r['immediate_audits'] for r in all_results),
        'escalations': sum(r['escalations'] for r in all_results),
        'final_audits': sum(r['final_audits'] for r in all_results)
    }
    
    # Compute deployment metrics
    metrics = DeploymentMetrics().compute_deployment_metrics(combined_results, labels)
    
    print("Deployment metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return metrics


def plot_training_history(hierarchical_history, baseline_history=None):
    """Plot training history."""
    plt.figure(figsize=(12, 4))
    
    # Plot hierarchical training
    plt.subplot(1, 2, 1)
    plt.plot(hierarchical_history['train_losses'], label='Train Loss')
    if hierarchical_history['val_losses']:
        plt.plot(hierarchical_history['val_losses'], label='Val Loss')
    plt.title('Hierarchical Model Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot baseline training
    if baseline_history is not None:
        plt.subplot(1, 2, 2)
        plt.plot(baseline_history, label='Baseline Loss')
        plt.title('Baseline Model Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main function demonstrating the complete pipeline."""
    print("=== Hierarchical Monitoring Complete Example ===\n")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create synthetic data
    dataset = create_synthetic_data(n_samples=2000, n_features=128, harmful_ratio=0.15)
    
    # Split data
    train_data, val_data, test_data = split_dataset(
        dataset, 
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, batch_size=32
    )
    
    # Train hierarchical model
    hierarchical_model, hierarchical_trainer = train_hierarchical_model(
        train_loader, val_loader, device
    )
    
    # Train baseline model
    baseline_model, baseline_trainer = train_baseline_model(
        train_loader, val_loader, device
    )
    
    # Evaluate models
    print("\n" + "="*50)
    print("HIERARCHICAL MODEL EVALUATION")
    print("="*50)
    hierarchical_metrics = evaluate_model(hierarchical_model, test_loader, device)
    
    print("\n" + "="*50)
    print("BASELINE MODEL EVALUATION")
    print("="*50)
    baseline_metrics = evaluate_model(baseline_model, test_loader, device)
    
    # Deploy models
    print("\n" + "="*50)
    print("HIERARCHICAL MODEL DEPLOYMENT")
    print("="*50)
    hierarchical_deploy_metrics = deploy_model(hierarchical_model, test_loader, device)
    
    print("\n" + "="*50)
    print("BASELINE MODEL DEPLOYMENT")
    print("="*50)
    baseline_deploy_metrics = deploy_model(baseline_model, test_loader, device)
    
    # Compare results
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(f"Hierarchical Model Recall: {hierarchical_metrics['recall']:.4f}")
    print(f"Baseline Model Recall: {baseline_metrics['recall']:.4f}")
    print(f"Improvement: {hierarchical_metrics['recall'] - baseline_metrics['recall']:.4f}")
    
    print(f"\nHierarchical Model Mean Cost: {hierarchical_metrics['mean_cost']:.4f}")
    print(f"Baseline Model Mean Cost: {baseline_metrics['mean_cost']:.4f}")
    print(f"Cost Difference: {hierarchical_metrics['mean_cost'] - baseline_metrics['mean_cost']:.4f}")
    
    # Plot training history
    hierarchical_history = {
        'train_losses': hierarchical_trainer.train_losses,
        'val_losses': hierarchical_trainer.val_losses
    }
    baseline_history = baseline_trainer.train_losses if hasattr(baseline_trainer, 'train_losses') else None
    
    plot_training_history(hierarchical_history, baseline_history)
    
    print("\n=== Example completed successfully! ===")
    print("Check 'training_history.png' for training curves.")


if __name__ == "__main__":
    main()
