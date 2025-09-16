"""
Toy experiment to derisk hierarchical monitoring approach.

This script creates a toy dataset with 100 8192-dimensional features and compares:
1. Hierarchical monitoring with soft gates (end-to-end training)
2. Baseline probe trained for standalone classification + hierarchical deployment

The goal is to validate that the hierarchical training approach improves recall
at the same budget compared to the standalone approach.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import random
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# Import our modules
from src.models.hierarchical import HierarchicalMonitor
from src.models.probe import LinearProbe
from experiment_utils import ExperimentLogger
from src.training.loss import HierarchicalLoss, BaselineLoss
from src.models.soft_gates import HierarchicalGates


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def generate_toy_dataset(
    n_samples: int = 100,
    n_features: int = 8192,
    harmful_ratio: float = 0.1,
    signal_strength: float = 0.3,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate toy dataset with realistic characteristics.

    Args:
        n_samples: Total number of samples
        n_features: Dimension of features
        harmful_ratio: Fraction of harmful examples
        signal_strength: How separable harmful/benign are
        seed: Random seed

    Returns:
        Tuple of (features, labels, monitor_scores)
    """
    set_seed(seed)

    n_harmful = int(n_samples * harmful_ratio)
    n_benign = n_samples - n_harmful

    # Generate base features from normal distribution
    features = torch.randn(n_samples, n_features)

    # Create labels
    labels = torch.zeros(n_samples)
    labels[:n_harmful] = 1  # First n_harmful are harmful

    # Add signal to harmful examples
    # Create a random "harmful direction" in feature space
    harmful_direction = torch.randn(n_features)
    harmful_direction = harmful_direction / torch.norm(harmful_direction)

    # First add signal to harmful examples normally
    for i in range(n_harmful):
        noise_scale = np.random.uniform(0.5, 1.5)
        features[i] += signal_strength * noise_scale * harmful_direction
        random_dims = torch.randperm(n_features)[:100]
        features[i, random_dims] += torch.randn(100) * signal_strength * 0.5

    # Then make 10% of ALL samples have completely random features
    random_feature_fraction = 0.1  # 10% of ALL examples get completely random features
    n_random_features = int(n_samples * random_feature_fraction)
    random_indices = torch.randperm(n_samples)[:n_random_features]

    for i in random_indices:
        # Completely random features (overwrite any signal that was there)
        features[i] = torch.randn(n_features)

    # Shuffle the dataset
    indices = torch.randperm(n_samples)
    features = features[indices]
    labels = labels[indices]

    # Track which examples had random features after shuffling
    was_random = torch.zeros(n_samples, dtype=bool)
    for i, orig_idx in enumerate(indices):
        was_random[i] = orig_idx in random_indices

    # Generate expensive monitor scores (good accuracy even on random features)
    monitor_scores = torch.randn(n_samples) * 0.4
    for i in range(n_samples):
        if labels[i] == 1:  # Harmful
            if was_random[i]:
                # Monitor has good accuracy even on random features
                monitor_scores[i] += 1.0 + torch.randn(1).item() * 0.3
            else:
                # Normal harmful examples
                monitor_scores[i] += 0.7 + torch.randn(1).item() * 0.4
        else:  # Benign
            monitor_scores[i] += torch.randn(1).item() * 0.2

    return features, labels, monitor_scores


def create_expensive_monitor(monitor_scores: torch.Tensor) -> callable:
    """Create expensive monitor function that returns precomputed scores."""
    def monitor_fn(features: torch.Tensor) -> torch.Tensor:
        # In a real setting, this would be expensive (e.g., GPT-4 call)
        # Here we just return the precomputed scores
        batch_indices = torch.arange(len(features))  # This is simplified
        return monitor_scores[batch_indices]
    return monitor_fn


def train_hierarchical_model(
    features: torch.Tensor,
    labels: torch.Tensor,
    monitor_scores: torch.Tensor,
    monitor_cost: float = 1.0,
    audit_cost: float = 20.0,
    budget: float = 2.0,
    epochs: int = 200,
    lr: float = 0.01
) -> HierarchicalMonitor:
    """Train hierarchical model with soft gates end-to-end."""

    n_samples, n_features = features.shape

    # Create model
    model = HierarchicalMonitor(
        input_dim=n_features,
        monitor_cost=monitor_cost,
        audit_cost=audit_cost,
        budget=budget
    )

    # Create loss function
    loss_fn = HierarchicalLoss(
        recall_loss_type="log_likelihood",
        cost_penalty_type="quadratic",
        lambda_cost=1.0
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    train_losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features, monitor_scores, training=True)

        # Compute loss
        losses = loss_fn(outputs, labels, budget)
        total_loss = losses['total_loss']

        # Backward pass
        total_loss.backward()
        optimizer.step()

        train_losses.append({
            'total': total_loss.item(),
            'recall': losses['recall_loss'].item(),
            'cost': losses['cost_loss'].item()
        })

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Total={total_loss.item():.4f}, "
                  f"Recall={losses['recall_loss'].item():.4f}, "
                  f"Cost={losses['cost_loss'].item():.4f}")

    return model, train_losses


def train_baseline_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.01
) -> LinearProbe:
    """Train baseline probe for standalone classification accuracy."""

    n_samples, n_features = features.shape

    # Create probe
    probe = LinearProbe(n_features)

    # Create loss function
    loss_fn = BaselineLoss("bce")

    # Optimizer
    optimizer = optim.Adam(probe.parameters(), lr=lr)

    # Training loop
    probe.train()
    train_losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        probe_scores = probe(features).squeeze(-1)

        # Compute loss
        loss = loss_fn(probe_scores, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: BCE Loss={loss.item():.4f}")

    return probe, train_losses


def evaluate_hierarchical_system(
    model: HierarchicalMonitor,
    features: torch.Tensor,
    labels: torch.Tensor,
    monitor_scores: torch.Tensor,
    budget_levels: List[float]
) -> Dict[float, Dict[str, float]]:
    """Evaluate hierarchical system at different budget levels."""

    model.eval()
    results = {}

    with torch.no_grad():
        for budget in budget_levels:
            # Temporarily set budget
            original_budget = model.budget
            model.budget = budget

            # Get model outputs
            outputs = model(features, monitor_scores, training=True)

            # Calculate metrics
            p_audit = outputs['p_audit']
            expected_costs = outputs['expected_total_cost']

            # Recall on harmful examples (fraction that get audited)
            harmful_mask = (labels == 1)
            if harmful_mask.any():
                # Count how many harmful examples get audited (p_audit > 0.5 threshold)
                audited_harmful = (p_audit[harmful_mask] > 0.5).sum().item()
                total_harmful = harmful_mask.sum().item()
                recall = audited_harmful / total_harmful
                harmful_count = total_harmful
            else:
                recall = 0.0
                harmful_count = 0

            # Average cost on benign examples
            benign_mask = (labels == 0)
            if benign_mask.any():
                avg_cost = expected_costs[benign_mask].mean().item()
                benign_count = benign_mask.sum().item()
            else:
                avg_cost = 0.0
                benign_count = 0

            # Overall metrics
            overall_cost = expected_costs.mean().item()

            results[budget] = {
                'recall': recall,
                'avg_benign_cost': avg_cost,
                'overall_cost': overall_cost,
                'harmful_count': harmful_count,
                'benign_count': benign_count
            }

            # Restore original budget
            model.budget = original_budget

    return results


def create_baseline_hierarchical_system(
    probe: LinearProbe,
    input_dim: int,
    monitor_cost: float = 1.0,
    audit_cost: float = 20.0,
    budget: float = 2.0
) -> HierarchicalMonitor:
    """Create hierarchical system using pre-trained baseline probe."""

    # Create hierarchical system
    model = HierarchicalMonitor(
        input_dim=input_dim,
        monitor_cost=monitor_cost,
        audit_cost=audit_cost,
        budget=budget
    )

    # Copy probe weights
    model.probe.linear.weight.data = probe.linear.weight.data.clone()
    model.probe.linear.bias.data = probe.linear.bias.data.clone()

    # Keep probe frozen and only train gates with simple heuristic
    for param in model.probe.parameters():
        param.requires_grad = False

    return model


def run_experiment():
    """Run the complete toy experiment."""

    # Parameters
    config = {
        "experiment_type": "toy_experiment_comparison",
        "n_samples": 500,
        "n_features": 8192,
        "harmful_ratio": 0.15,
        "signal_strength": 0.15,
        "monitor_cost": 1.0,
        "audit_cost": 20.0,
        "budget_levels": [0.5, 1.0, 2.0, 3.0, 5.0],
        "epochs": 200,
        "target_budget": 2.0
    }

    description = """
    Toy experiment comparing hierarchical monitoring approaches:

    1. Hierarchical training: End-to-end training with soft gates
    2. Baseline + Hierarchical: Standalone probe training + hierarchical deployment

    Dataset includes 10% examples with completely random features that only
    the expensive monitor can detect accurately.
    """

    with ExperimentLogger("toy_experiment", description, config) as logger:
        print("=== Toy Experiment: Hierarchical Monitoring vs Baseline ===\n")

        print(f"Dataset: {config['n_samples']} samples, {config['n_features']} features, "
              f"{config['harmful_ratio']:.1%} harmful rate")
        print(f"Costs: Monitor=${config['monitor_cost']}, Audit=${config['audit_cost']}")
        print(f"Budget levels: {config['budget_levels']}\n")

        # Generate dataset
        print("Generating toy dataset...")
        features, labels, monitor_scores = generate_toy_dataset(
            n_samples=config["n_samples"],
            n_features=config["n_features"],
            harmful_ratio=config["harmful_ratio"],
            signal_strength=config["signal_strength"]
        )

        harmful_count = (labels == 1).sum().item()
        benign_count = (labels == 0).sum().item()
        print(f"Generated {harmful_count} harmful and {benign_count} benign examples\n")

        # Save dataset
        logger.save_data({
            'features': features,
            'labels': labels,
            'monitor_scores': monitor_scores
        }, 'dataset')

        # Train hierarchical model (our approach)
        print("Training hierarchical model with soft gates...")
        hierarchical_model, hier_losses = train_hierarchical_model(
            features, labels, monitor_scores,
            monitor_cost=config["monitor_cost"], audit_cost=config["audit_cost"],
            budget=config["target_budget"], epochs=config["epochs"]
        )
        print()

        # Train baseline probe
        print("Training baseline probe for standalone classification...")
        baseline_probe, baseline_losses = train_baseline_probe(
            features, labels, epochs=config["epochs"]
        )
        print()

        # Create baseline hierarchical system
        print("Creating baseline hierarchical system...")
        baseline_hierarchical = create_baseline_hierarchical_system(
            baseline_probe, config["n_features"],
            monitor_cost=config["monitor_cost"], audit_cost=config["audit_cost"]
        )
        print()

        # Evaluate both systems
        print("Evaluating hierarchical system...")
        hier_results = evaluate_hierarchical_system(
            hierarchical_model, features, labels, monitor_scores, config["budget_levels"]
        )

        print("Evaluating baseline system...")
        baseline_results = evaluate_hierarchical_system(
            baseline_hierarchical, features, labels, monitor_scores, config["budget_levels"]
        )

        # Save training histories
        logger.save_data({
            'hierarchical_losses': hier_losses,
            'baseline_losses': baseline_losses
        }, 'training_losses')

        # Save results
        logger.save_data({
            'hierarchical_results': hier_results,
            'baseline_results': baseline_results
        }, 'evaluation_results')

        # Print results
        print("\n=== RESULTS COMPARISON ===")
        print(f"{'Budget':<8} {'Method':<12} {'Recall':<8} {'Avg Cost':<10} {'Overall Cost':<12}")
        print("-" * 60)

        for budget in config["budget_levels"]:
            hier_r = hier_results[budget]
            base_r = baseline_results[budget]

            print(f"${budget:<7} {'Hierarchical':<12} {hier_r['recall']:<8.3f} "
                  f"${hier_r['avg_benign_cost']:<9.2f} ${hier_r['overall_cost']:<11.2f}")
            print(f"${budget:<7} {'Baseline':<12} {base_r['recall']:<8.3f} "
                  f"${base_r['avg_benign_cost']:<9.2f} ${base_r['overall_cost']:<11.2f}")

            # Calculate improvement
            if base_r['recall'] > 0:
                recall_improvement = ((hier_r['recall'] - base_r['recall']) /
                                    base_r['recall']) * 100
                print(f"{'':>20} Improvement: {recall_improvement:+.1f}% recall")
            print()

        # Plot results
        plot_results(config["budget_levels"], hier_results, baseline_results, logger)

        # Print final summary
        print("\n=== SUMMARY ===")
        target_budget = config["target_budget"]
        if target_budget in hier_results:
            hier_recall = hier_results[target_budget]['recall']
            base_recall = baseline_results[target_budget]['recall']
            improvement = ((hier_recall - base_recall) / base_recall) * 100
            print(f"At ${target_budget} budget:")
            print(f"  Hierarchical training: {hier_recall:.3f} recall")
            print(f"  Baseline approach: {base_recall:.3f} recall")
            print(f"  Improvement: {improvement:+.1f}%")

            # Log final results
            final_results = {
                'target_budget': target_budget,
                'hierarchical_recall': hier_recall,
                'baseline_recall': base_recall,
                'improvement_percent': improvement,
                'harmful_count': harmful_count,
                'benign_count': benign_count
            }
            logger.log_results(final_results)

        return hier_results, baseline_results


def plot_results(
    budget_levels: List[float],
    hier_results: Dict[float, Dict[str, float]],
    baseline_results: Dict[float, Dict[str, float]],
    logger=None
):
    """Plot comparison results."""

    hier_recalls = [hier_results[b]['recall'] for b in budget_levels]
    base_recalls = [baseline_results[b]['recall'] for b in budget_levels]

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.plot(budget_levels, hier_recalls, 'o-', label='Hierarchical Training', linewidth=2)
    plt.plot(budget_levels, base_recalls, 's--', label='Baseline + Hierarchical', linewidth=2)
    plt.xlabel('Budget ($)')
    plt.ylabel('Recall on Harmful Examples')
    plt.title('Recall vs Budget')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot cost adherence
    hier_costs = [hier_results[b]['avg_benign_cost'] for b in budget_levels]
    base_costs = [baseline_results[b]['avg_benign_cost'] for b in budget_levels]

    plt.subplot(1, 2, 2)
    plt.plot(budget_levels, hier_costs, 'o-', label='Hierarchical Training', linewidth=2)
    plt.plot(budget_levels, base_costs, 's--', label='Baseline + Hierarchical', linewidth=2)
    plt.plot(budget_levels, budget_levels, 'k:', label='Target Budget', alpha=0.7)
    plt.xlabel('Budget ($)')
    plt.ylabel('Average Cost on Benign Examples ($)')
    plt.title('Cost Adherence')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if logger:
        logger.save_plot('comparison_results')
    else:
        plt.savefig('toy_experiment_results.png', dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    run_experiment()