#!/usr/bin/env python3
"""
Modular 3-phase curriculum learning with different loss function combinations.

This script implements the three-phase curriculum learning approach using
the modular loss functions from src/training/loss.py, allowing easy comparison
of different recall and cost loss combinations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
import os

# Import modular loss functions directly
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'training'))

# Import loss classes directly
import loss
from loss import HierarchicalLoss, RecallLoss, CostLoss
from experiment_utils import ExperimentLogger
from training_comparison import HardHierarchicalMonitor, generate_hard_dataset


def get_recall_loss_config(recall_type):
    """Get standard regularization configs for different recall losses."""
    configs = {
        "log_likelihood": {
            # Standard for log-likelihood: moderate regularization
            "weight_decay": 0.01,
        },
        "linear": {
            # Linear loss is smooth: can handle stronger regularization
            "weight_decay": 0.02,
        },
        "exponential": {
            # Exponential is unstable: needs strong regularization
            "weight_decay": 0.05,
        },
        "focal": {
            # Focal loss has gamma parameter and is designed for hard examples
            "weight_decay": 0.01,
            "gamma": 2.0,  # Standard focal loss gamma
        }
    }
    return configs.get(recall_type, {"weight_decay": 0.01})


def get_cost_loss_config(cost_type):
    """Get standard regularization configs for different cost losses."""
    configs = {
        "quadratic": {
            # Quadratic: scale for (cost_ratio-1)^2 to give ~1-16 range
            "scale": 2.0,  # Maps (0.25-5.0) range to reasonable loss values
            "weight_decay": 0.3,  # Very high regularization for quadratic cost
        },
        "exponential_quadratic": {
            # Exponential can explode: strong regularization, very small scale
            "scale": 0.1,  # Very small scale to prevent explosion
            "weight_decay": 0.6,  # Maximum regularization for exponential
        },
        "huber": {
            # Huber is robust but still needs proper scaling
            "scale": 1.5,  # Moderate scaling for huber robustness
            "delta": 1.0,  # Standard huber delta
            "weight_decay": 0.5,  # Maximum regularization for huber cost
        },
        "absolute": {
            # Absolute loss: moderate regularization
            "scale": 1.0,
            "weight_decay": 0.015,
        }
    }
    return configs.get(cost_type, {"scale": 1.0, "weight_decay": 0.01})


def get_combined_weight_decay(recall_type, cost_type):
    """Combine weight decay from both loss types."""
    recall_config = get_recall_loss_config(recall_type)
    cost_config = get_cost_loss_config(cost_type)

    # Take the maximum weight decay (most conservative)
    recall_wd = recall_config.get("weight_decay", 0.01)
    cost_wd = cost_config.get("weight_decay", 0.01)

    return max(recall_wd, cost_wd)


def evaluate_recall_cost(model, X, y, m, c_probe, c_monitor, c_audit):
    """Evaluate model using modular loss functions."""
    model.eval()
    with torch.no_grad():
        outputs = model(X, m)
        p_audit = outputs['p_audit']
        p_gate = outputs['p_gate']

        # Calculate costs including probe cost
        probe_cost = c_probe * torch.ones_like(p_audit)  # Probe always runs
        monitor_cost = c_monitor * p_gate  # Monitor cost when called
        audit_cost = c_audit * p_audit    # Audit cost when auditing
        expected_total_cost = probe_cost + monitor_cost + audit_cost

        # Binary classification accuracy
        probe_probs = torch.sigmoid(outputs['probe_scores'])
        probe_preds = (probe_probs > 0.5).float()
        accuracy = (probe_preds == y).float().mean().item()

        # Recall: P(audit | harmful)
        harmful_mask = (y == 1)
        if harmful_mask.any():
            recall = p_audit[harmful_mask].mean().item()
        else:
            recall = 0.0

        # Average cost
        avg_cost = expected_total_cost.mean().item()

        return accuracy, recall, avg_cost


def evaluate_all_losses(model, X, y, m, c_probe, c_monitor, c_audit, budget, hierarchical_loss, baseline_loss):
    """Evaluate all loss components for plotting."""
    model.eval()
    with torch.no_grad():
        outputs = model(X, m)
        p_audit = outputs['p_audit']
        p_gate = outputs['p_gate']

        # Classification loss (BCE on probe scores)
        probe_logits = outputs['probe_scores']
        classification_loss = baseline_loss(probe_logits, y.float()).item()

        # Prepare for hierarchical loss
        probe_cost = c_probe * torch.ones_like(p_audit)
        monitor_cost = c_monitor * p_gate
        audit_cost = c_audit * p_audit
        expected_total_cost = probe_cost + monitor_cost + audit_cost
        outputs['expected_total_cost'] = expected_total_cost

        # Hierarchical loss components
        loss_dict = hierarchical_loss(outputs, y, budget)
        recall_loss = loss_dict['recall_loss'].item()
        cost_loss = loss_dict['cost_loss'].item()
        total_loss = loss_dict['total_loss'].item()

        return classification_loss, recall_loss, cost_loss, total_loss


def evaluate_enforced_budget_recall(model, X, y, m, c_probe, c_monitor, c_audit, budget):
    """
    Evaluate recall under enforced budget constraint.

    Budget is per benign example. We allocate budget * n_benign total budget,
    then process examples sequentially until budget is exhausted.
    """
    model.eval()
    with torch.no_grad():
        # Shuffle data to simulate random order
        n_samples = len(X)
        indices = torch.randperm(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        m_shuffled = m[indices]

        # Get model predictions on shuffled data
        outputs = model(X_shuffled, m_shuffled)
        p_audit = outputs['p_audit']
        p_gate = outputs['p_gate']

        # Calculate costs for each sample
        probe_cost = c_probe
        monitor_cost = c_monitor * p_gate
        audit_cost = c_audit * p_audit
        total_cost_per_sample = probe_cost + monitor_cost + audit_cost

        # Calculate total budget based on benign examples
        n_benign = (y_shuffled == 0).sum().item()
        total_budget = budget * n_benign

        # Spend budget sequentially until exhausted
        cumulative_cost = torch.cumsum(total_cost_per_sample, dim=0)
        within_budget = cumulative_cost <= total_budget

        # Only consider samples within budget
        audited_samples = within_budget
        audited_labels = y_shuffled[audited_samples]
        audited_p_audit = p_audit[audited_samples]

        if len(audited_labels) == 0:
            return 0.0  # No samples could be processed within budget

        # Calculate recall: fraction of harmful examples that were audited
        harmful_mask = (audited_labels == 1)
        if not harmful_mask.any():
            return 0.0  # No harmful examples in audited samples

        # Raw recall = average audit probability on harmful examples within budget
        harmful_audit_probs = audited_p_audit[harmful_mask]
        enforced_recall = harmful_audit_probs.mean().item()

        return enforced_recall


def train_three_phase_curriculum(
    X_train, y_train, m_train, X_test, y_test, m_test,
    config, recall_loss_type="log_likelihood", cost_penalty_type="quadratic"
):
    """Train using three-phase curriculum with specified loss functions."""

    n_features = X_train.shape[1]
    n_epochs = config["n_epochs"]

    # Create models
    model_joint = HardHierarchicalMonitor(n_features, hidden_dim=64)  # More capacity
    model_baseline = HardHierarchicalMonitor(n_features, hidden_dim=64)
    model_baseline.load_state_dict(model_joint.state_dict())

    # Create modular loss functions with loss-specific regularization
    recall_config = get_recall_loss_config(recall_loss_type)
    cost_config = get_cost_loss_config(cost_penalty_type)

    hierarchical_loss = HierarchicalLoss(
        recall_loss_type=recall_loss_type,
        cost_penalty_type=cost_penalty_type,
        lambda_cost=config["lambda_cost"],
        recall_gamma=recall_config["gamma"] if "gamma" in recall_config else 2.0,
        cost_scale=cost_config["scale"],
        cost_delta=cost_config["delta"] if "delta" in cost_config else 1.0
    )

    baseline_loss = nn.BCEWithLogitsLoss()

    history = {
        'phase1_epochs': list(range(0, n_epochs)),
        'phase2_epochs': list(range(n_epochs, 2*n_epochs)),
        'phase3_epochs': list(range(2*n_epochs, 3*n_epochs)),

        # Track metrics for both methods
        'joint_test_accuracy': [], 'joint_test_recall': [], 'joint_test_cost': [],
        'joint_train_accuracy': [], 'joint_train_recall': [], 'joint_train_cost': [],
        'baseline_test_accuracy': [], 'baseline_test_recall': [], 'baseline_test_cost': [],
        'baseline_train_accuracy': [], 'baseline_train_recall': [], 'baseline_train_cost': [],

        # Track enforced budget recall (most interpretable metric)
        'joint_test_enforced_recall': [], 'joint_train_enforced_recall': [],
        'baseline_test_enforced_recall': [], 'baseline_train_enforced_recall': [],

        # Track individual loss components
        'joint_test_classification_loss': [], 'joint_test_recall_loss': [], 'joint_test_cost_loss': [], 'joint_test_total_loss': [],
        'joint_train_classification_loss': [], 'joint_train_recall_loss': [], 'joint_train_cost_loss': [], 'joint_train_total_loss': [],
        'baseline_test_classification_loss': [], 'baseline_test_recall_loss': [], 'baseline_test_cost_loss': [], 'baseline_test_total_loss': [],
        'baseline_train_classification_loss': [], 'baseline_train_recall_loss': [], 'baseline_train_cost_loss': [], 'baseline_train_total_loss': [],
    }

    # Get loss-specific weight decay
    combined_weight_decay = get_combined_weight_decay(recall_loss_type, cost_penalty_type)

    print(f"Using {recall_loss_type} recall loss and {cost_penalty_type} cost penalty")
    print(f"Combined weight decay: {combined_weight_decay}")

    # Phase 1: Both methods train probe for classification (shared phase)
    print("\n=== PHASE 1: Shared probe training ===")
    probe_optimizer = optim.Adam(model_joint.get_probe_parameters(),
                                lr=config["lr"], weight_decay=combined_weight_decay)

    for epoch in range(n_epochs):
        # Train shared probe
        model_joint.train()
        probe_optimizer.zero_grad()
        outputs = model_joint(X_train, m_train)
        loss = baseline_loss(outputs['probe_scores'], y_train.float())
        loss.backward()
        probe_optimizer.step()

        # Copy weights to baseline model
        model_baseline.load_state_dict(model_joint.state_dict())

        # Evaluate both models (they're identical at this point)
        test_acc, test_recall, test_cost = evaluate_recall_cost(
            model_joint, X_test, y_test, m_test,
            config["c_probe"], config["c_monitor"], config["c_audit"]
        )
        train_acc, train_recall, train_cost = evaluate_recall_cost(
            model_joint, X_train, y_train, m_train,
            config["c_probe"], config["c_monitor"], config["c_audit"]
        )

        # Evaluate loss components
        test_cls_loss, test_rec_loss, test_cost_loss, test_total_loss = evaluate_all_losses(
            model_joint, X_test, y_test, m_test,
            config["c_probe"], config["c_monitor"], config["c_audit"], config["budget"],
            hierarchical_loss, baseline_loss
        )
        train_cls_loss, train_rec_loss, train_cost_loss, train_total_loss = evaluate_all_losses(
            model_joint, X_train, y_train, m_train,
            config["c_probe"], config["c_monitor"], config["c_audit"], config["budget"],
            hierarchical_loss, baseline_loss
        )

        # Evaluate enforced budget recall (most interpretable metric)
        test_enforced_recall = evaluate_enforced_budget_recall(
            model_joint, X_test, y_test, m_test,
            config["c_probe"], config["c_monitor"], config["c_audit"], config["budget"]
        )
        train_enforced_recall = evaluate_enforced_budget_recall(
            model_joint, X_train, y_train, m_train,
            config["c_probe"], config["c_monitor"], config["c_audit"], config["budget"]
        )

        # Record for both methods (identical in phase 1)
        history['joint_test_accuracy'].append(test_acc)
        history['joint_test_recall'].append(test_recall)
        history['joint_test_cost'].append(test_cost)
        history['joint_train_accuracy'].append(train_acc)
        history['joint_train_recall'].append(train_recall)
        history['joint_train_cost'].append(train_cost)

        history['baseline_test_accuracy'].append(test_acc)
        history['baseline_test_recall'].append(test_recall)
        history['baseline_test_cost'].append(test_cost)
        history['baseline_train_accuracy'].append(train_acc)
        history['baseline_train_recall'].append(train_recall)
        history['baseline_train_cost'].append(train_cost)

        # Record enforced budget recall (identical in phase 1)
        history['joint_test_enforced_recall'].append(test_enforced_recall)
        history['joint_train_enforced_recall'].append(train_enforced_recall)
        history['baseline_test_enforced_recall'].append(test_enforced_recall)
        history['baseline_train_enforced_recall'].append(train_enforced_recall)

        # Record loss components (identical in phase 1)
        history['joint_test_classification_loss'].append(test_cls_loss)
        history['joint_test_recall_loss'].append(test_rec_loss)
        history['joint_test_cost_loss'].append(test_cost_loss)
        history['joint_test_total_loss'].append(test_total_loss)
        history['joint_train_classification_loss'].append(train_cls_loss)
        history['joint_train_recall_loss'].append(train_rec_loss)
        history['joint_train_cost_loss'].append(train_cost_loss)
        history['joint_train_total_loss'].append(train_total_loss)

        history['baseline_test_classification_loss'].append(test_cls_loss)
        history['baseline_test_recall_loss'].append(test_rec_loss)
        history['baseline_test_cost_loss'].append(test_cost_loss)
        history['baseline_test_total_loss'].append(test_total_loss)
        history['baseline_train_classification_loss'].append(train_cls_loss)
        history['baseline_train_recall_loss'].append(train_rec_loss)
        history['baseline_train_cost_loss'].append(train_cost_loss)
        history['baseline_train_total_loss'].append(train_total_loss)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Acc={test_acc:.3f}, Recall={test_recall:.3f}, Cost=${test_cost:.2f}")

    # Fit normalization on training data after Phase 1
    print("Fitting normalization statistics on training data...")
    model_joint.fit_normalization(X_train, m_train)
    model_baseline.fit_normalization(X_train, m_train)

    # Phase 2: Joint method trains probe+gates, Baseline continues probe-only
    print("\n=== PHASE 2: Joint training vs Continued probe training ===")

    # Joint method: train probe + gates together
    joint_optimizer = optim.Adam(model_joint.parameters(),
                                lr=config["lr"], weight_decay=combined_weight_decay)

    # Baseline method: continue training probe only
    baseline_probe_optimizer = optim.Adam(model_baseline.get_probe_parameters(),
                                        lr=config["lr"], weight_decay=combined_weight_decay)

    for epoch in range(n_epochs):
        # Train joint method (probe + gates)
        model_joint.train()
        joint_optimizer.zero_grad()

        outputs = model_joint(X_train, m_train)

        # Prepare outputs for hierarchical loss
        probe_cost = config["c_probe"] * torch.ones_like(outputs['p_audit'])
        monitor_cost = config["c_monitor"] * outputs['p_gate']
        audit_cost = config["c_audit"] * outputs['p_audit']
        expected_total_cost = probe_cost + monitor_cost + audit_cost

        outputs['expected_total_cost'] = expected_total_cost

        loss_dict = hierarchical_loss(outputs, y_train, config["budget"])
        loss_dict['total_loss'].backward()
        joint_optimizer.step()

        # Train baseline method (probe only)
        model_baseline.train()
        baseline_probe_optimizer.zero_grad()
        baseline_outputs = model_baseline(X_train, m_train)
        baseline_loss_val = baseline_loss(baseline_outputs['probe_scores'], y_train.float())
        baseline_loss_val.backward()
        baseline_probe_optimizer.step()

        # Evaluate both models
        joint_test_acc, joint_test_recall, joint_test_cost = evaluate_recall_cost(
            model_joint, X_test, y_test, m_test,
            config["c_probe"], config["c_monitor"], config["c_audit"]
        )
        joint_train_acc, joint_train_recall, joint_train_cost = evaluate_recall_cost(
            model_joint, X_train, y_train, m_train,
            config["c_probe"], config["c_monitor"], config["c_audit"]
        )

        baseline_test_acc, baseline_test_recall, baseline_test_cost = evaluate_recall_cost(
            model_baseline, X_test, y_test, m_test,
            config["c_probe"], config["c_monitor"], config["c_audit"]
        )
        baseline_train_acc, baseline_train_recall, baseline_train_cost = evaluate_recall_cost(
            model_baseline, X_train, y_train, m_train,
            config["c_probe"], config["c_monitor"], config["c_audit"]
        )

        # Evaluate loss components for both models
        joint_test_cls, joint_test_rec, joint_test_cost_loss, joint_test_total = evaluate_all_losses(
            model_joint, X_test, y_test, m_test, config["c_probe"], config["c_monitor"],
            config["c_audit"], config["budget"], hierarchical_loss, baseline_loss
        )
        joint_train_cls, joint_train_rec, joint_train_cost_loss, joint_train_total = evaluate_all_losses(
            model_joint, X_train, y_train, m_train, config["c_probe"], config["c_monitor"],
            config["c_audit"], config["budget"], hierarchical_loss, baseline_loss
        )

        baseline_test_cls, baseline_test_rec, baseline_test_cost_loss, baseline_test_total = evaluate_all_losses(
            model_baseline, X_test, y_test, m_test, config["c_probe"], config["c_monitor"],
            config["c_audit"], config["budget"], hierarchical_loss, baseline_loss
        )
        baseline_train_cls, baseline_train_rec, baseline_train_cost_loss, baseline_train_total = evaluate_all_losses(
            model_baseline, X_train, y_train, m_train, config["c_probe"], config["c_monitor"],
            config["c_audit"], config["budget"], hierarchical_loss, baseline_loss
        )

        # Record history
        history['joint_test_accuracy'].append(joint_test_acc)
        history['joint_test_recall'].append(joint_test_recall)
        history['joint_test_cost'].append(joint_test_cost)
        history['joint_train_accuracy'].append(joint_train_acc)
        history['joint_train_recall'].append(joint_train_recall)
        history['joint_train_cost'].append(joint_train_cost)

        history['baseline_test_accuracy'].append(baseline_test_acc)
        history['baseline_test_recall'].append(baseline_test_recall)
        history['baseline_test_cost'].append(baseline_test_cost)
        history['baseline_train_accuracy'].append(baseline_train_acc)
        history['baseline_train_recall'].append(baseline_train_recall)
        history['baseline_train_cost'].append(baseline_train_cost)

        # Evaluate enforced budget recall for both methods
        joint_test_enforced = evaluate_enforced_budget_recall(
            model_joint, X_test, y_test, m_test,
            config["c_probe"], config["c_monitor"], config["c_audit"], config["budget"]
        )
        joint_train_enforced = evaluate_enforced_budget_recall(
            model_joint, X_train, y_train, m_train,
            config["c_probe"], config["c_monitor"], config["c_audit"], config["budget"]
        )
        baseline_test_enforced = evaluate_enforced_budget_recall(
            model_baseline, X_test, y_test, m_test,
            config["c_probe"], config["c_monitor"], config["c_audit"], config["budget"]
        )
        baseline_train_enforced = evaluate_enforced_budget_recall(
            model_baseline, X_train, y_train, m_train,
            config["c_probe"], config["c_monitor"], config["c_audit"], config["budget"]
        )

        # Record enforced budget recall
        history['joint_test_enforced_recall'].append(joint_test_enforced)
        history['joint_train_enforced_recall'].append(joint_train_enforced)
        history['baseline_test_enforced_recall'].append(baseline_test_enforced)
        history['baseline_train_enforced_recall'].append(baseline_train_enforced)

        # Record loss components
        history['joint_test_classification_loss'].append(joint_test_cls)
        history['joint_test_recall_loss'].append(joint_test_rec)
        history['joint_test_cost_loss'].append(joint_test_cost_loss)
        history['joint_test_total_loss'].append(joint_test_total)
        history['joint_train_classification_loss'].append(joint_train_cls)
        history['joint_train_recall_loss'].append(joint_train_rec)
        history['joint_train_cost_loss'].append(joint_train_cost_loss)
        history['joint_train_total_loss'].append(joint_train_total)

        history['baseline_test_classification_loss'].append(baseline_test_cls)
        history['baseline_test_recall_loss'].append(baseline_test_rec)
        history['baseline_test_cost_loss'].append(baseline_test_cost_loss)
        history['baseline_test_total_loss'].append(baseline_test_total)
        history['baseline_train_classification_loss'].append(baseline_train_cls)
        history['baseline_train_recall_loss'].append(baseline_train_rec)
        history['baseline_train_cost_loss'].append(baseline_train_cost_loss)
        history['baseline_train_total_loss'].append(baseline_train_total)

        if epoch % 5 == 0:
            print(f"  Epoch {n_epochs + epoch}:")
            print(f"    Joint: Acc={joint_test_acc:.3f}, Recall={joint_test_recall:.3f}, Cost=${joint_test_cost:.2f}")
            print(f"    Baseline: Acc={baseline_test_acc:.3f}, Recall={baseline_test_recall:.3f}, Cost=${baseline_test_cost:.2f}")

    # Phase 3: Joint method continues, Baseline trains gates with frozen probe
    print("\n=== PHASE 3: Joint continues vs Baseline gates training ===")

    # Freeze baseline probe for gate training
    for param in model_baseline.probe.parameters():
        param.requires_grad = False

    baseline_gate_optimizer = optim.Adam(model_baseline.get_gate_parameters(),
                                        lr=config["lr"], weight_decay=combined_weight_decay)

    for epoch in range(n_epochs):
        # Joint method: NO TRAINING in Phase 3 (frozen model for fair comparison)
        # Just evaluate the model without any parameter updates

        # Train baseline gates only
        model_baseline.train()
        baseline_gate_optimizer.zero_grad()

        baseline_outputs = model_baseline(X_train, m_train)
        baseline_probe_cost = config["c_probe"] * torch.ones_like(baseline_outputs['p_audit'])
        baseline_monitor_cost = config["c_monitor"] * baseline_outputs['p_gate']
        baseline_audit_cost = config["c_audit"] * baseline_outputs['p_audit']
        baseline_expected_total_cost = baseline_probe_cost + baseline_monitor_cost + baseline_audit_cost
        baseline_outputs['expected_total_cost'] = baseline_expected_total_cost

        baseline_loss_dict = hierarchical_loss(baseline_outputs, y_train, config["budget"])
        baseline_loss_dict['total_loss'].backward()
        baseline_gate_optimizer.step()

        # Evaluate both models
        joint_test_acc, joint_test_recall, joint_test_cost = evaluate_recall_cost(
            model_joint, X_test, y_test, m_test,
            config["c_probe"], config["c_monitor"], config["c_audit"]
        )
        joint_train_acc, joint_train_recall, joint_train_cost = evaluate_recall_cost(
            model_joint, X_train, y_train, m_train,
            config["c_probe"], config["c_monitor"], config["c_audit"]
        )

        baseline_test_acc, baseline_test_recall, baseline_test_cost = evaluate_recall_cost(
            model_baseline, X_test, y_test, m_test,
            config["c_probe"], config["c_monitor"], config["c_audit"]
        )
        baseline_train_acc, baseline_train_recall, baseline_train_cost = evaluate_recall_cost(
            model_baseline, X_train, y_train, m_train,
            config["c_probe"], config["c_monitor"], config["c_audit"]
        )

        # Evaluate loss components for both models
        joint_test_cls, joint_test_rec, joint_test_cost_loss, joint_test_total = evaluate_all_losses(
            model_joint, X_test, y_test, m_test, config["c_probe"], config["c_monitor"],
            config["c_audit"], config["budget"], hierarchical_loss, baseline_loss
        )
        joint_train_cls, joint_train_rec, joint_train_cost_loss, joint_train_total = evaluate_all_losses(
            model_joint, X_train, y_train, m_train, config["c_probe"], config["c_monitor"],
            config["c_audit"], config["budget"], hierarchical_loss, baseline_loss
        )

        baseline_test_cls, baseline_test_rec, baseline_test_cost_loss, baseline_test_total = evaluate_all_losses(
            model_baseline, X_test, y_test, m_test, config["c_probe"], config["c_monitor"],
            config["c_audit"], config["budget"], hierarchical_loss, baseline_loss
        )
        baseline_train_cls, baseline_train_rec, baseline_train_cost_loss, baseline_train_total = evaluate_all_losses(
            model_baseline, X_train, y_train, m_train, config["c_probe"], config["c_monitor"],
            config["c_audit"], config["budget"], hierarchical_loss, baseline_loss
        )

        # Record history
        history['joint_test_accuracy'].append(joint_test_acc)
        history['joint_test_recall'].append(joint_test_recall)
        history['joint_test_cost'].append(joint_test_cost)
        history['joint_train_accuracy'].append(joint_train_acc)
        history['joint_train_recall'].append(joint_train_recall)
        history['joint_train_cost'].append(joint_train_cost)

        history['baseline_test_accuracy'].append(baseline_test_acc)
        history['baseline_test_recall'].append(baseline_test_recall)
        history['baseline_test_cost'].append(baseline_test_cost)
        history['baseline_train_accuracy'].append(baseline_train_acc)
        history['baseline_train_recall'].append(baseline_train_recall)
        history['baseline_train_cost'].append(baseline_train_cost)

        # Evaluate enforced budget recall for both methods
        joint_test_enforced = evaluate_enforced_budget_recall(
            model_joint, X_test, y_test, m_test,
            config["c_probe"], config["c_monitor"], config["c_audit"], config["budget"]
        )
        joint_train_enforced = evaluate_enforced_budget_recall(
            model_joint, X_train, y_train, m_train,
            config["c_probe"], config["c_monitor"], config["c_audit"], config["budget"]
        )
        baseline_test_enforced = evaluate_enforced_budget_recall(
            model_baseline, X_test, y_test, m_test,
            config["c_probe"], config["c_monitor"], config["c_audit"], config["budget"]
        )
        baseline_train_enforced = evaluate_enforced_budget_recall(
            model_baseline, X_train, y_train, m_train,
            config["c_probe"], config["c_monitor"], config["c_audit"], config["budget"]
        )

        # Record enforced budget recall
        history['joint_test_enforced_recall'].append(joint_test_enforced)
        history['joint_train_enforced_recall'].append(joint_train_enforced)
        history['baseline_test_enforced_recall'].append(baseline_test_enforced)
        history['baseline_train_enforced_recall'].append(baseline_train_enforced)

        # Record loss components
        history['joint_test_classification_loss'].append(joint_test_cls)
        history['joint_test_recall_loss'].append(joint_test_rec)
        history['joint_test_cost_loss'].append(joint_test_cost_loss)
        history['joint_test_total_loss'].append(joint_test_total)
        history['joint_train_classification_loss'].append(joint_train_cls)
        history['joint_train_recall_loss'].append(joint_train_rec)
        history['joint_train_cost_loss'].append(joint_train_cost_loss)
        history['joint_train_total_loss'].append(joint_train_total)

        history['baseline_test_classification_loss'].append(baseline_test_cls)
        history['baseline_test_recall_loss'].append(baseline_test_rec)
        history['baseline_test_cost_loss'].append(baseline_test_cost_loss)
        history['baseline_test_total_loss'].append(baseline_test_total)
        history['baseline_train_classification_loss'].append(baseline_train_cls)
        history['baseline_train_recall_loss'].append(baseline_train_rec)
        history['baseline_train_cost_loss'].append(baseline_train_cost_loss)
        history['baseline_train_total_loss'].append(baseline_train_total)

        if epoch % 5 == 0:
            print(f"  Epoch {2*n_epochs + epoch}:")
            print(f"    Joint: Acc={joint_test_acc:.3f}, Recall={joint_test_recall:.3f}, Cost=${joint_test_cost:.2f}")
            print(f"    Baseline: Acc={baseline_test_acc:.3f}, Recall={baseline_test_recall:.3f}, Cost=${baseline_test_cost:.2f}")

    return history, model_joint, model_baseline


def calculate_subplot_axis_limits(probe_scores, monitor_scores, padding_factor=0.1, min_padding=0.1):
    """Calculate dynamic axis limits for a single subplot."""
    import numpy as np

    # Check for NaN values and handle them
    if torch.isnan(probe_scores).any():
        print(f"Warning: NaN values detected in probe scores, using default limits")
        return [-4.0, 4.0], [-4.0, 4.0]  # Return safe default limits

    # Calculate ranges with padding
    monitor_min, monitor_max = monitor_scores.min().item(), monitor_scores.max().item()
    probe_min, probe_max = probe_scores.min().item(), probe_scores.max().item()

    monitor_range = monitor_max - monitor_min
    probe_range = probe_max - probe_min

    monitor_padding = max(padding_factor * monitor_range, min_padding)
    probe_padding = max(padding_factor * probe_range, min_padding)

    x_limits = [monitor_min - monitor_padding, monitor_max + monitor_padding]
    y_limits = [probe_min - probe_padding, probe_max + probe_padding]

    return x_limits, y_limits


def plot_modular_curriculum_results(histories, loss_combinations, config,
                                    models_joint, models_baseline, X_test, y_test, m_test):
    """Plot comprehensive comparison of different loss function combinations."""

    n_epochs = config["n_epochs"]
    n_combinations = len(loss_combinations)

    # Create 11 columns (4 losses + enforced budget + 3 joint outputs + 3 baseline outputs) x n_combinations rows layout
    n_cols = 11
    fig, axes = plt.subplots(n_combinations, n_cols, figsize=(n_cols * 1.5, n_combinations * 1.5))

    # Handle single combination case
    if n_combinations == 1:
        axes = axes.reshape(1, -1)

    # Create continuous timeline
    total_epochs = 3 * n_epochs
    epochs = list(range(total_epochs))

    # Loss formulas for each column
    loss_formulas = {
        'classification': 'BCE(probe_scores, labels)',
        'recall': '',  # Will be filled based on loss type
        'cost': '',    # Will be filled based on loss type
        'total': 'recall_loss + λ·cost_loss'
    }

    # Plot each loss combination in its own row
    for i, (recall_type, cost_type) in enumerate(loss_combinations):
        history = histories[i]

        # Update formulas for this combination
        if recall_type == 'log_likelihood':
            loss_formulas['recall'] = '-log(P(audit|harmful))'
        elif recall_type == 'linear':
            loss_formulas['recall'] = '1 - P(audit|harmful)'
        elif recall_type == 'exponential':
            loss_formulas['recall'] = 'exp(1 - P(audit|harmful)) - 1'
        elif recall_type == 'focal':
            loss_formulas['recall'] = '-(1-p)^γ·log(p)'

        if cost_type == 'quadratic':
            loss_formulas['cost'] = '2.0·(cost/budget - 1)²'
        elif cost_type == 'exponential_quadratic':
            loss_formulas['cost'] = 'exp(0.1·(cost/budget - 1)²) - 1'
        elif cost_type == 'huber':
            loss_formulas['cost'] = '1.5·huber(cost/budget - 1)'

        # Column 1: Classification Loss
        ax = axes[i, 0]
        if 'joint_test_classification_loss' in history:
            ax.plot(epochs, history['joint_test_classification_loss'], 'b-', linewidth=0.8, marker='o', markersize=3, label='Joint (test)')
            ax.plot(epochs, history['joint_train_classification_loss'], 'b:', linewidth=0.8, marker='s', markersize=1.5, alpha=0.7, label='Joint (train)')
            ax.plot(epochs, history['baseline_test_classification_loss'], 'r-', linewidth=0.8, marker='o', markersize=3, label='Baseline (test)')
            ax.plot(epochs, history['baseline_train_classification_loss'], 'r:', linewidth=0.8, marker='s', markersize=1.5, alpha=0.7, label='Baseline (train)')

        ax.set_title(f'Classification Loss\n{loss_formulas["classification"]}', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Column 2: Recall Loss
        ax = axes[i, 1]
        if 'joint_test_recall_loss' in history:
            ax.plot(epochs, history['joint_test_recall_loss'], 'b-', linewidth=0.8, marker='o', markersize=3)
            ax.plot(epochs, history['joint_train_recall_loss'], 'b:', linewidth=0.8, marker='s', markersize=1.5, alpha=0.7)
            ax.plot(epochs, history['baseline_test_recall_loss'], 'r-', linewidth=0.8, marker='o', markersize=3)
            ax.plot(epochs, history['baseline_train_recall_loss'], 'r:', linewidth=0.8, marker='s', markersize=1.5, alpha=0.7)

        ax.set_title(f'Recall Loss\n{loss_formulas["recall"]}', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Column 3: Cost Loss
        ax = axes[i, 2]
        if 'joint_test_cost_loss' in history:
            ax.plot(epochs, history['joint_test_cost_loss'], 'b-', linewidth=0.8, marker='o', markersize=3)
            ax.plot(epochs, history['joint_train_cost_loss'], 'b:', linewidth=0.8, marker='s', markersize=1.5, alpha=0.7)
            ax.plot(epochs, history['baseline_test_cost_loss'], 'r-', linewidth=0.8, marker='o', markersize=3)
            ax.plot(epochs, history['baseline_train_cost_loss'], 'r:', linewidth=0.8, marker='s', markersize=1.5, alpha=0.7)

        ax.set_title(f'Cost Loss\n{loss_formulas["cost"]}', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Column 4: Total Loss
        ax = axes[i, 3]
        if 'joint_test_total_loss' in history:
            ax.plot(epochs, history['joint_test_total_loss'], 'b-', linewidth=0.8, marker='o', markersize=3)
            ax.plot(epochs, history['joint_train_total_loss'], 'b:', linewidth=0.8, marker='s', markersize=1.5, alpha=0.7)
            ax.plot(epochs, history['baseline_test_total_loss'], 'r-', linewidth=0.8, marker='o', markersize=3)
            ax.plot(epochs, history['baseline_train_total_loss'], 'r:', linewidth=0.8, marker='s', markersize=1.5, alpha=0.7)

        ax.set_title(f'Total Loss\n{loss_formulas["total"]}', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Column 5: Enforced Budget Recall (most interpretable)
        ax = axes[i, 4]
        if 'joint_test_enforced_recall' in history:
            ax.plot(epochs, history['joint_test_enforced_recall'], 'b-', linewidth=0.8, marker='o', markersize=3)
            ax.plot(epochs, history['joint_train_enforced_recall'], 'b:', linewidth=0.8, marker='s', markersize=1.5, alpha=0.7)
            ax.plot(epochs, history['baseline_test_enforced_recall'], 'r-', linewidth=0.8, marker='o', markersize=3)
            ax.plot(epochs, history['baseline_train_enforced_recall'], 'r:', linewidth=0.8, marker='s', markersize=1.5, alpha=0.7)

        ax.set_title('Recall at Enforced Budget\n(interpretable metric)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)  # Recall is between 0 and 1

        # Create decision boundary plots using final joint model parameters
        final_joint_model = models_joint[i]
        final_joint_model.eval()

        # Extract actual model parameters for decision boundaries
        with torch.no_grad():
            # Get model parameters from the trained joint model
            u = final_joint_model.u.item()
            l = final_joint_model.l.item()
            beta_u = final_joint_model.beta_u.item()
            beta_l = final_joint_model.beta_l.item()
            alpha_1 = final_joint_model.alpha_1.item()
            alpha_2 = final_joint_model.alpha_2.item()
            beta_3 = final_joint_model.beta_3.item()
            a = final_joint_model.a.item()

        # Get probe scores for this specific model to calculate individual subplot limits
        with torch.no_grad():
            test_outputs = final_joint_model(X_test, m_test)
            probe_scores_norm = test_outputs['probe_scores_normalized']

        # Calculate individual axis limits for this subplot
        x_limits, y_limits = calculate_subplot_axis_limits(probe_scores_norm, m_test)

        print(f"Row {i+1} axis limits: Monitor [{x_limits[0]:.2f}, {x_limits[1]:.2f}], Probe [{y_limits[0]:.2f}, {y_limits[1]:.2f}]")

        # Create grid with individual dynamic limits
        monitor_range = np.linspace(x_limits[0], x_limits[1], 50)  # x-axis (individual)
        probe_range = np.linspace(y_limits[0], y_limits[1], 50)    # y-axis (individual)
        M2, M1 = np.meshgrid(monitor_range, probe_range)

        # Sigmoid function (from your code)
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

        # Compute probabilities (from your code)
        p1 = sigmoid(beta_u * (M1 - u))
        p2 = sigmoid(beta_l * (M1 - l))
        m3 = alpha_1 * M1 + alpha_2 * M2
        p3 = sigmoid(beta_3 * (m3 - a))

        p_gate = (1 - p1) * p2
        p_audit = p1 + p_gate * p3

        # Compute cost (from your code)
        cost = config["c_monitor"] * p_gate + config["c_audit"] * p_audit

        # Column 6: P(audit) contour plot with scatter
        ax = axes[i, 5]
        im1 = ax.contourf(M2, M1, p_audit, levels=20, cmap='Blues')
        ax.contour(M2, M1, p_audit, levels=[0.5, 0.8, 0.95], colors='white', linewidths=1, alpha=0.5)

        # Add scatter plot of test data (use probe scores calculated above)
        harmful_mask = y_test == 1
        benign_mask = y_test == 0
        # Use raw monitor scores for consistency across all plots
        ax.scatter(m_test[benign_mask], probe_scores_norm[benign_mask], c='lightblue', s=3, alpha=0.6, label='Benign')
        ax.scatter(m_test[harmful_mask], probe_scores_norm[harmful_mask], c='red', s=3, alpha=0.8, label='Harmful')

        ax.set_xlabel('Monitor Score', fontsize=8)
        ax.set_ylabel('Probe Score', fontsize=8)
        ax.set_title('Joint P(audit)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)

        # Column 7: P(gate) contour plot with scatter
        ax = axes[i, 6]
        im2 = ax.contourf(M2, M1, p_gate, levels=20, cmap='Greens')
        ax.contour(M2, M1, p_gate, levels=[0.25, 0.5, 0.75], colors='white', linewidths=1, alpha=0.5)

        # Add scatter plot of test data
        ax.scatter(m_test[benign_mask], probe_scores_norm[benign_mask], c='lightblue', s=3, alpha=0.6, label='Benign')
        ax.scatter(m_test[harmful_mask], probe_scores_norm[harmful_mask], c='red', s=3, alpha=0.8, label='Harmful')

        ax.set_xlabel('Monitor Score', fontsize=8)
        ax.set_ylabel('Probe Score', fontsize=8)
        ax.set_title('Joint P(gate)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)

        # Column 8: Expected Cost contour plot with scatter
        ax = axes[i, 7]
        max_cost = max(config["c_monitor"] + config["c_audit"], 25)
        im3 = ax.contourf(M2, M1, cost, levels=20, cmap='Reds', vmin=0, vmax=max_cost)
        # Contour at budget value
        ax.contour(M2, M1, cost, levels=[config["budget"]], colors='blue', linewidths=3, linestyles='--')

        # Add scatter plot of test data
        ax.scatter(m_test[benign_mask], probe_scores_norm[benign_mask], c='lightblue', s=3, alpha=0.6, label='Benign')
        ax.scatter(m_test[harmful_mask], probe_scores_norm[harmful_mask], c='red', s=3, alpha=0.8, label='Harmful')

        ax.set_xlabel('Monitor Score', fontsize=8)
        ax.set_ylabel('Probe Score', fontsize=8)
        ax.set_title('Joint Expected Cost', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)

        # Columns 8-10: Baseline method decision boundaries
        # Extract baseline model parameters
        final_baseline_model = models_baseline[i]
        final_baseline_model.eval()

        with torch.no_grad():
            # Get baseline model parameters
            u_baseline = final_baseline_model.u.item()
            l_baseline = final_baseline_model.l.item()
            beta_u_baseline = final_baseline_model.beta_u.item()
            beta_l_baseline = final_baseline_model.beta_l.item()
            alpha_1_baseline = final_baseline_model.alpha_1.item()
            alpha_2_baseline = final_baseline_model.alpha_2.item()
            beta_3_baseline = final_baseline_model.beta_3.item()
            a_baseline = final_baseline_model.a.item()

        # Get baseline model probe scores for scatter plots and calculate baseline-specific axis limits
        with torch.no_grad():
            baseline_test_outputs = final_baseline_model(X_test, m_test)
            baseline_probe_scores_norm = baseline_test_outputs['probe_scores_normalized']
            baseline_monitor_scores = baseline_test_outputs['monitor_scores']

        # Calculate individual axis limits for baseline plots
        x_limits_baseline, y_limits_baseline = calculate_subplot_axis_limits(baseline_probe_scores_norm, m_test)

        print(f"Row {i+1} baseline axis limits: Monitor [{x_limits_baseline[0]:.2f}, {x_limits_baseline[1]:.2f}], Probe [{y_limits_baseline[0]:.2f}, {y_limits_baseline[1]:.2f}]")

        # Create new grid for baseline plots with baseline-specific limits
        monitor_range_baseline = np.linspace(x_limits_baseline[0], x_limits_baseline[1], 50)
        probe_range_baseline = np.linspace(y_limits_baseline[0], y_limits_baseline[1], 50)
        M2_baseline, M1_baseline = np.meshgrid(monitor_range_baseline, probe_range_baseline)

        # Compute baseline probabilities using baseline model parameters and baseline grid
        p1_baseline = sigmoid(beta_u_baseline * (M1_baseline - u_baseline))
        p2_baseline = sigmoid(beta_l_baseline * (M1_baseline - l_baseline))
        m3_baseline = alpha_1_baseline * M1_baseline + alpha_2_baseline * M2_baseline
        p3_baseline = sigmoid(beta_3_baseline * (m3_baseline - a_baseline))

        p_gate_baseline = (1 - p1_baseline) * p2_baseline
        p_audit_baseline = p1_baseline + p_gate_baseline * p3_baseline
        cost_baseline = config["c_monitor"] * p_gate_baseline + config["c_audit"] * p_audit_baseline

        # Column 8: Baseline P(audit)
        ax = axes[i, 8]
        im1_baseline = ax.contourf(M2_baseline, M1_baseline, p_audit_baseline, levels=20, cmap='Blues')
        ax.contour(M2_baseline, M1_baseline, p_audit_baseline, levels=[0.5, 0.8, 0.95], colors='white', linewidths=1, alpha=0.5)

        # Add scatter plot of test data with baseline probe scores
        ax.scatter(m_test[benign_mask], baseline_probe_scores_norm[benign_mask], c='lightblue', s=3, alpha=0.6, label='Benign')
        ax.scatter(m_test[harmful_mask], baseline_probe_scores_norm[harmful_mask], c='red', s=3, alpha=0.8, label='Harmful')

        ax.set_xlabel('Monitor Score', fontsize=8)
        ax.set_ylabel('Probe Score', fontsize=8)
        ax.set_title('Baseline P(audit)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_limits_baseline)
        ax.set_ylim(y_limits_baseline)

        # Column 9: Baseline P(gate)
        ax = axes[i, 9]
        im2_baseline = ax.contourf(M2_baseline, M1_baseline, p_gate_baseline, levels=20, cmap='Greens')
        ax.contour(M2_baseline, M1_baseline, p_gate_baseline, levels=[0.25, 0.5, 0.75], colors='white', linewidths=1, alpha=0.5)

        # Add scatter plot of test data with baseline probe scores
        ax.scatter(m_test[benign_mask], baseline_probe_scores_norm[benign_mask], c='lightblue', s=3, alpha=0.6, label='Benign')
        ax.scatter(m_test[harmful_mask], baseline_probe_scores_norm[harmful_mask], c='red', s=3, alpha=0.8, label='Harmful')

        ax.set_xlabel('Monitor Score', fontsize=8)
        ax.set_ylabel('Probe Score', fontsize=8)
        ax.set_title('Baseline P(gate)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_limits_baseline)
        ax.set_ylim(y_limits_baseline)

        # Column 10: Baseline Expected Cost
        ax = axes[i, 10]
        im3_baseline = ax.contourf(M2_baseline, M1_baseline, cost_baseline, levels=20, cmap='Reds', vmin=0, vmax=max_cost)
        ax.contour(M2_baseline, M1_baseline, cost_baseline, levels=[config["budget"]], colors='blue', linewidths=3, linestyles='--')

        # Add scatter plot of test data with baseline probe scores
        ax.scatter(m_test[benign_mask], baseline_probe_scores_norm[benign_mask], c='lightblue', s=3, alpha=0.6, label='Benign')
        ax.scatter(m_test[harmful_mask], baseline_probe_scores_norm[harmful_mask], c='red', s=3, alpha=0.8, label='Harmful')

        ax.set_xlabel('Monitor Score', fontsize=8)
        ax.set_ylabel('Probe Score', fontsize=8)
        ax.set_title('Baseline Expected Cost', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_limits_baseline)
        ax.set_ylim(y_limits_baseline)

        # Add phase boundaries and labels only for loss plots (columns 0-4)
        for j in range(5):  # Only loss plots need phase boundaries
            ax = axes[i, j]
            ax.axvline(x=n_epochs, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=2*n_epochs, color='gray', linestyle='--', alpha=0.5)

            # Only add x-label to bottom row for loss plots (columns 0-4)
            if i == n_combinations - 1:
                ax.set_xlabel('Epoch')


        # Remove row labels as requested

    # Add legend at bottom of figure
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linestyle='-', linewidth=0.8, marker='o', markersize=3, label='Joint (test)'),
        Line2D([0], [0], color='blue', linestyle=':', linewidth=0.8, marker='s', markersize=1.5, alpha=0.7, label='Joint (train)'),
        Line2D([0], [0], color='red', linestyle='-', linewidth=0.8, marker='o', markersize=3, label='Baseline (test)'),
        Line2D([0], [0], color='red', linestyle=':', linewidth=0.8, marker='s', markersize=1.5, alpha=0.7, label='Baseline (train)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.01))

    plt.subplots_adjust(hspace=0.4, wspace=0.3, left=0.05, right=0.98, top=0.92, bottom=0.08)

    # Save and show
    import os
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure plots directory exists
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Adjust subplot spacing to ensure all plots are visible
    plt.subplots_adjust(left=0.05, bottom=0.08, right=0.98, top=0.92, wspace=0.3, hspace=0.4)

    filename = f"modular_curriculum_{timestamp}.png"
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')  # bbox_inches='tight' trims whitespace
    print(f"\nPlot saved as {filepath}")

    plt.show()


def run_modular_curriculum_comparison():
    """Run comparison of different loss function combinations."""

    config = {
        "experiment_type": "modular_curriculum",
        "n_samples": 2000,
        "n_features": 500,
        "harmful_ratio": 0.2,
        "signal_strength": 0.3,
        "c_probe": 0.000001,
        "c_monitor": 3.0,
        "c_audit": 20.0,
        "budget": 1.0,  # Budget per benign example
        "lambda_cost": 1.0,
        "n_epochs": 20,
        "lr": 0.005,  # Slightly lower LR to prevent overfitting
        "weight_decay": 0.05,  # Much stronger L2 regularization
        "seed": 42
    }

    # Different loss function combinations to test
    loss_combinations = [
        ("log_likelihood", "quadratic"),
        ("linear", "quadratic"),
        ("exponential", "exponential_quadratic"),
        ("focal", "huber"),
    ]

    print("=== MODULAR CURRICULUM LEARNING COMPARISON ===")
    print(f"Testing {len(loss_combinations)} loss function combinations:")
    for i, (recall_type, cost_type) in enumerate(loss_combinations):
        print(f"  {i+1}. Recall: {recall_type}, Cost: {cost_type}")
    print()

    # Generate dataset once for all experiments
    features, labels, monitor_scores = generate_hard_dataset(
        n_samples=config["n_samples"],
        n_features=config["n_features"],
        harmful_ratio=config["harmful_ratio"],
        signal_strength=config["signal_strength"],
        seed=config["seed"]
    )

    # Split data
    X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
        features, labels, monitor_scores, test_size=0.3, random_state=42, stratify=labels
    )

    histories = []
    models_joint = []
    models_baseline = []

    # Run experiments for each loss combination
    for i, (recall_type, cost_type) in enumerate(loss_combinations):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {i+1}: {recall_type} + {cost_type}")
        print(f"{'='*60}")

        history, model_joint, model_baseline = train_three_phase_curriculum(
            X_train, y_train, m_train, X_test, y_test, m_test,
            config, recall_type, cost_type
        )

        histories.append(history)
        models_joint.append(model_joint)
        models_baseline.append(model_baseline)

        # Print final results
        final_joint_acc = history['joint_test_accuracy'][-1]
        final_joint_recall = history['joint_test_recall'][-1]
        final_joint_cost = history['joint_test_cost'][-1]

        final_baseline_acc = history['baseline_test_accuracy'][-1]
        final_baseline_recall = history['baseline_test_recall'][-1]
        final_baseline_cost = history['baseline_test_cost'][-1]

        print(f"\nFinal Results:")
        print(f"  Joint     - Acc: {final_joint_acc:.3f}, Recall: {final_joint_recall:.3f}, Cost: ${final_joint_cost:.2f}")
        print(f"  Baseline  - Acc: {final_baseline_acc:.3f}, Recall: {final_baseline_recall:.3f}, Cost: ${final_baseline_cost:.2f}")
        print(f"  Δ Recall  = {final_joint_recall - final_baseline_recall:+.3f}")
        print(f"  Δ Cost    = ${final_joint_cost - final_baseline_cost:+.2f}")

    # Generate comparison plot
    plot_modular_curriculum_results(histories, loss_combinations, config,
                                    models_joint, models_baseline, X_test, y_test, m_test)

    return histories, models_joint, models_baseline, loss_combinations


if __name__ == "__main__":
    histories, models_joint, models_baseline, loss_combinations = run_modular_curriculum_comparison()