#!/usr/bin/env python3
"""
Fair 3-phase curriculum learning comparison:

Baseline (3 phases):
- Phase 1: n epochs training probe for standalone accuracy
- Phase 2: n epochs training probe for standalone accuracy (continues improving)
- Phase 3: n epochs training gating params for recall+cost (probe frozen)

Ours (2 phases):
- Phase 1: n epochs training probe for standalone accuracy
- Phase 2: n epochs training probe + gating params jointly for recall+cost

Where n1 = n2 = n3 = n for equal parameter update budget.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from experiment_utils import ExperimentLogger
from training_comparison import HardHierarchicalMonitor, generate_hard_dataset, recall_cost_loss

def run_three_phase_curriculum():
    config = {
        "experiment_type": "three_phase_curriculum",
        "n_samples": 1000,    # Much larger dataset for smooth curves
        "n_features": 500,    # High dimensional features
        "harmful_ratio": 0.2,
        "signal_strength": 0.3,
        "c_probe": 0.01,      # Probe cost (small but not negligible)
        "c_monitor": 1.0,     # Black box monitor cost (from LW post)
        "c_audit": 20.0,      # Audit cost (from LW post)
        "budget": 1.0,        # Budget per sample (from LW post examples)
        "lambda_cost": 1.0,
        "n_epochs": 20,       # Keep same number of epochs
        "lr": 0.001,          # Slightly higher LR for larger dataset
        "weight_decay": 0.01, # L2 regularization to prevent overfitting
        "seed": 42
    }

    with ExperimentLogger("three_phase_curriculum", "3-phase fair curriculum learning", config) as logger:
        print("=== THREE-PHASE CURRICULUM LEARNING ===")
        print(f"n1 = n2 = n3 = {config['n_epochs']} epochs")
        print()

        # Generate dataset
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

        # Create models
        model_joint = HardHierarchicalMonitor(config["n_features"])
        model_baseline = HardHierarchicalMonitor(config["n_features"])
        model_baseline.load_state_dict(model_joint.state_dict())

        n = config["n_epochs"]
        history = {
            # Timeline
            'phase1_epochs': list(range(0, n)),
            'phase2_epochs': list(range(n, 2*n)),
            'phase3_epochs': list(range(2*n, 3*n)),

            # Test accuracy tracking
            'joint_phase1_acc_test': [],
            'joint_phase2_acc_test': [],
            'baseline_phase1_acc_test': [],
            'baseline_phase2_acc_test': [],
            'baseline_phase3_acc_test': [],

            # Train accuracy tracking
            'joint_phase1_acc_train': [],
            'joint_phase2_acc_train': [],
            'baseline_phase1_acc_train': [],
            'baseline_phase2_acc_train': [],
            'baseline_phase3_acc_train': [],

            # Test loss tracking
            'joint_phase1_losses_test': [],
            'joint_phase2_losses_test': [],
            'baseline_phase1_losses_test': [],
            'baseline_phase2_losses_test': [],
            'baseline_phase3_losses_test': [],

            # Train loss tracking
            'joint_phase1_losses_train': [],
            'joint_phase2_losses_train': [],
            'baseline_phase1_losses_train': [],
            'baseline_phase2_losses_train': [],
            'baseline_phase3_losses_train': []
        }

        criterion_bce = nn.BCEWithLogitsLoss()

        print("=== PHASE 1: SHARED PROBE TRAINING ===")
        print("Both methods: Train probe for classification accuracy")

        # Create optimizers for Phase 1
        joint_opt = optim.Adam(model_joint.get_probe_parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        baseline_opt = optim.Adam(model_baseline.get_probe_parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

        # Phase 1: Both models train identically for classification
        for epoch in range(n):
            # Train joint model
            model_joint.train()
            joint_opt.zero_grad()
            joint_out = model_joint(X_train, m_train)
            joint_loss = criterion_bce(joint_out['probe_scores'], y_train.float())
            joint_loss.backward()
            joint_opt.step()

            # Train baseline model identically
            model_baseline.train()
            baseline_opt.zero_grad()
            baseline_out = model_baseline(X_train, m_train)
            baseline_loss = criterion_bce(baseline_out['probe_scores'], y_train.float())
            baseline_loss.backward()
            baseline_opt.step()

            # Evaluate test accuracy
            joint_acc_test = evaluate_probe_accuracy(model_joint, X_test, y_test, m_test)
            baseline_acc_test = evaluate_probe_accuracy(model_baseline, X_test, y_test, m_test)

            # Evaluate train accuracy
            joint_acc_train = evaluate_probe_accuracy(model_joint, X_train, y_train, m_train)
            baseline_acc_train = evaluate_probe_accuracy(model_baseline, X_train, y_train, m_train)

            # Evaluate recall+cost on test set
            joint_recall_cost_test = evaluate_recall_cost(model_joint, X_test, y_test, m_test, config)
            baseline_recall_cost_test = evaluate_recall_cost(model_baseline, X_test, y_test, m_test, config)

            # Evaluate recall+cost on train set
            joint_recall_cost_train = evaluate_recall_cost(model_joint, X_train, y_train, m_train, config)
            baseline_recall_cost_train = evaluate_recall_cost(model_baseline, X_train, y_train, m_train, config)

            # Store test performance
            history['joint_phase1_acc_test'].append(joint_acc_test)
            history['baseline_phase1_acc_test'].append(baseline_acc_test)
            history['joint_phase1_losses_test'].append(joint_recall_cost_test)
            history['baseline_phase1_losses_test'].append(baseline_recall_cost_test)

            # Store train performance
            history['joint_phase1_acc_train'].append(joint_acc_train)
            history['baseline_phase1_acc_train'].append(baseline_acc_train)
            history['joint_phase1_losses_train'].append(joint_recall_cost_train)
            history['baseline_phase1_losses_train'].append(baseline_recall_cost_train)

            print(f"  Epoch {epoch}: Joint_test={joint_acc_test:.4f}, Baseline_test={baseline_acc_test:.4f}")

        # Ensure identical after Phase 1
        model_baseline.load_state_dict(model_joint.state_dict())
        print(f"Phase 1 complete - models synchronized")
        print()

        print("=== PHASE 2: DIVERGENT TRAINING ===")
        print("Joint: Train probe + gates on recall+cost")
        print("Baseline: Continue training probe on classification")

        # Create optimizers once for Phase 2
        joint_opt = optim.Adam(model_joint.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        baseline_opt = optim.Adam(model_baseline.get_probe_parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

        # Phase 2: Joint trains on recall+cost, Baseline continues classification
        for epoch in range(n):
            # Joint method: train probe + gates on recall+cost
            model_joint.train()
            joint_opt.zero_grad()
            joint_out = model_joint(X_train, m_train)
            joint_costs = config["c_probe"] + joint_out['p_gate'] * config["c_monitor"] + joint_out['p_audit'] * config["c_audit"]
            joint_loss_dict = recall_cost_loss(joint_out['p_audit'], y_train, joint_costs, config["budget"], config["lambda_cost"])
            joint_loss_dict['total_loss'].backward()
            joint_opt.step()

            # Baseline method: continue training probe on classification
            model_baseline.train()
            baseline_opt.zero_grad()
            baseline_out = model_baseline(X_train, m_train)
            baseline_loss = criterion_bce(baseline_out['probe_scores'], y_train.float())
            baseline_loss.backward()
            baseline_opt.step()

            # Evaluate test accuracy
            joint_acc_test = evaluate_probe_accuracy(model_joint, X_test, y_test, m_test)
            baseline_acc_test = evaluate_probe_accuracy(model_baseline, X_test, y_test, m_test)

            # Evaluate train accuracy
            joint_acc_train = evaluate_probe_accuracy(model_joint, X_train, y_train, m_train)
            baseline_acc_train = evaluate_probe_accuracy(model_baseline, X_train, y_train, m_train)

            # Evaluate recall+cost on test set
            joint_recall_cost_test = evaluate_recall_cost(model_joint, X_test, y_test, m_test, config)
            baseline_recall_cost_test = evaluate_recall_cost(model_baseline, X_test, y_test, m_test, config)

            # Evaluate recall+cost on train set (for baseline)
            baseline_recall_cost_train = evaluate_recall_cost(model_baseline, X_train, y_train, m_train, config)

            # Store test performance
            history['joint_phase2_acc_test'].append(joint_acc_test)
            history['baseline_phase2_acc_test'].append(baseline_acc_test)
            history['joint_phase2_losses_test'].append(joint_recall_cost_test)
            history['baseline_phase2_losses_test'].append(baseline_recall_cost_test)

            # Store train performance
            history['joint_phase2_acc_train'].append(joint_acc_train)
            history['baseline_phase2_acc_train'].append(baseline_acc_train)

            # Evaluate recall+cost on train set for both methods (consistent evaluation)
            joint_recall_cost_train = evaluate_recall_cost(model_joint, X_train, y_train, m_train, config)

            # Store train performance using same evaluation function for both
            history['joint_phase2_losses_train'].append(joint_recall_cost_train)
            history['baseline_phase2_losses_train'].append(baseline_recall_cost_train)

            print(f"  Epoch {n + epoch}: Joint_test={joint_acc_test:.4f}, Baseline_test={baseline_acc_test:.4f}")

        print(f"Phase 2 complete")
        print()

        print("=== PHASE 3: BASELINE GATING TRAINING ===")
        print("Baseline: Train gates on recall+cost (probe frozen)")

        # Freeze baseline probe parameters for Phase 3
        for param in model_baseline.get_probe_parameters():
            param.requires_grad = False

        # Create optimizer for Phase 3 (gates only)
        baseline_gate_opt = optim.Adam(model_baseline.get_gate_parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

        # Phase 3: Baseline trains gates only
        for epoch in range(n):
            # Train only gates on recall+cost
            model_baseline.train()
            baseline_gate_opt.zero_grad()
            baseline_out = model_baseline(X_train, m_train)
            baseline_costs = config["c_probe"] + baseline_out['p_gate'] * config["c_monitor"] + baseline_out['p_audit'] * config["c_audit"]
            baseline_loss_dict = recall_cost_loss(baseline_out['p_audit'], y_train, baseline_costs, config["budget"], config["lambda_cost"])
            baseline_loss_dict['total_loss'].backward()
            baseline_gate_opt.step()

            # Evaluate test accuracy
            baseline_acc_test = evaluate_probe_accuracy(model_baseline, X_test, y_test, m_test)

            # Evaluate train accuracy
            baseline_acc_train = evaluate_probe_accuracy(model_baseline, X_train, y_train, m_train)

            # Evaluate recall+cost on test set
            baseline_recall_cost_test = evaluate_recall_cost(model_baseline, X_test, y_test, m_test, config)

            # Store test performance
            history['baseline_phase3_acc_test'].append(baseline_acc_test)
            history['baseline_phase3_losses_test'].append(baseline_recall_cost_test)

            # Evaluate recall+cost on train set (consistent evaluation)
            baseline_recall_cost_train = evaluate_recall_cost(model_baseline, X_train, y_train, m_train, config)

            # Store train performance using same evaluation function as test
            history['baseline_phase3_acc_train'].append(baseline_acc_train)
            history['baseline_phase3_losses_train'].append(baseline_recall_cost_train)

            print(f"  Epoch {2*n + epoch}: Baseline_test={baseline_acc_test:.4f} (probe frozen)")

        # Re-enable gradients for probe parameters
        for param in model_baseline.get_probe_parameters():
            param.requires_grad = True

        print("Phase 3 complete")
        print()

        # Save and plot
        logger.save_data(history, 'training_history')
        plot_three_phase_curriculum(history, config, logger)

        return model_joint, model_baseline, history

def evaluate_probe_accuracy(model, X_test, y_test, m_test):
    """Evaluate probe classification accuracy."""
    model.eval()
    with torch.no_grad():
        test_out = model(X_test, m_test)
        test_probs = torch.sigmoid(test_out['probe_scores'])
        test_preds = (test_probs > 0.5).float()
        accuracy = (test_preds == y_test).float().mean().item()
    return accuracy

def evaluate_recall_cost(model, X_train, y_train, m_train, config):
    """Evaluate recall+cost metrics without training."""
    model.eval()
    with torch.no_grad():
        outputs = model(X_train, m_train)
        costs = config["c_probe"] + outputs['p_gate'] * config["c_monitor"] + outputs['p_audit'] * config["c_audit"]
        loss_dict = recall_cost_loss(outputs['p_audit'], y_train, costs, config["budget"], config["lambda_cost"])

        return {
            'total': loss_dict['total_loss'].item(),
            'recall': loss_dict['recall_loss'].item(),
            'cost': loss_dict['cost_loss'].item()
        }

def plot_three_phase_curriculum(history, config, logger=None):
    """Plot 3-phase curriculum learning with train/test split: dotted (train), solid (test).

    Loss Definitions:
    - Classification Error: 1 - accuracy (binary classification)
    - Total Loss: -log P(audit|harmful) + λ * exp((cost/budget - 1)²)
    - Recall Loss: -log P(audit|harmful) (negative log probability of auditing harmful samples)
    - Cost Loss: exp((cost/budget - 1)²) (exponential penalty for exceeding budget)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    n = config["n_epochs"]

    # Create epoch ranges for each phase
    phase1_epochs = list(range(0, n))
    phase2_epochs = list(range(n, 2*n))
    phase3_epochs = list(range(2*n, 3*n))

    # Convert accuracy to error (1 - accuracy)
    def to_error(acc_list):
        return [1.0 - x for x in acc_list]

    # Extract accuracy data
    # Test data (solid lines)
    joint_p1_acc_test = to_error(history['joint_phase1_acc_test'])
    joint_p2_acc_test = to_error(history['joint_phase2_acc_test'])
    baseline_p1_acc_test = to_error(history['baseline_phase1_acc_test'])
    baseline_p2_acc_test = to_error(history['baseline_phase2_acc_test'])
    baseline_p3_acc_test = to_error(history['baseline_phase3_acc_test'])

    # Train data (dotted lines)
    joint_p1_acc_train = to_error(history['joint_phase1_acc_train'])
    joint_p2_acc_train = to_error(history['joint_phase2_acc_train'])
    baseline_p1_acc_train = to_error(history['baseline_phase1_acc_train'])
    baseline_p2_acc_train = to_error(history['baseline_phase2_acc_train'])
    baseline_p3_acc_train = to_error(history['baseline_phase3_acc_train'])

    # Extract loss data
    def extract_loss(loss_list, key, flip_recall=False):
        if not loss_list:
            return []
        values = [d[key] for d in loss_list]
        if flip_recall and key == 'recall':
            return [-v for v in values]
        return values

    # Test losses (solid lines)
    joint_p1_total_test = extract_loss(history['joint_phase1_losses_test'], 'total')
    joint_p2_total_test = extract_loss(history['joint_phase2_losses_test'], 'total')
    baseline_p1_total_test = extract_loss(history['baseline_phase1_losses_test'], 'total')
    baseline_p2_total_test = extract_loss(history['baseline_phase2_losses_test'], 'total')
    baseline_p3_total_test = extract_loss(history['baseline_phase3_losses_test'], 'total')

    joint_p1_recall_test = extract_loss(history['joint_phase1_losses_test'], 'recall', True)
    joint_p2_recall_test = extract_loss(history['joint_phase2_losses_test'], 'recall', True)
    baseline_p1_recall_test = extract_loss(history['baseline_phase1_losses_test'], 'recall', True)
    baseline_p2_recall_test = extract_loss(history['baseline_phase2_losses_test'], 'recall', True)
    baseline_p3_recall_test = extract_loss(history['baseline_phase3_losses_test'], 'recall', True)

    joint_p1_cost_test = extract_loss(history['joint_phase1_losses_test'], 'cost')
    joint_p2_cost_test = extract_loss(history['joint_phase2_losses_test'], 'cost')
    baseline_p1_cost_test = extract_loss(history['baseline_phase1_losses_test'], 'cost')
    baseline_p2_cost_test = extract_loss(history['baseline_phase2_losses_test'], 'cost')
    baseline_p3_cost_test = extract_loss(history['baseline_phase3_losses_test'], 'cost')

    # Train losses (dotted lines)
    joint_p1_total_train = extract_loss(history['joint_phase1_losses_train'], 'total')
    joint_p2_total_train = extract_loss(history['joint_phase2_losses_train'], 'total')
    baseline_p1_total_train = extract_loss(history['baseline_phase1_losses_train'], 'total')
    baseline_p2_total_train = extract_loss(history['baseline_phase2_losses_train'], 'total')
    baseline_p3_total_train = extract_loss(history['baseline_phase3_losses_train'], 'total')

    joint_p1_recall_train = extract_loss(history['joint_phase1_losses_train'], 'recall', True)
    joint_p2_recall_train = extract_loss(history['joint_phase2_losses_train'], 'recall', True)
    baseline_p1_recall_train = extract_loss(history['baseline_phase1_losses_train'], 'recall', True)
    baseline_p2_recall_train = extract_loss(history['baseline_phase2_losses_train'], 'recall', True)
    baseline_p3_recall_train = extract_loss(history['baseline_phase3_losses_train'], 'recall', True)

    joint_p1_cost_train = extract_loss(history['joint_phase1_losses_train'], 'cost')
    joint_p2_cost_train = extract_loss(history['joint_phase2_losses_train'], 'cost')
    baseline_p1_cost_train = extract_loss(history['baseline_phase1_losses_train'], 'cost')
    baseline_p2_cost_train = extract_loss(history['baseline_phase2_losses_train'], 'cost')
    baseline_p3_cost_train = extract_loss(history['baseline_phase3_losses_train'], 'cost')

    # Plot 1: Probe Classification Error
    ax = axes[0, 0]

    # Phase 1: Shared (both methods identical)
    ax.plot(phase1_epochs, joint_p1_acc_test, color='purple', linewidth=2, linestyle='-', label='Phase 1: Shared')
    ax.plot(phase1_epochs, joint_p1_acc_train, color='purple', linewidth=2, linestyle=':', alpha=0.7)

    # Phase boundaries
    ax.axvline(x=n-0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=2*n-0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)

    # Phase 2: Methods diverge
    ax.plot(phase2_epochs, joint_p2_acc_test, color='blue', linewidth=2, linestyle='-', label='Joint Method')
    ax.plot(phase2_epochs, baseline_p2_acc_test, color='red', linewidth=2, linestyle='-', label='Baseline Method')
    ax.plot(phase2_epochs, joint_p2_acc_train, color='blue', linewidth=2, linestyle=':', alpha=0.7)
    ax.plot(phase2_epochs, baseline_p2_acc_train, color='red', linewidth=2, linestyle=':', alpha=0.7)

    # Phase 3: Only baseline continues
    joint_p3_acc_test_flat = [joint_p2_acc_test[-1]] * n if joint_p2_acc_test else [0.3] * n
    joint_p3_acc_train_flat = [joint_p2_acc_train[-1]] * n if joint_p2_acc_train else [0.3] * n
    ax.plot(phase3_epochs, joint_p3_acc_test_flat, color='blue', linewidth=2, linestyle='-', alpha=0.7)
    ax.plot(phase3_epochs, joint_p3_acc_train_flat, color='blue', linewidth=2, linestyle=':', alpha=0.7)
    ax.plot(phase3_epochs, baseline_p3_acc_test, color='red', linewidth=2, linestyle='-')
    ax.plot(phase3_epochs, baseline_p3_acc_train, color='red', linewidth=2, linestyle=':')

    # Add dummy lines for legend
    ax.plot([], [], color='black', linewidth=2, linestyle='-', label='Test Performance')
    ax.plot([], [], color='black', linewidth=2, linestyle=':', alpha=0.7, label='Train Performance')

    ax.set_title('Classification Error', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error Rate')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    # Plot 2: Total Loss
    ax = axes[0, 1]

    # Phase 1 - test (solid) and train (dotted)
    ax.plot(phase1_epochs, joint_p1_total_test, color='purple', linewidth=2, linestyle='-', )
    ax.plot(phase1_epochs, joint_p1_total_train, color='purple', linewidth=2, linestyle=':', )

    ax.axvline(x=n-0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=2*n-0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)

    # Phase 2 - test (solid) and train (dotted)
    ax.plot(phase2_epochs, joint_p2_total_test, color='blue', linewidth=2, linestyle='-', )
    ax.plot(phase2_epochs, baseline_p2_total_test, color='red', linewidth=2, linestyle='-', )
    ax.plot(phase2_epochs, joint_p2_total_train, color='blue', linewidth=2, linestyle=':', )
    ax.plot(phase2_epochs, baseline_p2_total_train, color='red', linewidth=2, linestyle=':', )

    # Phase 3 - test (solid) and train (dotted)
    joint_p3_total_test_flat = [joint_p2_total_test[-1]] * n if joint_p2_total_test else [1.5] * n
    joint_p3_total_train_flat = [joint_p2_total_train[-1]] * n if joint_p2_total_train else [1.5] * n
    ax.plot(phase3_epochs, joint_p3_total_test_flat, color='blue', linewidth=2, linestyle='-', alpha=0.7)
    ax.plot(phase3_epochs, joint_p3_total_train_flat, color='blue', linewidth=2, linestyle=':', alpha=0.7)
    ax.plot(phase3_epochs, baseline_p3_total_test, color='red', linewidth=2, linestyle='-')
    ax.plot(phase3_epochs, baseline_p3_total_train, color='red', linewidth=2, linestyle=':')

    ax.set_title('Total Loss (Recall + Cost)', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)

    # Plot 3: Recall Performance (higher = better)
    ax = axes[1, 0]

    # Phase 1 - test (solid) and train (dotted)
    ax.plot(phase1_epochs, joint_p1_recall_test, color='purple', linewidth=2, linestyle='-', )
    ax.plot(phase1_epochs, joint_p1_recall_train, color='purple', linewidth=2, linestyle=':', )

    ax.axvline(x=n-0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=2*n-0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)

    # Phase 2 - test (solid) and train (dotted)
    ax.plot(phase2_epochs, joint_p2_recall_test, color='blue', linewidth=2, linestyle='-', )
    ax.plot(phase2_epochs, baseline_p2_recall_test, color='red', linewidth=2, linestyle='-', )
    ax.plot(phase2_epochs, joint_p2_recall_train, color='blue', linewidth=2, linestyle=':', )
    ax.plot(phase2_epochs, baseline_p2_recall_train, color='red', linewidth=2, linestyle=':', )

    # Phase 3 - test (solid) and train (dotted)
    joint_p3_recall_test_flat = [joint_p2_recall_test[-1]] * n if joint_p2_recall_test else [-0.04] * n
    joint_p3_recall_train_flat = [joint_p2_recall_train[-1]] * n if joint_p2_recall_train else [-0.04] * n
    ax.plot(phase3_epochs, joint_p3_recall_test_flat, color='blue', linewidth=2, linestyle='-', alpha=0.7)
    ax.plot(phase3_epochs, joint_p3_recall_train_flat, color='blue', linewidth=2, linestyle=':', alpha=0.7)
    ax.plot(phase3_epochs, baseline_p3_recall_test, color='red', linewidth=2, linestyle='-')
    ax.plot(phase3_epochs, baseline_p3_recall_train, color='red', linewidth=2, linestyle=':')

    ax.set_title('Recall Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall Loss')
    ax.grid(True, alpha=0.3)

    # Plot 4: Cost Performance
    ax = axes[1, 1]

    # Phase 1 - test (solid) and train (dotted)
    ax.plot(phase1_epochs, joint_p1_cost_test, color='purple', linewidth=2, linestyle='-', )
    ax.plot(phase1_epochs, joint_p1_cost_train, color='purple', linewidth=2, linestyle=':', )

    ax.axvline(x=n-0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=2*n-0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)

    # Phase 2 - test (solid) and train (dotted)
    ax.plot(phase2_epochs, joint_p2_cost_test, color='blue', linewidth=2, linestyle='-', )
    ax.plot(phase2_epochs, baseline_p2_cost_test, color='red', linewidth=2, linestyle='-', )
    ax.plot(phase2_epochs, joint_p2_cost_train, color='blue', linewidth=2, linestyle=':', )
    ax.plot(phase2_epochs, baseline_p2_cost_train, color='red', linewidth=2, linestyle=':', )

    # Phase 3 - test (solid) and train (dotted)
    joint_p3_cost_test_flat = [joint_p2_cost_test[-1]] * n if joint_p2_cost_test else [1.4] * n
    joint_p3_cost_train_flat = [joint_p2_cost_train[-1]] * n if joint_p2_cost_train else [1.4] * n
    ax.plot(phase3_epochs, joint_p3_cost_test_flat, color='blue', linewidth=2, linestyle='-', alpha=0.7)
    ax.plot(phase3_epochs, joint_p3_cost_train_flat, color='blue', linewidth=2, linestyle=':', alpha=0.7)
    ax.plot(phase3_epochs, baseline_p3_cost_test, color='red', linewidth=2, linestyle='-')
    ax.plot(phase3_epochs, baseline_p3_cost_train, color='red', linewidth=2, linestyle=':')

    ax.set_title('Cost Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost Loss')
    ax.grid(True, alpha=0.3)

    # Add loss definitions as text box
    loss_text = """Loss Definitions:
• Classification Error = 1 - accuracy
• Total Loss = Recall Loss + λ·Cost Loss (λ=1.0)
• Recall Loss = -P(audit|harmful)
• Cost Loss = exp(clamp((cost/budget - 1)², max=5))

Legend: Solid = Test, Dotted = Train"""

    fig.text(0.02, 0.02, loss_text, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if logger:
        logger.save_plot('three_phase_curriculum')
    else:
        plt.savefig('three_phase_curriculum.png', dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    run_three_phase_curriculum()