import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
from experiment_utils import ExperimentLogger

# Set style
plt.style.use('default')
sns.set_palette("husl")

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def plot_decision_boundaries(u=2.0, l=-0.5, beta_u=3.0, beta_l=3.0,
                            alpha_1=0.4, alpha_2=0.65, beta_3=3.0, a=0.1,
                            c_monitor=1.0, c_audit=20.0, budget=2.0):
    """
    Plot decision boundaries with monitor score on x-axis and probe score on y-axis.
    """
    # Create grid
    monitor_range = np.linspace(-4, 4, 100)  # x-axis
    probe_range = np.linspace(-4, 4, 100)    # y-axis
    M2, M1 = np.meshgrid(monitor_range, probe_range)

    # Compute probabilities
    p1 = sigmoid(beta_u * (M1 - u))
    p2 = sigmoid(beta_l * (M1 - l))
    m3 = alpha_1 * M1 + alpha_2 * M2
    p3 = sigmoid(beta_3 * (m3 - a))

    p_gate = (1 - p1) * p2
    p_audit = p1 + p_gate * p3

    # Compute cost
    cost = c_monitor * p_gate + c_audit * p_audit

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: P(audit)
    ax1 = axes[0]
    im1 = ax1.contourf(M2, M1, p_audit, levels=20, cmap='Blues')
    ax1.axhline(y=u, color='red', linestyle='--', linewidth=2, label=f'u={u:.1f}')
    ax1.axhline(y=l, color='green', linestyle='--', linewidth=2, label=f'l={l:.1f}')

    # Add ensemble boundary
    if alpha_1 != 0:
        monitor_line = np.linspace(-4, 4, 100)
        probe_line = (a - alpha_2 * monitor_line) / alpha_1
        mask = (probe_line >= -4) & (probe_line <= 4)
        ax1.plot(monitor_line[mask], probe_line[mask], 'orange', linestyle=':',
                linewidth=2, label=f'm₃={a:.1f}')

    ax1.contour(M2, M1, p_audit, levels=[0.5, 0.8, 0.95], colors='white', linewidths=1, alpha=0.5)
    ax1.set_xlabel('Monitor Score (m₂)')
    ax1.set_ylabel('Probe Score (m₁)')
    ax1.set_title('P(audit) - Total Audit Probability')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-4, 4])
    ax1.set_ylim([-4, 4])
    plt.colorbar(im1, ax=ax1, label='P(audit)')

    # Plot 2: P(gate)
    ax2 = axes[1]
    im2 = ax2.contourf(M2, M1, p_gate, levels=20, cmap='Greens')
    ax2.axhline(y=u, color='red', linestyle='--', linewidth=2)
    ax2.axhline(y=l, color='green', linestyle='--', linewidth=2)
    ax2.contour(M2, M1, p_gate, levels=[0.25, 0.5, 0.75], colors='white', linewidths=1, alpha=0.5)
    ax2.set_xlabel('Monitor Score (m₂)')
    ax2.set_ylabel('Probe Score (m₁)')
    ax2.set_title('P(gate) - Monitor Call Probability')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-4, 4])
    ax2.set_ylim([-4, 4])
    plt.colorbar(im2, ax=ax2, label='P(gate)')

    # Plot 3: Expected Cost
    ax3 = axes[2]
    max_cost = max(c_monitor + c_audit, 25)
    im3 = ax3.contourf(M2, M1, cost, levels=20, cmap='Reds', vmin=0, vmax=max_cost)
    ax3.axhline(y=u, color='red', linestyle='--', linewidth=2)
    ax3.axhline(y=l, color='green', linestyle='--', linewidth=2)

    # Add ensemble boundary
    if alpha_1 != 0:
        monitor_line = np.linspace(-4, 4, 100)
        probe_line = (a - alpha_2 * monitor_line) / alpha_1
        mask = (probe_line >= -4) & (probe_line <= 4)
        ax3.plot(monitor_line[mask], probe_line[mask], 'orange', linestyle=':',
                linewidth=2, label=f'm₃={a:.1f}')

    # Contour at budget value
    ax3.contour(M2, M1, cost, levels=[budget], colors='blue', linewidths=3, linestyles='--')
    contour_levels = [budget/2, budget*2, budget*3]
    ax3.contour(M2, M1, cost, levels=contour_levels, colors='black', linewidths=0.5, alpha=0.5)

    ax3.set_xlabel('Monitor Score (m₂)')
    ax3.set_ylabel('Probe Score (m₁)')
    ax3.set_title(f'Expected Cost (Budget=${budget:.1f})')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-4, 4])
    ax3.set_ylim([-4, 4])
    plt.colorbar(im3, ax=ax3, label='Cost ($)')

    plt.suptitle('Hierarchical Monitoring System - Decision Boundaries', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

    print(f"Gate parameters: u={u:.2f}, l={l:.2f}, β_u={beta_u:.1f}, β_l={beta_l:.1f}")
    print(f"Ensemble parameters: α₁={alpha_1:.2f}, α₂={alpha_2:.2f}, β₃={beta_3:.1f}, a={a:.2f}")
    print(f"Cost structure: Monitor=${c_monitor:.1f}, Audit=${c_audit:.1f}, Budget=${budget:.1f}")


class CurriculumHierarchicalMonitor(nn.Module):
    """Hierarchical monitor with separate probe and gate parameters for curriculum learning."""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        # Probe network (for standalone classification)
        self.probe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Gate parameters (learnable thresholds and slopes)
        self.u = nn.Parameter(torch.tensor(2.0))  # upper threshold
        self.l = nn.Parameter(torch.tensor(-0.5))  # lower threshold
        self.beta_u = nn.Parameter(torch.tensor(3.0))  # upper slope
        self.beta_l = nn.Parameter(torch.tensor(3.0))  # lower slope

        # Ensemble parameters
        self.alpha_1 = nn.Parameter(torch.tensor(0.4))  # probe weight
        self.alpha_2 = nn.Parameter(torch.tensor(0.65))  # monitor weight
        self.beta_3 = nn.Parameter(torch.tensor(3.0))   # ensemble slope
        self.a = nn.Parameter(torch.tensor(0.1))        # ensemble threshold

    def forward(self, features, monitor_scores):
        """
        Forward pass through hierarchical system.

        Args:
            features: Input features for probe
            monitor_scores: Pre-computed expensive monitor scores

        Returns:
            Dictionary with probabilities and costs
        """
        # Get probe scores
        m1 = self.probe(features).squeeze()
        m2 = monitor_scores

        # Gate probabilities
        p1 = torch.sigmoid(self.beta_u * (m1 - self.u))  # P(immediate audit)
        p2 = torch.sigmoid(self.beta_l * (m1 - self.l))  # P(monitor call)

        # Ensemble score and probability
        m3 = self.alpha_1 * m1 + self.alpha_2 * m2
        p3 = torch.sigmoid(self.beta_3 * (m3 - self.a))  # P(audit | monitor called)

        # Final probabilities
        p_gate = (1 - p1) * p2  # P(monitor called)
        p_audit = p1 + p_gate * p3  # P(total audit)

        return {
            'probe_scores': m1,
            'monitor_scores': m2,
            'ensemble_scores': m3,
            'p_immediate': p1,
            'p_gate': p_gate,
            'p_ensemble': p3,
            'p_audit': p_audit
        }

    def get_probe_parameters(self):
        """Get only probe parameters for stage 1 training."""
        return self.probe.parameters()

    def get_gate_parameters(self):
        """Get only gate parameters for stage 2 training."""
        return [self.u, self.l, self.beta_u, self.beta_l, self.alpha_1, self.alpha_2, self.beta_3, self.a]


def recall_cost_loss(p_audit, labels, costs, budget, lambda_cost=1.0):
    """
    Combined recall-cost loss function.

    Recall loss: -P(audit|harmful) (maximize recall on harmful examples)
    Cost loss: exp((cost/budget - 1)^2) (penalize cost deviations from budget)
    """
    harmful_mask = (labels == 1)

    # Recall loss: maximize P(audit|harmful)
    if harmful_mask.any():
        recall_loss = -p_audit[harmful_mask].mean()
    else:
        recall_loss = torch.tensor(0.0)

    # Cost loss: penalize deviation from budget
    avg_cost = costs.mean()
    cost_ratio = avg_cost / budget
    cost_loss = torch.exp((cost_ratio - 1) ** 2)

    total_loss = recall_loss + lambda_cost * cost_loss

    return {
        'total_loss': total_loss,
        'recall_loss': recall_loss,
        'cost_loss': cost_loss,
        'avg_cost': avg_cost.item(),
        'cost_ratio': cost_ratio.item()
    }


def generate_curriculum_dataset(n_samples=1000, n_features=100, harmful_ratio=0.15,
                               signal_strength=0.2, seed=42):
    """Generate challenging dataset for curriculum learning."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_harmful = int(n_samples * harmful_ratio)
    n_benign = n_samples - n_harmful

    # Generate base features
    features = torch.randn(n_samples, n_features)
    labels = torch.zeros(n_samples)
    labels[:n_harmful] = 1

    # Add signal to harmful examples (weak signal)
    harmful_direction = torch.randn(n_features)
    harmful_direction = harmful_direction / torch.norm(harmful_direction)

    for i in range(n_harmful):
        noise_scale = np.random.uniform(0.8, 1.2)
        features[i] += signal_strength * noise_scale * harmful_direction

        # Add noise to random dimensions
        random_dims = torch.randperm(n_features)[:20]
        features[i, random_dims] += torch.randn(20) * signal_strength * 0.3

    # Make 10% of examples have completely random features but monitor detects them
    n_random = int(n_samples * 0.1)
    random_indices = torch.randperm(n_samples)[:n_random]
    for i in random_indices:
        features[i] = torch.randn(n_features)

    # Shuffle dataset
    indices = torch.randperm(n_samples)
    features = features[indices]
    labels = labels[indices]

    # Track which were random
    was_random = torch.zeros(n_samples, dtype=bool)
    for i, orig_idx in enumerate(indices):
        was_random[i] = orig_idx in random_indices

    # Generate monitor scores (expensive but good)
    monitor_scores = torch.randn(n_samples) * 0.3
    for i in range(n_samples):
        if labels[i] == 1:  # Harmful
            if was_random[i]:
                # Monitor has very good accuracy on random features
                monitor_scores[i] += 1.2 + torch.randn(1).item() * 0.2
            else:
                # Normal harmful examples
                monitor_scores[i] += 0.8 + torch.randn(1).item() * 0.3
        else:  # Benign
            monitor_scores[i] += torch.randn(1).item() * 0.15

    return features, labels, monitor_scores


def train_curriculum(features, labels, monitor_scores, c_monitor=1.0, c_audit=20.0,
                    budget=2.0, lambda_cost=1.0, epochs_stage1=100, epochs_stage2=50,
                    epochs_joint=50, lr=0.001):
    """
    Curriculum learning:
    1. Train probe for standalone classification
    2. Train gate parameters while freezing probe
    3. Joint fine-tuning of both
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split data
    X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
        features, labels, monitor_scores, test_size=0.3, random_state=42, stratify=labels
    )

    # Convert to tensors
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)
    m_train = m_train.to(device)
    m_test = m_test.to(device)

    model = CurriculumHierarchicalMonitor(features.shape[1]).to(device)

    # Track training history
    history = {
        'stage1_losses': [],
        'stage2_losses': [],
        'joint_losses': [],
        'stage1_accuracy': [],
        'stage2_metrics': [],
        'joint_metrics': []
    }

    print("=== STAGE 1: Training Probe for Standalone Classification ===")

    # Stage 1: Train only probe parameters
    probe_optimizer = optim.Adam(model.get_probe_parameters(), lr=lr)
    criterion_bce = nn.BCEWithLogitsLoss()

    for epoch in range(epochs_stage1):
        model.train()
        probe_optimizer.zero_grad()

        # Forward pass (only need probe scores)
        outputs = model(X_train, m_train)
        probe_logits = outputs['probe_scores']

        # Binary classification loss
        loss = criterion_bce(probe_logits, y_train.float())
        loss.backward()
        probe_optimizer.step()

        history['stage1_losses'].append(loss.item())

        # Evaluate accuracy every 20 epochs
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test, m_test)
                test_probs = torch.sigmoid(test_outputs['probe_scores'])
                test_preds = (test_probs > 0.5).float()
                accuracy = (test_preds == y_test).float().mean().item()
                history['stage1_accuracy'].append((epoch, accuracy))
                print(f"Epoch {epoch}: Loss={loss.item():.4f}, Test Accuracy={accuracy:.4f}")

    print(f"Stage 1 Complete. Final probe accuracy: {accuracy:.4f}")

    print("\n=== STAGE 2: Training Gate Parameters (Probe Frozen) ===")

    # Stage 2: Freeze probe, train gate parameters
    for param in model.probe.parameters():
        param.requires_grad = False

    gate_optimizer = optim.Adam(model.get_gate_parameters(), lr=lr)

    for epoch in range(epochs_stage2):
        model.train()
        gate_optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train, m_train)
        p_audit = outputs['p_audit']
        p_gate = outputs['p_gate']

        # Compute costs
        costs = c_monitor * p_gate + c_audit * p_audit

        # Recall-cost loss
        loss_dict = recall_cost_loss(p_audit, y_train, costs, budget, lambda_cost)
        loss = loss_dict['total_loss']

        loss.backward()
        gate_optimizer.step()

        history['stage2_losses'].append(loss_dict)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Total Loss={loss.item():.4f}, "
                  f"Recall Loss={loss_dict['recall_loss'].item():.4f}, "
                  f"Cost Loss={loss_dict['cost_loss'].item():.4f}, "
                  f"Avg Cost=${loss_dict['avg_cost']:.2f}")

    print("Stage 2 Complete.")

    print("\n=== STAGE 3: Joint Fine-tuning ===")

    # Stage 3: Unfreeze probe and train everything jointly
    for param in model.probe.parameters():
        param.requires_grad = True

    joint_optimizer = optim.Adam(model.parameters(), lr=lr * 0.1)  # Lower LR for fine-tuning

    for epoch in range(epochs_joint):
        model.train()
        joint_optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train, m_train)
        p_audit = outputs['p_audit']
        p_gate = outputs['p_gate']

        # Compute costs
        costs = c_monitor * p_gate + c_audit * p_audit

        # Recall-cost loss
        loss_dict = recall_cost_loss(p_audit, y_train, costs, budget, lambda_cost)
        loss = loss_dict['total_loss']

        loss.backward()
        joint_optimizer.step()

        history['joint_losses'].append(loss_dict)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Total Loss={loss.item():.4f}, "
                  f"Recall Loss={loss_dict['recall_loss'].item():.4f}, "
                  f"Cost Loss={loss_dict['cost_loss'].item():.4f}, "
                  f"Avg Cost=${loss_dict['avg_cost']:.2f}")

    print("Joint training complete!")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test, m_test)
        test_p_audit = test_outputs['p_audit']
        test_p_gate = test_outputs['p_gate']
        test_costs = c_monitor * test_p_gate + c_audit * test_p_audit

        # Compute final metrics
        harmful_mask = (y_test == 1)
        if harmful_mask.any():
            final_recall = (test_p_audit[harmful_mask] > 0.5).float().mean().item()
        else:
            final_recall = 0.0

        final_cost = test_costs.mean().item()

        print(f"\n=== FINAL RESULTS ===")
        print(f"Test Recall: {final_recall:.4f}")
        print(f"Average Cost: ${final_cost:.2f}")
        print(f"Budget: ${budget:.2f}")
        print(f"Cost Ratio: {final_cost/budget:.3f}")

    return model, history


def plot_training_history(history, logger=None):
    """Plot training curves for all three stages."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Stage 1: Probe training
    ax = axes[0, 0]
    ax.plot(history['stage1_losses'])
    ax.set_title('Stage 1: Probe BCE Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True)

    # Stage 1: Accuracy
    ax = axes[1, 0]
    if history['stage1_accuracy']:
        epochs, accs = zip(*history['stage1_accuracy'])
        ax.plot(epochs, accs, 'o-')
    ax.set_title('Stage 1: Probe Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.grid(True)

    # Stage 2: Gate training losses
    ax = axes[0, 1]
    if history['stage2_losses']:
        total_losses = [x['total_loss'].item() for x in history['stage2_losses']]
        recall_losses = [x['recall_loss'].item() for x in history['stage2_losses']]
        cost_losses = [x['cost_loss'].item() for x in history['stage2_losses']]

        ax.plot(total_losses, label='Total Loss')
        ax.plot(recall_losses, label='Recall Loss')
        ax.plot(cost_losses, label='Cost Loss')
        ax.legend()
    ax.set_title('Stage 2: Gate Training Losses')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True)

    # Stage 2: Cost tracking
    ax = axes[1, 1]
    if history['stage2_losses']:
        costs = [x['avg_cost'] for x in history['stage2_losses']]
        ax.plot(costs)
    ax.set_title('Stage 2: Average Cost')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost ($)')
    ax.grid(True)

    # Stage 3: Joint training losses
    ax = axes[0, 2]
    if history['joint_losses']:
        total_losses = [x['total_loss'].item() for x in history['joint_losses']]
        recall_losses = [x['recall_loss'].item() for x in history['joint_losses']]
        cost_losses = [x['cost_loss'].item() for x in history['joint_losses']]

        ax.plot(total_losses, label='Total Loss')
        ax.plot(recall_losses, label='Recall Loss')
        ax.plot(cost_losses, label='Cost Loss')
        ax.legend()
    ax.set_title('Stage 3: Joint Training Losses')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True)

    # Stage 3: Cost tracking
    ax = axes[1, 2]
    if history['joint_losses']:
        costs = [x['avg_cost'] for x in history['joint_losses']]
        ax.plot(costs)
    ax.set_title('Stage 3: Average Cost')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost ($)')
    ax.grid(True)

    plt.suptitle('Curriculum Learning Training History', fontsize=16)
    plt.tight_layout()

    if logger:
        logger.save_plot('training_history')

    plt.show()


def run_curriculum_experiment():
    """Run the full curriculum learning experiment."""

    # Experiment configuration
    config = {
        "experiment_type": "curriculum_learning",
        "n_samples": 2000,
        "n_features": 50,
        "harmful_ratio": 0.15,
        "signal_strength": 0.15,
        "c_monitor": 1.0,
        "c_audit": 20.0,
        "budget": 3.0,
        "lambda_cost": 1.0,
        "epochs_stage1": 100,
        "epochs_stage2": 50,
        "epochs_joint": 50,
        "lr": 0.001,
        "seed": 42
    }

    description = """
    Curriculum learning experiment for hierarchical monitoring.

    Three-stage training process:
    1. Train probe for standalone classification (BCE loss)
    2. Train gate parameters with probe frozen (recall-cost loss)
    3. Joint fine-tuning of both probe and gate parameters

    Loss function: Recall = -P(audit|harmful), Cost = exp((cost/budget - 1)^2)
    """

    with ExperimentLogger("curriculum_learning", description, config) as logger:
        print("=== CURRICULUM LEARNING EXPERIMENT ===")
        print("Generating challenging dataset...")

        features, labels, monitor_scores = generate_curriculum_dataset(
            n_samples=config["n_samples"],
            n_features=config["n_features"],
            harmful_ratio=config["harmful_ratio"],
            signal_strength=config["signal_strength"],
            seed=config["seed"]
        )

        print(f"Dataset: {len(features)} samples, {features.shape[1]} features")
        print(f"Harmful examples: {labels.sum().item():.0f} ({labels.mean().item():.1%})")

        # Save dataset
        logger.save_data({
            'features': features,
            'labels': labels,
            'monitor_scores': monitor_scores
        }, 'dataset')

        # Train with curriculum learning
        model, history = train_curriculum(
            features, labels, monitor_scores,
            c_monitor=config["c_monitor"],
            c_audit=config["c_audit"],
            budget=config["budget"],
            lambda_cost=config["lambda_cost"],
            epochs_stage1=config["epochs_stage1"],
            epochs_stage2=config["epochs_stage2"],
            epochs_joint=config["epochs_joint"],
            lr=config["lr"]
        )

        # Save training history
        logger.save_data(history, 'training_history')

        # Plot training history
        plot_training_history(history, logger)

        # Extract learned parameters
        with torch.no_grad():
            learned_params = {
                'u': model.u.item(),
                'l': model.l.item(),
                'beta_u': model.beta_u.item(),
                'beta_l': model.beta_l.item(),
                'alpha_1': model.alpha_1.item(),
                'alpha_2': model.alpha_2.item(),
                'beta_3': model.beta_3.item(),
                'a': model.a.item()
            }

        print(f"\n=== LEARNED PARAMETERS ===")
        print(f"Gate thresholds: u={learned_params['u']:.3f}, l={learned_params['l']:.3f}")
        print(f"Gate slopes: β_u={learned_params['beta_u']:.3f}, β_l={learned_params['beta_l']:.3f}")
        print(f"Ensemble weights: α₁={learned_params['alpha_1']:.3f}, α₂={learned_params['alpha_2']:.3f}")
        print(f"Ensemble params: β₃={learned_params['beta_3']:.3f}, a={learned_params['a']:.3f}")

        # Plot decision boundaries with learned parameters
        plot_decision_boundaries(
            u=learned_params['u'], l=learned_params['l'],
            beta_u=learned_params['beta_u'], beta_l=learned_params['beta_l'],
            alpha_1=learned_params['alpha_1'], alpha_2=learned_params['alpha_2'],
            beta_3=learned_params['beta_3'], a=learned_params['a'],
            c_monitor=config["c_monitor"], c_audit=config["c_audit"], budget=config["budget"]
        )
        logger.save_plot('decision_boundaries')

        # Save model and results
        logger.save_model(model, 'curriculum_model')

        # Log final results
        final_results = {
            'learned_parameters': learned_params,
            'final_loss_stage1': history['stage1_losses'][-1] if history['stage1_losses'] else None,
            'final_loss_stage2': {k: v.item() if hasattr(v, 'item') else v
                                 for k, v in history['stage2_losses'][-1].items()} if history['stage2_losses'] else None,
            'final_loss_joint': {k: v.item() if hasattr(v, 'item') else v
                                for k, v in history['joint_losses'][-1].items()} if history['joint_losses'] else None,
        }

        logger.log_results(final_results)

        return model, history


if __name__ == "__main__":
    # Run the experiment
    model, history = run_curriculum_experiment()