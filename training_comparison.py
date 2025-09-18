import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import seaborn as sns
from experiment_utils import ExperimentLogger

# Set style
plt.style.use('default')
sns.set_palette("husl")

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class HardHierarchicalMonitor(nn.Module):
    """More challenging hierarchical monitor for comparison experiments."""

    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()

        # Smaller probe network to make task harder
        self.probe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

        # Gate parameters
        self.u = nn.Parameter(torch.tensor(1.5))   # upper threshold
        self.l = nn.Parameter(torch.tensor(-1.0))  # lower threshold
        self.beta_u = nn.Parameter(torch.tensor(2.0))  # upper slope
        self.beta_l = nn.Parameter(torch.tensor(2.0))  # lower slope

        # Ensemble parameters
        self.alpha_1 = nn.Parameter(torch.tensor(0.5))  # probe weight
        self.alpha_2 = nn.Parameter(torch.tensor(0.5))  # monitor weight
        self.beta_3 = nn.Parameter(torch.tensor(2.0))   # ensemble slope
        self.a = nn.Parameter(torch.tensor(3.0))        # ensemble threshold (start high for low audit)

        # Normalization statistics (buffers, not parameters)
        self.register_buffer('probe_mean', torch.tensor(0.0))
        self.register_buffer('probe_std', torch.tensor(1.0))
        self.register_buffer('monitor_mean', torch.tensor(0.0))
        self.register_buffer('monitor_std', torch.tensor(1.0))
        self.normalization_fitted = False

    def fit_normalization(self, train_features, train_monitor_scores):
        """Fit normalization statistics on training data (standard ML practice)."""
        with torch.no_grad():
            # Get probe scores on training data
            probe_scores = self.probe(train_features).squeeze()

            # Compute normalization statistics (fit once, use throughout training)
            self.probe_mean = probe_scores.mean()
            self.probe_std = probe_scores.std() + 1e-8
            self.monitor_mean = train_monitor_scores.mean()
            self.monitor_std = train_monitor_scores.std() + 1e-8

            self.normalization_fitted = True


    def forward(self, features, monitor_scores):
        # Get raw probe logits (for BCE loss)
        probe_logits = self.probe(features).squeeze()

        # Get normalized scores (for gating decisions)
        if self.normalization_fitted:
            probe_scores_normalized = (probe_logits - self.probe_mean) / self.probe_std
            monitor_scores_normalized = (monitor_scores - self.monitor_mean) / self.monitor_std
        else:
            probe_scores_normalized = probe_logits
            monitor_scores_normalized = monitor_scores

        m1 = probe_scores_normalized
        m2 = monitor_scores_normalized

        # Gate probabilities (now acting on normalized scores)
        p1 = torch.sigmoid(self.beta_u * (m1 - self.u))  # P(immediate audit)
        p2 = torch.sigmoid(self.beta_l * (m1 - self.l))  # P(monitor call)

        # Ensemble score and probability (using normalized scores)
        m3 = self.alpha_1 * m1 + self.alpha_2 * m2
        p3 = torch.sigmoid(self.beta_3 * (m3 - self.a))  # P(audit | monitor called)

        # Final probabilities
        p_gate = (1 - p1) * p2  # P(monitor called)
        p_audit = p1 + p_gate * p3  # P(total audit)

        return {
            'probe_scores': probe_logits,  # raw logits for BCE loss
            'monitor_scores': monitor_scores_normalized,  # normalized scores
            'probe_scores_normalized': probe_scores_normalized,  # normalized scores for gating
            'ensemble_scores': m3,
            'p_immediate': p1,
            'p_gate': p_gate,
            'p_ensemble': p3,
            'p_audit': p_audit
        }

    def get_probe_parameters(self):
        return self.probe.parameters()

    def get_gate_parameters(self):
        return [self.u, self.l, self.beta_u, self.beta_l, self.alpha_1, self.alpha_2, self.beta_3, self.a]

def generate_hard_dataset(n_samples=800, n_features=32, harmful_ratio=0.2,
                         signal_strength=0.3, seed=42):
    """
    Generate dataset according to new specification:
    - y = +1 for misaligned (harmful) and -1 for aligned (benign)
    - latent suspiciousness L = y + N(0, 0.8)
    - probe suspiciousness p = L + N(0, 1.2)
    - monitor suspiciousness m = L + N(0, 0.8)
    - monitor score is m
    - feature vector = random unit vector + m * v (where v is fixed random unit vector)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_harmful = int(n_samples * harmful_ratio)
    n_benign = n_samples - n_harmful

    # Generate labels: +1 for misaligned/harmful, -1 for aligned/benign
    y = -torch.ones(n_samples)  # Start with -1 for benign
    y[:n_harmful] = 1  # First n_harmful samples are misaligned/harmful

    # Convert to 0/1 labels for compatibility with existing code
    labels = torch.zeros(n_samples)
    labels[:n_harmful] = 1  # 1 for harmful, 0 for benign

    # Generate latent suspiciousness: L = y + N(0, 0.8)
    L = y + torch.randn(n_samples) * 0.8

    # Generate probe suspiciousness: p = L + N(0, 1.2)
    probe_suspiciousness = L + torch.randn(n_samples) * 1.2

    # Generate monitor suspiciousness: m = L + N(0, 0.8)
    monitor_suspiciousness = L + torch.randn(n_samples) * 0.8

    # Monitor scores are just the monitor suspiciousness
    monitor_scores = monitor_suspiciousness

    # Generate fixed random unit vector v (same across all samples)
    v = torch.randn(n_features)
    v = v / torch.norm(v)  # normalize to unit vector

    # Generate feature vectors: random unit vector + m * v
    features = torch.zeros(n_samples, n_features)
    for i in range(n_samples):
        # Random unit vector for each sample
        random_unit = torch.randn(n_features)
        random_unit = random_unit / torch.norm(random_unit)

        # Feature vector = random unit vector + m * v
        features[i] = random_unit + monitor_scores[i] * v

    # Shuffle the dataset
    indices = torch.randperm(n_samples)
    features = features[indices]
    labels = labels[indices]
    monitor_scores = monitor_scores[indices]

    return features, labels, monitor_scores

def recall_cost_loss(p_audit, labels, costs, budget, lambda_cost=1.0):
    """Combined recall-cost loss function."""
    harmful_mask = (labels == 1)

    # Recall loss: maximize P(audit|harmful)
    if harmful_mask.any():
        recall_loss = -p_audit[harmful_mask].mean()
    else:
        recall_loss = torch.tensor(0.0)

    # Cost loss: penalize deviation from budget (more stable)
    avg_cost = costs.mean()
    cost_ratio = avg_cost / budget
    # Use clamped exponential to avoid explosions
    deviation = torch.clamp((cost_ratio - 1) ** 2, max=5.0)  # Clamp to prevent explosion
    cost_loss = torch.exp(deviation)

    total_loss = recall_loss + lambda_cost * cost_loss

    return {
        'total_loss': total_loss,
        'recall_loss': recall_loss,
        'cost_loss': cost_loss,
        'avg_cost': avg_cost.item(),
        'cost_ratio': cost_ratio.item()
    }

def train_joint_method(features, labels, monitor_scores, c_monitor=1.0, c_audit=20.0,
                      budget=3.0, lambda_cost=1.0, epochs_stage1=20, epochs_stage2=25, lr=0.002):
    """Method 1: Train probe for classification, then train probe + gates jointly."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split data
    X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
        features, labels, monitor_scores, test_size=0.3, random_state=42, stratify=labels
    )

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    m_train = m_train.to(device)

    model = HardHierarchicalMonitor(features.shape[1]).to(device)

    history = {
        'stage1_losses': [],
        'stage2_losses': [],
        'stage1_accuracy': []
    }

    print("  Stage 1: Training probe for classification accuracy")

    # Stage 1: Train probe only for classification
    probe_optimizer = optim.Adam(model.get_probe_parameters(), lr=lr)
    criterion_bce = nn.BCEWithLogitsLoss()

    for epoch in range(epochs_stage1):
        model.train()
        probe_optimizer.zero_grad()

        outputs = model(X_train, m_train)
        probe_logits = outputs['probe_scores']
        loss = criterion_bce(probe_logits, y_train.float())
        loss.backward()
        probe_optimizer.step()

        history['stage1_losses'].append(loss.item())

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test.to(device), m_test.to(device))
                test_probs = torch.sigmoid(test_outputs['probe_scores'])
                test_preds = (test_probs > 0.5).float()
                accuracy = (test_preds == y_test.to(device)).float().mean().item()
                history['stage1_accuracy'].append((epoch, accuracy))
                print(f"    Epoch {epoch}: BCE Loss={loss.item():.4f}, Accuracy={accuracy:.4f}")

    print(f"  Stage 1 complete. Final accuracy: {accuracy:.4f}")

    print("  Stage 2: Training probe + gates jointly for recall + cost")

    # Stage 2: Train probe + gates jointly for recall + cost
    joint_optimizer = optim.Adam(model.parameters(), lr=lr * 0.5)

    for epoch in range(epochs_stage2):
        model.train()
        joint_optimizer.zero_grad()

        outputs = model(X_train, m_train)
        p_audit = outputs['p_audit']
        p_gate = outputs['p_gate']
        costs = c_monitor * p_gate + c_audit * p_audit

        loss_dict = recall_cost_loss(p_audit, y_train, costs, budget, lambda_cost)
        loss = loss_dict['total_loss']

        loss.backward()
        joint_optimizer.step()

        history['stage2_losses'].append(loss_dict)

        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: Total={loss.item():.4f}, "
                  f"Recall={loss_dict['recall_loss'].item():.4f}, "
                  f"Cost={loss_dict['avg_cost']:.2f}")

    print("  Stage 2 complete.")

    return model, history

def train_frozen_method(features, labels, monitor_scores, c_monitor=1.0, c_audit=20.0,
                       budget=3.0, lambda_cost=1.0, epochs_stage1=20, epochs_stage2=25, lr=0.002):
    """Method 2: Train probe for classification, then train ONLY gates (probe frozen)."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split data
    X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
        features, labels, monitor_scores, test_size=0.3, random_state=42, stratify=labels
    )

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    m_train = m_train.to(device)

    model = HardHierarchicalMonitor(features.shape[1]).to(device)

    history = {
        'stage1_losses': [],
        'stage2_losses': [],
        'stage1_accuracy': []
    }

    print("  Stage 1: Training probe for classification accuracy")

    # Stage 1: Train probe only for classification
    probe_optimizer = optim.Adam(model.get_probe_parameters(), lr=lr)
    criterion_bce = nn.BCEWithLogitsLoss()

    for epoch in range(epochs_stage1):
        model.train()
        probe_optimizer.zero_grad()

        outputs = model(X_train, m_train)
        probe_logits = outputs['probe_scores']
        loss = criterion_bce(probe_logits, y_train.float())
        loss.backward()
        probe_optimizer.step()

        history['stage1_losses'].append(loss.item())

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test.to(device), m_test.to(device))
                test_probs = torch.sigmoid(test_outputs['probe_scores'])
                test_preds = (test_probs > 0.5).float()
                accuracy = (test_preds == y_test.to(device)).float().mean().item()
                history['stage1_accuracy'].append((epoch, accuracy))
                print(f"    Epoch {epoch}: BCE Loss={loss.item():.4f}, Accuracy={accuracy:.4f}")

    print(f"  Stage 1 complete. Final accuracy: {accuracy:.4f}")

    print("  Stage 2: Training ONLY gates (probe frozen) for recall + cost")

    # Stage 2: Freeze probe, train only gates
    for param in model.probe.parameters():
        param.requires_grad = False

    gate_optimizer = optim.Adam(model.get_gate_parameters(), lr=lr)

    for epoch in range(epochs_stage2):
        model.train()
        gate_optimizer.zero_grad()

        outputs = model(X_train, m_train)
        p_audit = outputs['p_audit']
        p_gate = outputs['p_gate']
        costs = c_monitor * p_gate + c_audit * p_audit

        loss_dict = recall_cost_loss(p_audit, y_train, costs, budget, lambda_cost)
        loss = loss_dict['total_loss']

        loss.backward()
        gate_optimizer.step()

        history['stage2_losses'].append(loss_dict)

        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: Total={loss.item():.4f}, "
                  f"Recall={loss_dict['recall_loss'].item():.4f}, "
                  f"Cost={loss_dict['avg_cost']:.2f}")

    print("  Stage 2 complete.")

    return model, history

def evaluate_model(model, features, labels, monitor_scores, c_monitor=1.0, c_audit=20.0):
    """Evaluate model performance."""
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        outputs = model(features.to(device), monitor_scores.to(device))
        p_audit = outputs['p_audit']
        p_gate = outputs['p_gate']

        # Calculate recall (fraction of harmful examples audited)
        harmful_mask = (labels == 1)
        if harmful_mask.any():
            recall = (p_audit[harmful_mask] > 0.5).float().mean().item()
        else:
            recall = 0.0

        # Calculate average cost
        costs = c_monitor * p_gate + c_audit * p_audit
        avg_cost = costs.mean().item()

        # Calculate precision
        audit_mask = (p_audit > 0.5)
        if audit_mask.any():
            precision = (labels[audit_mask] == 1).float().mean().item()
        else:
            precision = 0.0

        return {
            'recall': recall,
            'precision': precision,
            'avg_cost': avg_cost,
            'n_audited': audit_mask.sum().item(),
            'n_harmful': harmful_mask.sum().item()
        }

def plot_training_comparison(joint_history, frozen_history, logger=None):
    """Plot side-by-side comparison with equivalent metrics for both approaches."""

    # Extract stage 2 data (recall+cost optimization) from both methods
    joint_recall_losses = []
    joint_cost_values = []
    joint_total_losses = []

    if joint_history['stage2_losses']:
        for loss_dict in joint_history['stage2_losses']:
            joint_recall_losses.append(-loss_dict['recall_loss'].item())
            joint_cost_values.append(loss_dict['avg_cost'])
            joint_total_losses.append(loss_dict['total_loss'].item())

    frozen_recall_losses = []
    frozen_cost_values = []
    frozen_total_losses = []

    if frozen_history['stage2_losses']:
        for loss_dict in frozen_history['stage2_losses']:
            frozen_recall_losses.append(-loss_dict['recall_loss'].item())
            frozen_cost_values.append(loss_dict['avg_cost'])
            frozen_total_losses.append(loss_dict['total_loss'].item())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Total Loss Comparison
    ax = axes[0, 0]
    if joint_total_losses:
        ax.plot(joint_total_losses, label='Joint (Probe + Gates)', color='blue', linewidth=2)
    if frozen_total_losses:
        ax.plot(frozen_total_losses, label='Frozen Probe (Gates Only)', color='red', linewidth=2)
    ax.set_title('Total Loss (Recall + Cost)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Recall Loss Comparison
    ax = axes[0, 1]
    if joint_recall_losses:
        ax.plot(joint_recall_losses, label='Joint (Probe + Gates)', color='blue', linewidth=2)
    if frozen_recall_losses:
        ax.plot(frozen_recall_losses, label='Frozen Probe (Gates Only)', color='red', linewidth=2)
    ax.set_title('Recall Performance')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall Loss (higher = better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cost Evolution Comparison
    ax = axes[0, 2]
    if joint_cost_values:
        ax.plot(joint_cost_values, label='Joint (Probe + Gates)', color='blue', linewidth=2)
    if frozen_cost_values:
        ax.plot(frozen_cost_values, label='Frozen Probe (Gates Only)', color='red', linewidth=2)
    ax.set_title('Average Cost Evolution')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Cost ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final performance comparison bars
    ax = axes[1, 0]
    if joint_total_losses and frozen_total_losses:
        joint_final = joint_total_losses[-1]
        frozen_final = frozen_total_losses[-1]

        bars = ax.bar(['Joint Training', 'Frozen Probe'], [joint_final, frozen_final],
                     color=['blue', 'red'], alpha=0.7)
        ax.set_title('Final Total Loss')
        ax.set_ylabel('Loss')

        # Add value labels
        ax.text(0, joint_final + 0.01, f'{joint_final:.3f}', ha='center', va='bottom')
        ax.text(1, frozen_final + 0.01, f'{frozen_final:.3f}', ha='center', va='bottom')

    ax = axes[1, 1]
    if joint_recall_losses and frozen_recall_losses:
        joint_final = joint_recall_losses[-1]
        frozen_final = frozen_recall_losses[-1]

        bars = ax.bar(['Joint Training', 'Frozen Probe'], [joint_final, frozen_final],
                     color=['blue', 'red'], alpha=0.7)
        ax.set_title('Final Recall Performance')
        ax.set_ylabel('Recall Loss')

        # Add value labels
        ax.text(0, joint_final + 0.01, f'{joint_final:.3f}', ha='center', va='bottom')
        ax.text(1, frozen_final + 0.01, f'{frozen_final:.3f}', ha='center', va='bottom')

    ax = axes[1, 2]
    if joint_cost_values and frozen_cost_values:
        joint_final = joint_cost_values[-1]
        frozen_final = frozen_cost_values[-1]

        bars = ax.bar(['Joint Training', 'Frozen Probe'], [joint_final, frozen_final],
                     color=['blue', 'red'], alpha=0.7)
        ax.set_title('Final Average Cost')
        ax.set_ylabel('Cost ($)')

        # Add value labels
        ax.text(0, joint_final + 0.1, f'${joint_final:.2f}', ha='center', va='bottom')
        ax.text(1, frozen_final + 0.1, f'${frozen_final:.2f}', ha='center', va='bottom')

    plt.suptitle('Training Method Comparison\nBoth: Stage 1 (Probe Classification) → Stage 2 (Recall+Cost Optimization)', fontsize=16)
    plt.tight_layout()

    if logger:
        logger.save_plot('training_comparison')

    plt.show()

def plot_decision_boundaries_comparison(joint_model, frozen_model, features, labels,
                                       monitor_scores, logger=None):
    """Plot decision boundaries for both training methods with data scatter."""

    # Extract learned parameters from both models
    with torch.no_grad():
        joint_params = {
            'u': joint_model.u.item(),
            'l': joint_model.l.item(),
            'beta_u': joint_model.beta_u.item(),
            'beta_l': joint_model.beta_l.item(),
            'alpha_1': joint_model.alpha_1.item(),
            'alpha_2': joint_model.alpha_2.item(),
            'beta_3': joint_model.beta_3.item(),
            'a': joint_model.a.item()
        }

        frozen_params = {
            'u': frozen_model.u.item(),
            'l': frozen_model.l.item(),
            'beta_u': frozen_model.beta_u.item(),
            'beta_l': frozen_model.beta_l.item(),
            'alpha_1': frozen_model.alpha_1.item(),
            'alpha_2': frozen_model.alpha_2.item(),
            'beta_3': frozen_model.beta_3.item(),
            'a': frozen_model.a.item()
        }

    # Get probe scores for both models
    device = next(joint_model.parameters()).device
    with torch.no_grad():
        joint_model.eval()
        frozen_model.eval()

        joint_outputs = joint_model(features.to(device), monitor_scores.to(device))
        frozen_outputs = frozen_model(features.to(device), monitor_scores.to(device))

        joint_probe_scores = joint_outputs['probe_scores'].cpu().numpy()
        frozen_probe_scores = frozen_outputs['probe_scores'].cpu().numpy()

    # Convert to numpy for plotting
    monitor_scores_np = monitor_scores.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Create figure with 2 rows (methods) x 3 cols (p_audit, p_gate, cost)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    methods = [
        ("Joint Training (Probe + Gates)", joint_params, joint_probe_scores, axes[0]),
        ("Frozen Probe (Gates Only)", frozen_params, frozen_probe_scores, axes[1])
    ]

    for method_name, params, probe_scores, method_axes in methods:

        # Create grid for contour plots
        monitor_range = np.linspace(-3, 3, 50)
        probe_range = np.linspace(-3, 3, 50)
        M2, M1 = np.meshgrid(monitor_range, probe_range)

        # Compute probabilities using learned parameters
        p1 = sigmoid(params['beta_u'] * (M1 - params['u']))
        p2 = sigmoid(params['beta_l'] * (M1 - params['l']))
        m3 = params['alpha_1'] * M1 + params['alpha_2'] * M2
        p3 = sigmoid(params['beta_3'] * (m3 - params['a']))

        p_gate = (1 - p1) * p2
        p_audit = p1 + p_gate * p3

        # Compute cost
        cost = 1.0 * p_gate + 20.0 * p_audit

        # Plot 1: P(audit)
        ax = method_axes[0]
        im1 = ax.contourf(M2, M1, p_audit, levels=20, cmap='Blues', alpha=0.7)

        # Scatter plot of actual data
        harmful_mask = labels_np == 1
        ax.scatter(monitor_scores_np[harmful_mask], probe_scores[harmful_mask],
                  c='red', s=30, alpha=0.8, marker='x', label='Harmful')
        ax.scatter(monitor_scores_np[~harmful_mask], probe_scores[~harmful_mask],
                  c='green', s=15, alpha=0.6, marker='o', label='Benign')

        ax.axhline(y=params['u'], color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax.axhline(y=params['l'], color='orange', linestyle='--', linewidth=2, alpha=0.8)

        ax.set_xlabel('Monitor Score (m₂)')
        ax.set_ylabel('Probe Score (m₁)')
        ax.set_title(f'{method_name}\nP(audit) - Total Audit Probability')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        plt.colorbar(im1, ax=ax, label='P(audit)')

        # Plot 2: P(gate)
        ax = method_axes[1]
        im2 = ax.contourf(M2, M1, p_gate, levels=20, cmap='Greens', alpha=0.7)

        # Scatter plot of actual data
        ax.scatter(monitor_scores_np[harmful_mask], probe_scores[harmful_mask],
                  c='red', s=30, alpha=0.8, marker='x', label='Harmful')
        ax.scatter(monitor_scores_np[~harmful_mask], probe_scores[~harmful_mask],
                  c='green', s=15, alpha=0.6, marker='o', label='Benign')

        ax.axhline(y=params['u'], color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax.axhline(y=params['l'], color='orange', linestyle='--', linewidth=2, alpha=0.8)

        ax.set_xlabel('Monitor Score (m₂)')
        ax.set_ylabel('Probe Score (m₁)')
        ax.set_title('P(gate) - Monitor Call Probability')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        plt.colorbar(im2, ax=ax, label='P(gate)')

        # Plot 3: Expected Cost
        ax = method_axes[2]
        im3 = ax.contourf(M2, M1, cost, levels=20, cmap='Reds', alpha=0.7)

        # Scatter plot of actual data
        ax.scatter(monitor_scores_np[harmful_mask], probe_scores[harmful_mask],
                  c='red', s=30, alpha=0.8, marker='x', label='Harmful')
        ax.scatter(monitor_scores_np[~harmful_mask], probe_scores[~harmful_mask],
                  c='green', s=15, alpha=0.6, marker='o', label='Benign')

        ax.axhline(y=params['u'], color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax.axhline(y=params['l'], color='orange', linestyle='--', linewidth=2, alpha=0.8)

        # Budget contour
        ax.contour(M2, M1, cost, levels=[3.0], colors='blue', linewidths=3, linestyles='--')

        ax.set_xlabel('Monitor Score (m₂)')
        ax.set_ylabel('Probe Score (m₁)')
        ax.set_title('Expected Cost (Budget=$3.0)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        plt.colorbar(im3, ax=ax, label='Cost ($)')

    plt.suptitle('Decision Boundaries Comparison: Joint vs Frozen Training\nRed X = Harmful, Green O = Benign', fontsize=16)
    plt.tight_layout()

    if logger:
        logger.save_plot('decision_boundaries_comparison')

    plt.show()

    # Print parameter comparison
    print(f"\n=== LEARNED PARAMETERS COMPARISON ===")
    print(f"{'Parameter':<12} {'Joint':<8} {'Frozen':<8} {'Difference':<10}")
    print("-" * 45)
    for param in ['u', 'l', 'beta_u', 'beta_l', 'alpha_1', 'alpha_2', 'beta_3', 'a']:
        joint_val = joint_params[param]
        frozen_val = frozen_params[param]
        diff = joint_val - frozen_val
        print(f"{param:<12} {joint_val:<8.3f} {frozen_val:<8.3f} {diff:<+10.3f}")

def run_training_comparison():
    """Run side-by-side comparison of training regimes."""

    config = {
        "experiment_type": "training_comparison",
        "n_samples": 800,
        "n_features": 32,
        "harmful_ratio": 0.2,
        "signal_strength": 0.3,
        "c_monitor": 5.0,
        "c_audit": 20.0,
        "budget": 3.0,
        "lambda_cost": 1.0,
        "curriculum_epochs": [15, 10, 10],  # stage1, stage2, joint
        "joint_epochs": 35,
        "lr": 0.002,
        "seed": 42
    }

    description = """
    Side-by-side comparison of two training regimes for hierarchical monitoring:

    1. Curriculum Learning:
       - Stage 1: Train probe standalone (BCE loss)
       - Stage 2: Train gates with frozen probe (recall-cost loss)
       - Stage 3: Joint fine-tuning

    2. Joint Training:
       - Train probe and gates together from start (recall-cost loss)

    Dataset is much harder: 20% adversarial examples, weak signals, multiple patterns.
    """

    with ExperimentLogger("training_comparison", description, config) as logger:
        print("=== TRAINING REGIME COMPARISON ===")
        print("Generating hard dataset...")

        features, labels, monitor_scores = generate_hard_dataset(
            n_samples=config["n_samples"],
            n_features=config["n_features"],
            harmful_ratio=config["harmful_ratio"],
            signal_strength=config["signal_strength"],
            seed=config["seed"]
        )

        harmful_count = (labels == 1).sum().item()
        print(f"Dataset: {len(features)} samples, {harmful_count} harmful ({harmful_count/len(features):.1%})")

        # Save dataset
        logger.save_data({
            'features': features,
            'labels': labels,
            'monitor_scores': monitor_scores
        }, 'dataset')

        print("\n=== STAGE 1: TRAIN PROBE FOR CLASSIFICATION (SHARED) ===")

        # Train a single probe that both methods will use
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
            features, labels, monitor_scores, test_size=0.3, random_state=42, stratify=labels
        )
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        m_train = m_train.to(device)

        # Train the shared probe
        shared_model = HardHierarchicalMonitor(features.shape[1]).to(device)
        probe_optimizer = optim.Adam(shared_model.get_probe_parameters(), lr=config["lr"])
        criterion_bce = nn.BCEWithLogitsLoss()

        stage1_losses = []
        for epoch in range(config["curriculum_epochs"][0]):
            shared_model.train()
            probe_optimizer.zero_grad()
            outputs = shared_model(X_train, m_train)
            probe_logits = outputs['probe_scores']
            loss = criterion_bce(probe_logits, y_train.float())
            loss.backward()
            probe_optimizer.step()
            stage1_losses.append(loss.item())

            if epoch % 10 == 0:
                shared_model.eval()
                with torch.no_grad():
                    test_outputs = shared_model(X_test.to(device), m_test.to(device))
                    test_probs = torch.sigmoid(test_outputs['probe_scores'])
                    test_preds = (test_probs > 0.5).float()
                    accuracy = (test_preds == y_test.to(device)).float().mean().item()
                    print(f"  Epoch {epoch}: BCE Loss={loss.item():.4f}, Accuracy={accuracy:.4f}")

        print(f"  Shared probe training complete. Final accuracy: {accuracy:.4f}")

        print("\n=== METHOD 1: JOINT TRAINING (Probe + Gates) ===")
        # Create model 1 and copy probe weights
        joint_model = HardHierarchicalMonitor(features.shape[1]).to(device)
        joint_model.load_state_dict(shared_model.state_dict())

        joint_history = {'stage1_losses': stage1_losses, 'stage2_losses': []}
        joint_optimizer = optim.Adam(joint_model.parameters(), lr=config["lr"] * 0.5)

        for epoch in range(config["curriculum_epochs"][1]):
            joint_model.train()
            joint_optimizer.zero_grad()
            outputs = joint_model(X_train, m_train)
            p_audit = outputs['p_audit']
            p_gate = outputs['p_gate']
            costs = config["c_monitor"] * p_gate + config["c_audit"] * p_audit
            loss_dict = recall_cost_loss(p_audit, y_train, costs, config["budget"], config["lambda_cost"])
            loss = loss_dict['total_loss']
            loss.backward()
            joint_optimizer.step()
            joint_history['stage2_losses'].append(loss_dict)

            if epoch % 10 == 0:
                print(f"    Epoch {epoch}: Total={loss.item():.4f}, "
                      f"Recall={loss_dict['recall_loss'].item():.4f}, "
                      f"Cost={loss_dict['avg_cost']:.2f}")

        print("\n=== METHOD 2: FROZEN PROBE (Gates Only) ===")
        # Create model 2 and copy probe weights
        frozen_model = HardHierarchicalMonitor(features.shape[1]).to(device)
        frozen_model.load_state_dict(shared_model.state_dict())

        # Freeze probe parameters
        for param in frozen_model.probe.parameters():
            param.requires_grad = False

        frozen_history = {'stage1_losses': stage1_losses, 'stage2_losses': []}
        gate_optimizer = optim.Adam(frozen_model.get_gate_parameters(), lr=config["lr"])

        for epoch in range(config["curriculum_epochs"][1]):
            frozen_model.train()
            gate_optimizer.zero_grad()
            outputs = frozen_model(X_train, m_train)
            p_audit = outputs['p_audit']
            p_gate = outputs['p_gate']
            costs = config["c_monitor"] * p_gate + config["c_audit"] * p_audit
            loss_dict = recall_cost_loss(p_audit, y_train, costs, config["budget"], config["lambda_cost"])
            loss = loss_dict['total_loss']
            loss.backward()
            gate_optimizer.step()
            frozen_history['stage2_losses'].append(loss_dict)

            if epoch % 10 == 0:
                print(f"    Epoch {epoch}: Total={loss.item():.4f}, "
                      f"Recall={loss_dict['recall_loss'].item():.4f}, "
                      f"Cost={loss_dict['avg_cost']:.2f}")

        # Evaluate both models
        print("\n=== EVALUATION ===")
        joint_results = evaluate_model(
            joint_model, features, labels, monitor_scores,
            config["c_monitor"], config["c_audit"]
        )

        frozen_results = evaluate_model(
            frozen_model, features, labels, monitor_scores,
            config["c_monitor"], config["c_audit"]
        )

        print("Method 1 - Joint Training (Probe + Gates):")
        print(f"  Recall: {joint_results['recall']:.3f}")
        print(f"  Precision: {joint_results['precision']:.3f}")
        print(f"  Avg Cost: ${joint_results['avg_cost']:.2f}")
        print(f"  Audited: {joint_results['n_audited']}/{len(features)}")

        print("Method 2 - Frozen Probe (Gates Only):")
        print(f"  Recall: {frozen_results['recall']:.3f}")
        print(f"  Precision: {frozen_results['precision']:.3f}")
        print(f"  Avg Cost: ${frozen_results['avg_cost']:.2f}")
        print(f"  Audited: {frozen_results['n_audited']}/{len(features)}")

        # Compare
        recall_diff = joint_results['recall'] - frozen_results['recall']
        cost_diff = joint_results['avg_cost'] - frozen_results['avg_cost']

        print(f"\n=== COMPARISON ===")
        print(f"Recall difference: {recall_diff:+.3f} (Joint - Frozen)")
        print(f"Cost difference: ${cost_diff:+.2f} (Joint - Frozen)")

        if recall_diff > 0.01:
            print("🏆 Joint training (probe + gates) achieves better recall!")
        elif recall_diff < -0.01:
            print("🏆 Frozen probe (gates only) achieves better recall!")
        else:
            print("🤝 Similar recall performance")

        # Save results
        logger.save_data({
            'joint_history': joint_history,
            'frozen_history': frozen_history
        }, 'training_histories')

        logger.save_data({
            'joint_results': joint_results,
            'frozen_results': frozen_results
        }, 'evaluation_results')

        # Plot training comparison
        plot_training_comparison(joint_history, frozen_history, logger)

        # Plot decision boundaries for both methods
        plot_decision_boundaries_comparison(joint_model, frozen_model,
                                          features, labels, monitor_scores, logger)

        # Save models
        logger.save_model(joint_model, 'joint_model')
        logger.save_model(frozen_model, 'frozen_model')

        # Log final results
        final_results = {
            'joint_recall': joint_results['recall'],
            'frozen_recall': frozen_results['recall'],
            'recall_difference': recall_diff,
            'joint_cost': joint_results['avg_cost'],
            'frozen_cost': frozen_results['avg_cost'],
            'cost_difference': cost_diff,
            'winner': 'joint' if recall_diff > 0.01 else 'frozen' if recall_diff < -0.01 else 'tie'
        }

        logger.log_results(final_results)

        return joint_model, frozen_model, joint_results, frozen_results

if __name__ == "__main__":
    joint_model, frozen_model, joint_results, frozen_results = run_training_comparison()