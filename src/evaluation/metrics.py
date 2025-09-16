"""
Evaluation metrics for hierarchical monitoring.

This module provides comprehensive evaluation metrics for assessing
the performance of hierarchical monitoring systems.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import pandas as pd


class HierarchicalMetrics:
    """
    Comprehensive metrics for hierarchical monitoring systems.
    
    This class computes various metrics including recall, precision,
    cost efficiency, and budget adherence.
    """
    
    def __init__(self, monitor_cost: float = 1.0, audit_cost: float = 20.0):
        self.monitor_cost = monitor_cost
        self.audit_cost = audit_cost
    
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        budget: float,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute comprehensive metrics for hierarchical monitoring.
        
        Args:
            outputs: Dictionary from HierarchicalMonitor.forward()
            labels: Binary labels where 1 = harmful
            budget: Budget per output
            threshold: Threshold for binary decisions
            
        Returns:
            Dictionary containing all metrics
        """
        p_audit = outputs['p_audit']
        p_gate = outputs['p_gate']
        expected_total_cost = outputs['expected_total_cost']
        
        # Convert to numpy for sklearn compatibility
        p_audit_np = p_audit.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        expected_costs_np = expected_total_cost.detach().cpu().numpy()
        
        metrics = {}
        
        # Classification metrics
        metrics.update(self._compute_classification_metrics(p_audit_np, labels_np, threshold))
        
        # Cost metrics
        metrics.update(self._compute_cost_metrics(expected_costs_np, labels_np, budget))
        
        # Hierarchical-specific metrics
        metrics.update(self._compute_hierarchical_metrics(outputs, labels_np))
        
        return metrics
    
    def _compute_classification_metrics(
        self,
        p_audit: np.ndarray,
        labels: np.ndarray,
        threshold: float
    ) -> Dict[str, float]:
        """Compute standard classification metrics."""
        # Binary predictions
        predictions = (p_audit > threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        
        # Basic metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # AUC metrics
        try:
            auc_roc = roc_auc_score(labels, p_audit)
        except ValueError:
            auc_roc = 0.5  # Default when only one class present
        
        try:
            auc_pr = average_precision_score(labels, p_audit)
        except ValueError:
            auc_pr = labels.mean()  # Default when only one class present
        
        return {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
    
    def _compute_cost_metrics(
        self,
        expected_costs: np.ndarray,
        labels: np.ndarray,
        budget: float
    ) -> Dict[str, float]:
        """Compute cost-related metrics."""
        # Overall cost metrics
        mean_cost = expected_costs.mean()
        cost_std = expected_costs.std()
        budget_violation_rate = (expected_costs > budget).mean()
        
        # Cost metrics for benign examples
        benign_mask = (labels == 0)
        if benign_mask.any():
            benign_costs = expected_costs[benign_mask]
            benign_mean_cost = benign_costs.mean()
            benign_budget_violation_rate = (benign_costs > budget).mean()
        else:
            benign_mean_cost = 0.0
            benign_budget_violation_rate = 0.0
        
        # Cost efficiency metrics
        cost_per_harmful_detected = mean_cost / max(labels.mean(), 1e-8)
        
        return {
            'mean_cost': mean_cost,
            'cost_std': cost_std,
            'budget_violation_rate': budget_violation_rate,
            'benign_mean_cost': benign_mean_cost,
            'benign_budget_violation_rate': benign_budget_violation_rate,
            'cost_per_harmful_detected': cost_per_harmful_detected,
            'budget_utilization': mean_cost / budget
        }
    
    def _compute_hierarchical_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute hierarchical-specific metrics."""
        p1 = outputs['p1'].detach().cpu().numpy()
        p2 = outputs['p2'].detach().cpu().numpy()
        p_gate = outputs['p_gate'].detach().cpu().numpy()
        
        # Gate usage statistics
        immediate_audit_rate = p1.mean()
        escalation_rate = p_gate.mean()
        
        # Gate effectiveness (for harmful examples)
        harmful_mask = (labels == 1)
        if harmful_mask.any():
            harmful_immediate_audit_rate = p1[harmful_mask].mean()
            harmful_escalation_rate = p_gate[harmful_mask].mean()
        else:
            harmful_immediate_audit_rate = 0.0
            harmful_escalation_rate = 0.0
        
        return {
            'immediate_audit_rate': immediate_audit_rate,
            'escalation_rate': escalation_rate,
            'harmful_immediate_audit_rate': harmful_immediate_audit_rate,
            'harmful_escalation_rate': harmful_escalation_rate
        }


class DeploymentMetrics:
    """
    Metrics for evaluating deployed hierarchical monitoring systems.
    
    This evaluates the actual deployment behavior including
    real costs and decisions.
    """
    
    def __init__(self, monitor_cost: float = 1.0, audit_cost: float = 20.0):
        self.monitor_cost = monitor_cost
        self.audit_cost = audit_cost
    
    def compute_deployment_metrics(
        self,
        deployment_results: Dict[str, Any],
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute metrics for deployment results.
        
        Args:
            deployment_results: Dictionary from HierarchicalMonitor.deploy()
            labels: Binary labels where 1 = harmful
            
        Returns:
            Dictionary containing deployment metrics
        """
        audit_decisions = deployment_results['audit_decisions']
        monitor_calls = deployment_results['monitor_calls']
        total_costs = deployment_results['total_costs']
        
        labels_np = labels.detach().cpu().numpy()
        audit_decisions_np = audit_decisions.detach().cpu().numpy()
        monitor_calls_np = monitor_calls.detach().cpu().numpy()
        total_costs_np = total_costs.detach().cpu().numpy()
        
        metrics = {}
        
        # Detection performance
        metrics.update(self._compute_detection_metrics(audit_decisions_np, labels_np))
        
        # Cost analysis
        metrics.update(self._compute_deployment_costs(
            total_costs_np, monitor_calls_np, audit_decisions_np, labels_np
        ))
        
        # System behavior
        metrics.update(self._compute_system_behavior(deployment_results, labels_np))
        
        return metrics
    
    def _compute_detection_metrics(
        self,
        audit_decisions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute detection performance metrics."""
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, audit_decisions).ravel()
        
        # Detection metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'deployment_precision': precision,
            'deployment_recall': recall,
            'deployment_specificity': specificity,
            'deployment_f1': f1,
            'deployment_tp': tp,
            'deployment_fp': fp,
            'deployment_tn': tn,
            'deployment_fn': fn
        }
    
    def _compute_deployment_costs(
        self,
        total_costs: np.ndarray,
        monitor_calls: np.ndarray,
        audit_decisions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute actual deployment costs."""
        # Overall cost statistics
        mean_cost = total_costs.mean()
        cost_std = total_costs.std()
        total_monitor_cost = monitor_calls.sum() * self.monitor_cost
        total_audit_cost = audit_decisions.sum() * self.audit_cost
        
        # Cost breakdown
        monitor_call_rate = monitor_calls.mean()
        audit_rate = audit_decisions.mean()
        
        # Cost per harmful detection
        harmful_detected = ((labels == 1) & (audit_decisions == 1)).sum()
        cost_per_harmful_detected = (
            total_costs.sum() / max(harmful_detected, 1)
        )
        
        return {
            'deployment_mean_cost': mean_cost,
            'deployment_cost_std': cost_std,
            'deployment_total_monitor_cost': total_monitor_cost,
            'deployment_total_audit_cost': total_audit_cost,
            'deployment_monitor_call_rate': monitor_call_rate,
            'deployment_audit_rate': audit_rate,
            'deployment_cost_per_harmful_detected': cost_per_harmful_detected
        }
    
    def _compute_system_behavior(
        self,
        deployment_results: Dict[str, Any],
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute system behavior metrics."""
        immediate_audits = deployment_results['immediate_audits']
        escalations = deployment_results['escalations']
        final_audits = deployment_results['final_audits']
        total_samples = len(labels)
        
        # Behavior rates
        immediate_audit_rate = immediate_audits / total_samples
        escalation_rate = escalations / total_samples
        final_audit_rate = final_audits / total_samples
        
        # Effectiveness for harmful examples
        harmful_samples = (labels == 1).sum()
        if harmful_samples > 0:
            harmful_audit_rate = (
                deployment_results['audit_decisions'].sum().item() / harmful_samples
            )
        else:
            harmful_audit_rate = 0.0
        
        return {
            'immediate_audit_rate': immediate_audit_rate,
            'escalation_rate': escalation_rate,
            'final_audit_rate': final_audit_rate,
            'harmful_audit_rate': harmful_audit_rate
        }


def evaluate_at_budget(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    budgets: List[float],
    device: str = "cpu"
) -> pd.DataFrame:
    """
    Evaluate model performance at different budget levels.
    
    Args:
        model: Trained hierarchical monitoring model
        data_loader: Data loader for evaluation
        budgets: List of budget levels to evaluate
        device: Device to run evaluation on
        
    Returns:
        DataFrame with results at each budget level
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            monitor_scores = batch.get('monitor_scores', None)
            if monitor_scores is not None:
                monitor_scores = monitor_scores.to(device)
            
            # Forward pass
            outputs = model(features, monitor_scores, training=True)
            
            # Evaluate at different budgets
            for budget in budgets:
                # Temporarily set budget
                original_budget = model.budget
                model.budget = budget
                
                # Compute metrics
                metrics = HierarchicalMetrics(
                    model.monitor_cost, model.audit_cost
                ).compute_metrics(outputs, labels, budget)
                
                # Add budget info
                metrics['budget'] = budget
                results.append(metrics)
                
                # Restore original budget
                model.budget = original_budget
    
    return pd.DataFrame(results)


# Example usage and testing
if __name__ == "__main__":
    # Test metrics computation
    print("Testing metrics computation...")
    
    # Mock data
    batch_size = 100
    outputs = {
        'p_audit': torch.sigmoid(torch.randn(batch_size)),
        'p_gate': torch.sigmoid(torch.randn(batch_size)),
        'expected_total_cost': torch.rand(batch_size) * 5.0
    }
    labels = torch.randint(0, 2, (batch_size,))
    budget = 2.0
    
    # Compute metrics
    metrics = HierarchicalMetrics().compute_metrics(outputs, labels, budget)
    print("Sample metrics:")
    for key, value in list(metrics.items())[:5]:
        print(f"  {key}: {value:.4f}")
    
    print("Metrics classes implemented successfully!")
