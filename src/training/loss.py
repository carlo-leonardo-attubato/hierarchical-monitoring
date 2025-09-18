"""
Loss functions for hierarchical monitoring training.

This module implements the differentiable loss functions that balance
recall on harmful outputs with cost constraints on benign outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Callable
import math


class RecallLoss(nn.Module):
    """
    Loss function for maximizing recall on harmful outputs.
    
    This penalizes low audit probabilities for harmful examples (y=1).
    """
    
    def __init__(self, loss_type: str = "log_likelihood", gamma: float = 2.0):
        super().__init__()
        self.loss_type = loss_type
        self.gamma = gamma
    
    def forward(self, p_audit: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute recall loss for harmful examples.
        
        Args:
            p_audit: Audit probabilities of shape (batch_size,)
            labels: Binary labels of shape (batch_size,) where 1 = harmful
            
        Returns:
            Recall loss (scalar)
        """
        harmful_mask = (labels == 1)
        if not harmful_mask.any():
            return torch.tensor(0.0, device=p_audit.device)
        
        harmful_p_audit = p_audit[harmful_mask]
        
        if self.loss_type == "log_likelihood":
            # Negative log-likelihood: -log(p)
            loss = -torch.log(harmful_p_audit + 1e-8).mean()
        
        elif self.loss_type == "linear":
            # Linear: 1 - p
            loss = (1 - harmful_p_audit).mean()
        
        elif self.loss_type == "quadratic":
            # Quadratic: (1 - p)^2
            loss = ((1 - harmful_p_audit) ** 2).mean()
        
        elif self.loss_type == "exponential":
            # Exponential: exp(1 - p) - 1
            loss = (torch.exp(1 - harmful_p_audit) - 1).mean()
        
        elif self.loss_type == "focal":
            # Focal loss: -(1-p)^γ * log(p)
            loss = -((1 - harmful_p_audit) ** self.gamma * 
                    torch.log(harmful_p_audit + 1e-8)).mean()
        
        else:
            raise ValueError(f"Unknown recall loss type: {self.loss_type}")
        
        return loss


class CostLoss(nn.Module):
    """
    Loss function for enforcing budget constraints on benign outputs.
    
    This penalizes deviation from the budget for benign examples (y=0).
    """
    
    def __init__(
        self, 
        penalty_type: str = "quadratic", 
        scale: float = 1.0,
        delta: float = 1.0
    ):
        super().__init__()
        self.penalty_type = penalty_type
        self.scale = scale
        self.delta = delta
    
    def forward(
        self, 
        expected_costs: torch.Tensor, 
        labels: torch.Tensor, 
        budget: float
    ) -> torch.Tensor:
        """
        Compute cost loss for benign examples.
        
        Args:
            expected_costs: Expected costs of shape (batch_size,)
            labels: Binary labels of shape (batch_size,) where 0 = benign
            budget: Budget per output
            
        Returns:
            Cost loss (scalar)
        """
        benign_mask = (labels == 0)
        if not benign_mask.any():
            return torch.tensor(0.0, device=expected_costs.device)

        benign_costs = expected_costs[benign_mask]
        n_benign = benign_mask.sum().item()
        per_benign_budget = budget  # budget is already per-benign
        total_budget = n_benign * per_benign_budget
        cost_ratio = benign_costs.sum() / total_budget  # r = total_cost / (n_benign * budget_per_benign)
        
        if self.penalty_type == "quadratic":
            # Quadratic: s * (r - 1)^2
            loss = self.scale * ((cost_ratio - 1) ** 2)
        
        elif self.penalty_type == "absolute":
            # Absolute: s * |r - 1|
            loss = self.scale * torch.abs(cost_ratio - 1)
        
        elif self.penalty_type == "huber":
            # Huber: smooth absolute loss
            diff = torch.abs(cost_ratio - 1)
            loss = torch.where(
                diff <= self.delta,
                self.scale * 0.5 * diff ** 2,
                self.scale * self.delta * (diff - self.delta / 2)
            )
        
        elif self.penalty_type == "exponential_quadratic":
            # Exponential quadratic: exp(s * (r - 1)^2) - 1
            loss = torch.exp(self.scale * ((cost_ratio - 1) ** 2)) - 1
        
        elif self.penalty_type == "exponential_linear":
            # Exponential linear: exp(s * (r - 1))
            loss = torch.exp(self.scale * (cost_ratio - 1))
        
        elif self.penalty_type == "exponential_absolute":
            # Exponential absolute: exp(s * |r - 1|) - 1
            loss = torch.exp(self.scale * torch.abs(cost_ratio - 1)) - 1
        
        else:
            raise ValueError(f"Unknown cost penalty type: {self.penalty_type}")
        
        return loss


class HierarchicalLoss(nn.Module):
    """
    Combined loss function for hierarchical monitoring.
    
    This combines recall loss (on harmful examples) and cost loss (on benign examples)
    with a weighting parameter λ.
    """
    
    def __init__(
        self,
        recall_loss_type: str = "log_likelihood",
        cost_penalty_type: str = "quadratic",
        lambda_cost: float = 1.0,
        recall_gamma: float = 2.0,
        cost_scale: float = 1.0,
        cost_delta: float = 1.0
    ):
        super().__init__()
        self.lambda_cost = lambda_cost
        
        self.recall_loss = RecallLoss(
            loss_type=recall_loss_type,
            gamma=recall_gamma
        )
        
        self.cost_loss = CostLoss(
            penalty_type=cost_penalty_type,
            scale=cost_scale,
            delta=cost_delta
        )
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        budget: float
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the combined hierarchical loss.
        
        Args:
            outputs: Dictionary from HierarchicalMonitor.forward()
            labels: Binary labels of shape (batch_size,)
            budget: Budget per output
            
        Returns:
            Dictionary containing individual and combined losses
        """
        p_audit = outputs['p_audit']
        expected_total_cost = outputs['expected_total_cost']
        
        # Recall loss (only on harmful examples)
        recall_loss = self.recall_loss(p_audit, labels)
        
        # Cost loss (only on benign examples)
        cost_loss = self.cost_loss(expected_total_cost, labels, budget)
        
        # Combined loss
        total_loss = recall_loss + self.lambda_cost * cost_loss
        
        return {
            'total_loss': total_loss,
            'recall_loss': recall_loss,
            'cost_loss': cost_loss,
            'lambda_cost': self.lambda_cost
        }


class BaselineLoss(nn.Module):
    """
    Baseline loss functions for comparison.
    
    These implement standard classification losses that don't consider
    the hierarchical monitoring context.
    """
    
    def __init__(self, loss_type: str = "bce"):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type == "focal":
            self.loss_fn = FocalLoss()
        else:
            raise ValueError(f"Unknown baseline loss type: {loss_type}")
    
    def forward(
        self,
        probe_scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute baseline loss.
        
        Args:
            probe_scores: Probe scores of shape (batch_size,)
            labels: Binary labels of shape (batch_size,)
            
        Returns:
            Loss value
        """
        if self.loss_type == "bce":
            return self.loss_fn(probe_scores, labels.float())
        else:
            return self.loss_fn(probe_scores, labels)


class FocalLoss(nn.Module):
    """
    Focal loss implementation for handling class imbalance.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits of shape (batch_size,)
            targets: Binary labels of shape (batch_size,)
            
        Returns:
            Focal loss
        """
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


def create_loss(
    loss_type: str = "hierarchical",
    **kwargs
) -> nn.Module:
    """
    Factory function to create different types of losses.
    
    Args:
        loss_type: Type of loss ("hierarchical", "bce", "focal")
        **kwargs: Additional arguments for loss creation
        
    Returns:
        Initialized loss function
    """
    if loss_type == "hierarchical":
        return HierarchicalLoss(**kwargs)
    elif loss_type in ["bce", "focal"]:
        return BaselineLoss(loss_type=loss_type)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test hierarchical loss
    loss_fn = HierarchicalLoss(
        recall_loss_type="log_likelihood",
        cost_penalty_type="quadratic",
        lambda_cost=1.0
    )
    
    # Mock outputs from HierarchicalMonitor
    batch_size = 10
    outputs = {
        'p_audit': torch.sigmoid(torch.randn(batch_size)),
        'expected_total_cost': torch.rand(batch_size) * 5.0
    }
    labels = torch.randint(0, 2, (batch_size,))
    budget = 2.0
    
    losses = loss_fn(outputs, labels, budget)
    print("Hierarchical loss components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Test baseline loss
    baseline_loss = BaselineLoss("bce")
    probe_scores = torch.randn(batch_size)
    loss_value = baseline_loss(probe_scores, labels.float())
    print(f"\nBaseline BCE loss: {loss_value.item():.4f}")
