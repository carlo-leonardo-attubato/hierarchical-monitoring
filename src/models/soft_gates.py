"""
Soft-gating architecture for hierarchical monitoring.

This module implements the probabilistic gating mechanism that decides when to
escalate to expensive monitors or directly audit outputs. The gates are
differentiable, enabling end-to-end optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SoftGate(nn.Module):
    """
    Soft gate that converts continuous scores to probabilities using sigmoid.
    
    The gate implements: p = σ(β * (score - threshold))
    where β controls the sharpness and threshold is the decision boundary.
    
    Args:
        sharpness_init: Initial value for sharpness parameter β
        threshold_init: Initial value for threshold parameter
        learnable_sharpness: Whether to learn the sharpness parameter
        learnable_threshold: Whether to learn the threshold parameter
    """
    
    def __init__(
        self,
        sharpness_init: float = 1.0,
        threshold_init: float = 0.0,
        learnable_sharpness: bool = True,
        learnable_threshold: bool = True
    ):
        super().__init__()
        
        # Sharpness parameter (β)
        if learnable_sharpness:
            self.sharpness = nn.Parameter(torch.tensor(sharpness_init))
        else:
            self.register_buffer('sharpness', torch.tensor(sharpness_init))
        
        # Threshold parameter
        if learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(threshold_init))
        else:
            self.register_buffer('threshold', torch.tensor(threshold_init))
    
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply soft gate to scores.
        
        Args:
            scores: Input scores of shape (batch_size,)
            
        Returns:
            Probabilities of shape (batch_size,)
        """
        # Ensure sharpness is positive
        sharpness = F.softplus(self.sharpness) if self.sharpness.requires_grad else self.sharpness
        
        # Apply sigmoid gate: p = σ(β * (score - threshold))
        return torch.sigmoid(sharpness * (scores - self.threshold))
    
    def get_threshold(self) -> torch.Tensor:
        """Get the current threshold value."""
        return self.threshold
    
    def get_sharpness(self) -> torch.Tensor:
        """Get the current sharpness value."""
        return F.softplus(self.sharpness) if self.sharpness.requires_grad else self.sharpness


class HierarchicalGates(nn.Module):
    """
    Complete hierarchical gating system with three decision stages.
    
    The system implements:
    1. Immediate audit gate (p1): Direct audit for high-confidence harmful cases
    2. Escalation gate (p2): Call expensive monitor for uncertain cases
    3. Final decision gate (p3): Final audit decision based on ensemble
    
    Args:
        upper_threshold_init: Initial threshold for immediate audit gate
        lower_threshold_init: Initial threshold for escalation gate
        upper_sharpness_init: Initial sharpness for immediate audit gate
        lower_sharpness_init: Initial sharpness for escalation gate
        ensemble_weights_init: Initial weights for ensemble [α1, α2]
    """
    
    def __init__(
        self,
        upper_threshold_init: float = 0.5,
        lower_threshold_init: float = -0.5,
        upper_sharpness_init: float = 2.0,
        lower_sharpness_init: float = 2.0,
        ensemble_weights_init: Tuple[float, float] = (0.5, 0.5)
    ):
        super().__init__()
        
        # Immediate audit gate (upper gate)
        self.upper_gate = SoftGate(
            sharpness_init=upper_sharpness_init,
            threshold_init=upper_threshold_init
        )
        
        # Escalation gate (lower gate)
        self.lower_gate = SoftGate(
            sharpness_init=lower_sharpness_init,
            threshold_init=lower_threshold_init
        )
        
        # Ensemble weights for combining probe and monitor scores
        self.ensemble_weights = nn.Parameter(torch.tensor(ensemble_weights_init))
    
    def forward(
        self,
        probe_scores: torch.Tensor,
        monitor_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through hierarchical gates.
        
        Args:
            probe_scores: Probe scores of shape (batch_size,)
            monitor_scores: Monitor scores of shape (batch_size,) or None
            
        Returns:
            Tuple of (p1, p2, p3, p_gate) where:
            - p1: Immediate audit probabilities
            - p2: Escalation probabilities  
            - p3: Final audit probabilities
            - p_gate: Monitor call probabilities
        """
        # Stage 1: Immediate audit gate
        p1 = self.upper_gate(probe_scores)
        
        # Stage 2: Escalation gate
        p2 = self.lower_gate(probe_scores)
        
        # Probability of calling monitor (only if not immediately audited)
        p_gate = (1 - p1) * p2
        
        # Stage 3: Final decision (ensemble of probe and monitor)
        if monitor_scores is not None:
            # Normalize ensemble weights to sum to 1
            weights = F.softmax(self.ensemble_weights, dim=0)
            ensemble_scores = weights[0] * probe_scores + weights[1] * monitor_scores
        else:
            # If no monitor scores, use probe scores only
            ensemble_scores = probe_scores
        
        p3 = torch.sigmoid(ensemble_scores)
        
        return p1, p2, p3, p_gate
    
    def get_parameters(self) -> dict:
        """Get all learnable parameters."""
        return {
            'upper_threshold': self.upper_gate.get_threshold(),
            'lower_threshold': self.lower_gate.get_threshold(),
            'upper_sharpness': self.upper_gate.get_sharpness(),
            'lower_sharpness': self.lower_gate.get_sharpness(),
            'ensemble_weights': F.softmax(self.ensemble_weights, dim=0)
        }


class HardGates(nn.Module):
    """
    Hard threshold version of the gating system for comparison.
    
    This implements the same logic but with hard thresholds instead of
    soft gates. Useful for ablation studies.
    """
    
    def __init__(
        self,
        upper_threshold: float = 0.5,
        lower_threshold: float = -0.5,
        ensemble_weights: Tuple[float, float] = (0.5, 0.5)
    ):
        super().__init__()
        self.register_buffer('upper_threshold', torch.tensor(upper_threshold))
        self.register_buffer('lower_threshold', torch.tensor(lower_threshold))
        self.register_buffer('ensemble_weights', torch.tensor(ensemble_weights))
    
    def forward(
        self,
        probe_scores: torch.Tensor,
        monitor_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with hard thresholds."""
        # Stage 1: Immediate audit (hard threshold)
        p1 = (probe_scores > self.upper_threshold).float()
        
        # Stage 2: Escalation (hard threshold)
        p2 = (probe_scores > self.lower_threshold).float()
        
        # Probability of calling monitor
        p_gate = (1 - p1) * p2
        
        # Stage 3: Final decision
        if monitor_scores is not None:
            ensemble_scores = (
                self.ensemble_weights[0] * probe_scores + 
                self.ensemble_weights[1] * monitor_scores
            )
        else:
            ensemble_scores = probe_scores
        
        p3 = (ensemble_scores > 0).float()
        
        return p1, p2, p3, p_gate


def create_gates(
    gate_type: str = "soft",
    **kwargs
) -> nn.Module:
    """
    Factory function to create different types of gates.
    
    Args:
        gate_type: Type of gates ("soft" or "hard")
        **kwargs: Additional arguments for gate creation
        
    Returns:
        Initialized gates model
    """
    if gate_type == "soft":
        return HierarchicalGates(**kwargs)
    elif gate_type == "hard":
        return HardGates(**kwargs)
    else:
        raise ValueError(f"Unknown gate type: {gate_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test soft gates
    gates = HierarchicalGates()
    probe_scores = torch.randn(10)  # batch_size=10
    monitor_scores = torch.randn(10)
    
    p1, p2, p3, p_gate = gates(probe_scores, monitor_scores)
    print(f"Immediate audit probabilities: {p1}")
    print(f"Escalation probabilities: {p2}")
    print(f"Final audit probabilities: {p3}")
    print(f"Monitor call probabilities: {p_gate}")
    
    # Test hard gates
    hard_gates = HardGates()
    p1_hard, p2_hard, p3_hard, p_gate_hard = hard_gates(probe_scores, monitor_scores)
    print(f"Hard gate immediate audit: {p1_hard}")
    print(f"Hard gate escalation: {p2_hard}")
