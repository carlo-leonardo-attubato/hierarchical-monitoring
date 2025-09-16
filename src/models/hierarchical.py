"""
Complete hierarchical monitoring system.

This module implements the full hierarchical monitoring system that combines
the probe, soft gates, and deployment logic as described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np

from .probe import LinearProbe
from .soft_gates import HierarchicalGates


class HierarchicalMonitor(nn.Module):
    """
    Complete hierarchical monitoring system.
    
    This implements the three-stage monitoring system:
    1. Probe evaluation (cheap)
    2. Conditional monitor call (expensive) 
    3. Final audit decision (expensive)
    
    Args:
        input_dim: Dimension of input features (internal activations)
        monitor_cost: Cost of calling the expensive monitor
        audit_cost: Cost of auditing an output
        budget: Budget per output (used during training)
        probe_type: Type of probe ("linear" or "mlp")
        gate_type: Type of gates ("soft" or "hard")
        **kwargs: Additional arguments for probe and gates
    """
    
    def __init__(
        self,
        input_dim: int,
        monitor_cost: float = 1.0,
        audit_cost: float = 20.0,
        budget: float = 2.0,
        probe_type: str = "linear",
        gate_type: str = "soft",
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.monitor_cost = monitor_cost
        self.audit_cost = audit_cost
        self.budget = budget
        
        # Initialize probe
        if probe_type == "linear":
            self.probe = LinearProbe(input_dim, **kwargs)
        else:
            raise ValueError(f"Probe type {probe_type} not implemented yet")
        
        # Initialize gates
        if gate_type == "soft":
            self.gates = HierarchicalGates(**kwargs)
        else:
            raise ValueError(f"Gate type {gate_type} not implemented yet")
    
    def forward(
        self,
        features: torch.Tensor,
        monitor_scores: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hierarchical monitoring system.
        
        Args:
            features: Input features of shape (batch_size, input_dim)
            monitor_scores: Precomputed monitor scores of shape (batch_size,) or None
            training: Whether in training mode (uses soft probabilities)
            
        Returns:
            Dictionary containing all intermediate values and decisions
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Stage 1: Probe evaluation (always computed)
        probe_scores = self.probe(features).squeeze(-1)  # (batch_size,)
        
        # Stage 2: Gating decisions
        p1, p2, p3, p_gate = self.gates(probe_scores, monitor_scores)
        
        # Compute total audit probability
        p_audit = p1 + p_gate * p3
        
        # Compute expected costs
        expected_monitor_cost = p_gate * self.monitor_cost
        expected_audit_cost = p_audit * self.audit_cost
        expected_total_cost = expected_monitor_cost + expected_audit_cost
        
        return {
            'probe_scores': probe_scores,
            'p1': p1,  # Immediate audit probability
            'p2': p2,  # Escalation probability
            'p3': p3,  # Final audit probability
            'p_gate': p_gate,  # Monitor call probability
            'p_audit': p_audit,  # Total audit probability
            'expected_monitor_cost': expected_monitor_cost,
            'expected_audit_cost': expected_audit_cost,
            'expected_total_cost': expected_total_cost
        }
    
    def deploy(
        self,
        features: torch.Tensor,
        monitor_fn: callable,
        sample_decisions: bool = True
    ) -> Dict[str, Any]:
        """
        Deploy the hierarchical monitoring system with actual decisions.
        
        This is the deployment version that makes actual binary decisions
        and calls the expensive monitor when needed.
        
        Args:
            features: Input features of shape (batch_size, input_dim)
            monitor_fn: Function that takes features and returns monitor scores
            sample_decisions: Whether to sample decisions or use deterministic thresholds
            
        Returns:
            Dictionary containing deployment results
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Get probe scores
        probe_scores = self.probe(features).squeeze(-1)
        
        # Get gate probabilities
        p1, p2, p3, p_gate = self.gates(probe_scores, None)
        
        # Make decisions
        if sample_decisions:
            # Sample from Bernoulli distributions
            a1 = torch.bernoulli(p1)  # Immediate audit decision
            a2 = torch.bernoulli(p2)  # Escalation decision
        else:
            # Use deterministic thresholds (0.5)
            a1 = (p1 > 0.5).float()
            a2 = (p2 > 0.5).float()
        
        # Track costs and decisions
        total_cost = torch.zeros(batch_size, device=device)
        audit_decisions = torch.zeros(batch_size, device=device)
        monitor_calls = torch.zeros(batch_size, device=device)
        
        # Stage 1: Immediate audit
        immediate_audit_mask = (a1 == 1)
        audit_decisions[immediate_audit_mask] = 1
        total_cost[immediate_audit_mask] = self.audit_cost
        
        # Stage 2: Escalation (only for non-immediate audits)
        escalation_mask = (a1 == 0) & (a2 == 1)
        monitor_calls[escalation_mask] = 1
        total_cost[escalation_mask] = self.monitor_cost
        
        # Stage 3: Final decision (for escalated cases)
        if escalation_mask.any():
            # Call expensive monitor for escalated cases
            escalated_features = features[escalation_mask]
            monitor_scores = monitor_fn(escalated_features)
            
            # Get final probabilities for escalated cases
            _, _, p3_escalated, _ = self.gates(
                probe_scores[escalation_mask], 
                monitor_scores
            )
            
            # Make final audit decisions
            if sample_decisions:
                a3 = torch.bernoulli(p3_escalated)
            else:
                a3 = (p3_escalated > 0.5).float()
            
            # Update audit decisions and costs
            final_audit_mask = escalation_mask.clone()
            final_audit_mask[escalation_mask] = (a3 == 1)
            audit_decisions[final_audit_mask] = 1
            total_cost[final_audit_mask] += self.audit_cost
        
        return {
            'probe_scores': probe_scores,
            'audit_decisions': audit_decisions,
            'monitor_calls': monitor_calls,
            'total_costs': total_cost,
            'immediate_audits': immediate_audit_mask.sum().item(),
            'escalations': escalation_mask.sum().item(),
            'final_audits': (audit_decisions & ~immediate_audit_mask).sum().item()
        }
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all learnable parameters."""
        params = self.gates.get_parameters()
        params['probe_weights'] = self.probe.get_weights()
        if self.probe.get_bias() is not None:
            params['probe_bias'] = self.probe.get_bias()
        return params
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Set all learnable parameters."""
        # Set probe parameters
        if 'probe_weights' in params:
            self.probe.set_weights(params['probe_weights'])
        if 'probe_bias' in params and self.probe.get_bias() is not None:
            self.probe.set_bias(params['probe_bias'])
        
        # Set gate parameters
        if 'upper_threshold' in params:
            self.gates.upper_gate.threshold.data = params['upper_threshold']
        if 'lower_threshold' in params:
            self.gates.lower_gate.threshold.data = params['lower_threshold']
        if 'upper_sharpness' in params:
            self.gates.upper_gate.sharpness.data = params['upper_sharpness']
        if 'lower_sharpness' in params:
            self.gates.lower_gate.sharpness.data = params['lower_sharpness']
        if 'ensemble_weights' in params:
            self.gates.ensemble_weights.data = params['ensemble_weights']


def create_hierarchical_monitor(
    input_dim: int,
    monitor_cost: float = 1.0,
    audit_cost: float = 20.0,
    budget: float = 2.0,
    **kwargs
) -> HierarchicalMonitor:
    """
    Factory function to create a hierarchical monitor.
    
    Args:
        input_dim: Dimension of input features
        monitor_cost: Cost of calling monitor
        audit_cost: Cost of auditing
        budget: Budget per output
        **kwargs: Additional arguments
        
    Returns:
        Initialized hierarchical monitor
    """
    return HierarchicalMonitor(
        input_dim=input_dim,
        monitor_cost=monitor_cost,
        audit_cost=audit_cost,
        budget=budget,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    # Create monitor
    monitor = HierarchicalMonitor(
        input_dim=128,
        monitor_cost=1.0,
        audit_cost=20.0,
        budget=2.0
    )
    
    # Test forward pass (training mode)
    features = torch.randn(10, 128)
    monitor_scores = torch.randn(10)
    
    results = monitor(features, monitor_scores, training=True)
    print("Training mode results:")
    for key, value in results.items():
        print(f"  {key}: {value.shape}")
    
    # Test deployment mode
    def dummy_monitor_fn(features):
        return torch.randn(features.shape[0])
    
    deploy_results = monitor.deploy(features, dummy_monitor_fn, sample_decisions=False)
    print("\nDeployment results:")
    for key, value in deploy_results.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")
