"""
Linear probe implementation for hierarchical monitoring.

This module implements a simple linear probe that maps internal activations
to monitoring scores. The probe is designed to be fast and have near-zero
marginal cost per evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LinearProbe(nn.Module):
    """
    Linear probe for mapping internal activations to monitoring scores.
    
    The probe performs a simple linear transformation: score = W @ x + b
    where W is a weight matrix and b is a bias vector.
    
    Args:
        input_dim: Dimension of input features (internal activations)
        output_dim: Dimension of output scores (default: 1 for binary classification)
        bias: Whether to include bias term (default: True)
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Linear layer: input_dim -> output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linear probe.
        
        Args:
            x: Input features of shape (batch_size, input_dim)
            
        Returns:
            Monitoring scores of shape (batch_size, output_dim)
        """
        return self.linear(x)
    
    def get_weights(self) -> torch.Tensor:
        """Get the weight matrix."""
        return self.linear.weight
    
    def get_bias(self) -> Optional[torch.Tensor]:
        """Get the bias vector."""
        return self.linear.bias
    
    def set_weights(self, weights: torch.Tensor):
        """Set the weight matrix."""
        if weights.shape != self.linear.weight.shape:
            raise ValueError(f"Weight shape mismatch: expected {self.linear.weight.shape}, got {weights.shape}")
        self.linear.weight.data = weights
    
    def set_bias(self, bias: torch.Tensor):
        """Set the bias vector."""
        if self.linear.bias is None:
            raise ValueError("Cannot set bias: bias is disabled")
        if bias.shape != self.linear.bias.shape:
            raise ValueError(f"Bias shape mismatch: expected {self.linear.bias.shape}, got {bias.shape}")
        self.linear.bias.data = bias


class MLPProbe(nn.Module):
    """
    Multi-layer perceptron probe for more complex mappings.
    
    This is an alternative to the linear probe that can learn more complex
    decision boundaries, but at higher computational cost.
    
    Args:
        input_dim: Dimension of input features
        hidden_dims: List of hidden layer dimensions
        output_dim: Dimension of output scores
        activation: Activation function to use
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 1,
        activation: str = "relu",
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        return self.network(x)


def create_probe(
    probe_type: str,
    input_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different types of probes.
    
    Args:
        probe_type: Type of probe ("linear" or "mlp")
        input_dim: Dimension of input features
        **kwargs: Additional arguments for probe creation
        
    Returns:
        Initialized probe model
    """
    if probe_type == "linear":
        return LinearProbe(input_dim, **kwargs)
    elif probe_type == "mlp":
        return MLPProbe(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test linear probe
    probe = LinearProbe(input_dim=128, output_dim=1)
    x = torch.randn(32, 128)  # batch_size=32, input_dim=128
    scores = probe(x)
    print(f"Linear probe output shape: {scores.shape}")
    
    # Test MLP probe
    mlp_probe = MLPProbe(input_dim=128, hidden_dims=[64, 32], output_dim=1)
    scores = mlp_probe(x)
    print(f"MLP probe output shape: {scores.shape}")
