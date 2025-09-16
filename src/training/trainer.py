"""
Training loop for hierarchical monitoring.

This module implements the training procedure that jointly optimizes
the probe parameters and gating parameters using gradient descent.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Any
import numpy as np
from tqdm import tqdm
import logging

from ..models.hierarchical import HierarchicalMonitor
from .loss import HierarchicalLoss, BaselineLoss


class HierarchicalTrainer:
    """
    Trainer for hierarchical monitoring systems.
    
    This handles the complete training loop including optimization,
    evaluation, and checkpointing.
    """
    
    def __init__(
        self,
        model: HierarchicalMonitor,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        device: str = "cpu",
        log_interval: int = 100
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Initialize loss function
        if loss_fn is None:
            self.loss_fn = HierarchicalLoss()
        else:
            self.loss_fn = loss_fn
        
        # Initialize optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.log_interval = log_interval
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_recall_loss = 0.0
        total_cost_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            monitor_scores = batch.get('monitor_scores', None)
            if monitor_scores is not None:
                monitor_scores = monitor_scores.to(self.device)
            
            # Forward pass
            outputs = self.model(features, monitor_scores, training=True)
            
            # Compute loss
            losses = self.loss_fn(outputs, labels, self.model.budget)
            loss = losses['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_recall_loss += losses['recall_loss'].item()
            total_cost_loss += losses['cost_loss'].item()
            num_batches += 1
            self.step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recall': f"{losses['recall_loss'].item():.4f}",
                'cost': f"{losses['cost_loss'].item():.4f}"
            })
            
            # Log intermediate results
            if self.step % self.log_interval == 0:
                self.logger.info(
                    f"Step {self.step}: Loss={loss.item():.4f}, "
                    f"Recall={losses['recall_loss'].item():.4f}, "
                    f"Cost={losses['cost_loss'].item():.4f}"
                )
        
        # Compute epoch averages
        avg_loss = total_loss / num_batches
        avg_recall_loss = total_recall_loss / num_batches
        avg_cost_loss = total_cost_loss / num_batches
        
        self.train_losses.append(avg_loss)
        
        return {
            'train_loss': avg_loss,
            'train_recall_loss': avg_recall_loss,
            'train_cost_loss': avg_cost_loss
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary containing validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_recall_loss = 0.0
        total_cost_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                monitor_scores = batch.get('monitor_scores', None)
                if monitor_scores is not None:
                    monitor_scores = monitor_scores.to(self.device)
                
                # Forward pass
                outputs = self.model(features, monitor_scores, training=True)
                
                # Compute loss
                losses = self.loss_fn(outputs, labels, self.model.budget)
                
                # Update metrics
                total_loss += losses['total_loss'].item()
                total_recall_loss += losses['recall_loss'].item()
                total_cost_loss += losses['cost_loss'].item()
                num_batches += 1
        
        # Compute averages
        avg_loss = total_loss / num_batches
        avg_recall_loss = total_recall_loss / num_batches
        avg_cost_loss = total_cost_loss / num_batches
        
        self.val_losses.append(avg_loss)
        
        return {
            'val_loss': avg_loss,
            'val_recall_loss': avg_recall_loss,
            'val_cost_loss': avg_cost_loss
        }
    
    def train(
        self,
        epochs: int,
        save_best: bool = True,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            epochs: Number of epochs to train
            save_best: Whether to save the best model
            save_path: Path to save the model
            
        Returns:
            Dictionary containing training history
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss={train_metrics['train_loss']:.4f}, "
                f"Val Loss={val_metrics.get('val_loss', 'N/A'):.4f}"
            )
            
            # Save best model
            if save_best and val_metrics:
                val_loss = val_metrics['val_loss']
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if save_path:
                        self.save_checkpoint(save_path)
                    self.logger.info(f"New best model saved (val_loss={val_loss:.4f})")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'step': self.step,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']


class BaselineTrainer:
    """
    Trainer for baseline models (standard classification).
    
    This trains probes using standard classification losses
    without considering the hierarchical monitoring context.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        device: str = "cpu"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Initialize loss function
        if loss_fn is None:
            self.loss_fn = BaselineLoss("bce")
        else:
            self.loss_fn = loss_fn
        
        # Initialize optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            probe_scores = self.model(features).squeeze(-1)
            
            # Compute loss
            loss = self.loss_fn(probe_scores, labels.float())
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Validate the model."""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                probe_scores = self.model(features).squeeze(-1)
                loss = self.loss_fn(probe_scores, labels.float())
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, epochs: int) -> List[float]:
        """Train the model for multiple epochs."""
        train_losses = []
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            train_losses.append(train_loss)
            
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        return train_losses


# Example usage and testing
if __name__ == "__main__":
    # This would be used with actual data loaders
    print("Trainer classes implemented successfully!")
    print("Use with HierarchicalMonitor and appropriate data loaders.")
