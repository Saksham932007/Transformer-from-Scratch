"""
Training Utilities for Transformer Models

This module provides comprehensive training utilities including optimizers,
learning rate schedulers, loss functions, and training loops.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from config import TransformerConfig, ModelCheckpoint, ExperimentTracker
import time
import math


class AdamOptimizer:
    """
    Adam optimizer implementation for training transformer models.
    
    Adam combines the advantages of AdaGrad and RMSProp optimizers,
    maintaining per-parameter learning rates and momentum estimates.
    
    Attributes:
        learning_rate (float): Learning rate.
        beta1 (float): Exponential decay rate for first moment estimates.
        beta2 (float): Exponential decay rate for second moment estimates.
        epsilon (float): Small constant for numerical stability.
        weight_decay (float): L2 regularization strength.
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8, 
                 weight_decay: float = 0.0):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate.
            beta1: Exponential decay rate for first moment estimates.
            beta2: Exponential decay rate for second moment estimates.
            epsilon: Small constant for numerical stability.
            weight_decay: L2 regularization strength.
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        # State variables
        self.t = 0  # Time step
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
    
    def step(self, parameters: Dict[str, np.ndarray], 
             gradients: Dict[str, np.ndarray]):
        """
        Perform a single optimization step.
        
        Args:
            parameters: Dictionary of parameter arrays.
            gradients: Dictionary of gradient arrays.
        """
        self.t += 1
        
        for name, param in parameters.items():
            if name not in gradients:
                continue
                
            grad = gradients[name]
            
            # Add weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Initialize moment estimates if first time
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)
            
            # Update biased first moment estimate
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected moment estimates
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def zero_grad(self):
        """Reset optimizer state (called before backward pass)."""
        pass  # In our simplified implementation, gradients are computed fresh each time


class LearningRateScheduler:
    """
    Learning rate scheduler with warmup and decay strategies.
    """
    
    def __init__(self, optimizer: AdamOptimizer, warmup_steps: int = 4000, 
                 d_model: int = 512, decay_type: str = "transformer"):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule.
            warmup_steps: Number of warmup steps.
            d_model: Model dimension (for transformer scheduling).
            decay_type: Type of decay ('transformer', 'cosine', 'linear').
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.decay_type = decay_type
        self.initial_lr = optimizer.learning_rate
        self.step_count = 0
    
    def step(self):
        """Update learning rate based on current step."""
        self.step_count += 1
        
        if self.decay_type == "transformer":
            # Transformer learning rate schedule with warmup
            lr = (self.d_model ** -0.5) * min(
                self.step_count ** -0.5,
                self.step_count * (self.warmup_steps ** -1.5)
            )
        elif self.decay_type == "cosine":
            # Cosine annealing
            if self.step_count <= self.warmup_steps:
                lr = self.initial_lr * self.step_count / self.warmup_steps
            else:
                progress = (self.step_count - self.warmup_steps) / (10000 - self.warmup_steps)
                lr = self.initial_lr * 0.5 * (1 + math.cos(math.pi * progress))
        else:  # linear
            if self.step_count <= self.warmup_steps:
                lr = self.initial_lr * self.step_count / self.warmup_steps
            else:
                lr = self.initial_lr * (1 - (self.step_count - self.warmup_steps) / 10000)
        
        self.optimizer.learning_rate = max(lr, 1e-7)  # Minimum learning rate
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.learning_rate


class LossFunction:
    """
    Collection of loss functions for transformer training.
    """
    
    @staticmethod
    def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray, 
                          label_smoothing: float = 0.0) -> Tuple[float, np.ndarray]:
        """
        Compute cross-entropy loss with optional label smoothing.
        
        Args:
            logits: Model predictions of shape (batch_size, vocab_size).
            targets: Target labels of shape (batch_size,).
            label_smoothing: Label smoothing factor.
            
        Returns:
            Tuple of (loss_value, gradients).
        """
        batch_size, vocab_size = logits.shape
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Create one-hot targets
        one_hot_targets = np.zeros((batch_size, vocab_size))
        one_hot_targets[np.arange(batch_size), targets] = 1.0
        
        # Apply label smoothing
        if label_smoothing > 0:
            smooth_targets = (1 - label_smoothing) * one_hot_targets + \
                           label_smoothing / vocab_size
        else:
            smooth_targets = one_hot_targets
        
        # Compute loss
        loss = -np.sum(smooth_targets * np.log(probs + 1e-12)) / batch_size
        
        # Compute gradients
        gradients = (probs - smooth_targets) / batch_size
        
        return loss, gradients
    
    @staticmethod
    def masked_cross_entropy_loss(logits: np.ndarray, targets: np.ndarray, 
                                 mask: np.ndarray, label_smoothing: float = 0.0) -> Tuple[float, np.ndarray]:
        """
        Compute masked cross-entropy loss for sequences with padding.
        
        Args:
            logits: Model predictions of shape (batch_size, seq_len, vocab_size).
            targets: Target labels of shape (batch_size, seq_len).
            mask: Binary mask of shape (batch_size, seq_len).
            label_smoothing: Label smoothing factor.
            
        Returns:
            Tuple of (loss_value, gradients).
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for easier computation
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        mask_flat = mask.reshape(-1)
        
        # Apply softmax
        exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Create one-hot targets
        one_hot_targets = np.zeros((batch_size * seq_len, vocab_size))
        valid_indices = mask_flat.astype(bool)
        one_hot_targets[valid_indices, targets_flat[valid_indices]] = 1.0
        
        # Apply label smoothing
        if label_smoothing > 0:
            smooth_targets = (1 - label_smoothing) * one_hot_targets + \
                           label_smoothing / vocab_size
        else:
            smooth_targets = one_hot_targets
        
        # Compute loss only for valid positions
        losses = -np.sum(smooth_targets * np.log(probs + 1e-12), axis=-1)
        masked_losses = losses * mask_flat
        loss = np.sum(masked_losses) / np.sum(mask_flat)
        
        # Compute gradients
        gradients = (probs - smooth_targets) * mask_flat[:, np.newaxis]
        gradients = gradients.reshape(batch_size, seq_len, vocab_size)
        
        return loss, gradients


class Trainer:
    """
    Comprehensive trainer for transformer models.
    """
    
    def __init__(self, model, config: TransformerConfig):
        """
        Initialize trainer.
        
        Args:
            model: Transformer model to train.
            config: Training configuration.
        """
        self.model = model
        self.config = config
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamOptimizer(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = LearningRateScheduler(
            self.optimizer, 
            warmup_steps=config.warmup_steps,
            d_model=config.d_model
        )
        
        # Initialize tracking
        self.checkpoint_manager = ModelCheckpoint(config.model_save_path)
        self.experiment_tracker = ExperimentTracker(
            experiment_name=f"transformer_{int(time.time())}",
            log_dir=config.log_path
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Log initial configuration
        self.experiment_tracker.log_config(config)
    
    def compute_gradients(self, loss_fn: Callable, inputs: np.ndarray, 
                         targets: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute gradients using numerical differentiation (simplified).
        
        In a real implementation, you would use automatic differentiation.
        
        Args:
            loss_fn: Loss function to differentiate.
            inputs: Input data.
            targets: Target data.
            
        Returns:
            Dictionary of gradients for each parameter.
        """
        # This is a simplified gradient computation
        # In practice, you'd use automatic differentiation
        gradients = {}
        eps = 1e-7
        
        # Get current loss
        current_loss, _ = loss_fn(inputs, targets)
        
        # Numerical gradients for output projection (simplified)
        if hasattr(self.model, 'output_projection'):
            grad_shape = self.model.output_projection.shape
            grad = np.zeros(grad_shape)
            
            # Compute gradients for a few random elements (for demonstration)
            for _ in range(min(10, grad.size)):
                i, j = np.random.randint(0, grad_shape[0]), np.random.randint(0, grad_shape[1])
                
                # Forward perturbation
                self.model.output_projection[i, j] += eps
                loss_plus, _ = loss_fn(inputs, targets)
                
                # Backward perturbation
                self.model.output_projection[i, j] -= 2 * eps
                loss_minus, _ = loss_fn(inputs, targets)
                
                # Restore original value
                self.model.output_projection[i, j] += eps
                
                # Numerical gradient
                grad[i, j] = (loss_plus - loss_minus) / (2 * eps)
            
            gradients['output_projection'] = grad
        
        return gradients
    
    def train_step(self, batch_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch_data: Tuple of (inputs, targets).
            
        Returns:
            Dictionary of metrics for this step.
        """
        inputs, targets = batch_data
        
        # Set model to training mode
        self.model.set_training_mode(True)
        
        # Forward pass
        outputs = self.model.forward(inputs)
        
        # Compute loss
        loss, _ = LossFunction.cross_entropy_loss(
            outputs[-1].reshape(1, -1), 
            targets.reshape(-1),
            label_smoothing=self.config.label_smoothing
        )
        
        # Compute gradients (simplified)
        def loss_fn(inp, tgt):
            out = self.model.forward(inp)
            return LossFunction.cross_entropy_loss(
                out[-1].reshape(1, -1), 
                tgt.reshape(-1),
                label_smoothing=self.config.label_smoothing
            )
        
        gradients = self.compute_gradients(loss_fn, inputs, targets)
        
        # Clip gradients
        if self.config.gradient_clip_norm > 0:
            total_norm = 0
            for grad in gradients.values():
                total_norm += np.sum(grad ** 2)
            total_norm = np.sqrt(total_norm)
            
            if total_norm > self.config.gradient_clip_norm:
                clip_factor = self.config.gradient_clip_norm / total_norm
                for name in gradients:
                    gradients[name] *= clip_factor
        
        # Optimizer step
        parameters = {}
        if hasattr(self.model, 'output_projection'):
            parameters['output_projection'] = self.model.output_projection
        
        self.optimizer.step(parameters, gradients)
        self.scheduler.step()
        
        # Update global step
        self.global_step += 1
        
        # Prepare metrics
        metrics = {
            'loss': loss,
            'learning_rate': self.scheduler.get_lr(),
            'global_step': self.global_step
        }
        
        return metrics
    
    def train_epoch(self, train_data: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_data: List of (input, target) tuples.
            
        Returns:
            Dictionary of epoch metrics.
        """
        epoch_losses = []
        
        for batch_idx, batch_data in enumerate(train_data):
            metrics = self.train_step(batch_data)
            epoch_losses.append(metrics['loss'])
            
            # Log metrics
            if self.global_step % 100 == 0:
                self.experiment_tracker.log_metrics(self.global_step, metrics)
                print(f"Step {self.global_step}: Loss = {metrics['loss']:.4f}, "
                      f"LR = {metrics['learning_rate']:.2e}")
            
            # Save checkpoint
            if self.global_step % self.config.save_every_n_steps == 0:
                self.save_checkpoint(metrics['loss'])
        
        avg_loss = np.mean(epoch_losses)
        
        return {
            'epoch': self.epoch,
            'avg_loss': avg_loss,
            'num_batches': len(train_data)
        }
    
    def evaluate(self, eval_data: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """
        Evaluate model on validation data.
        
        Args:
            eval_data: List of (input, target) tuples for evaluation.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.set_training_mode(False)
        
        eval_losses = []
        
        for batch_data in eval_data:
            inputs, targets = batch_data
            
            # Forward pass
            outputs = self.model.forward(inputs)
            
            # Compute loss
            loss, _ = LossFunction.cross_entropy_loss(
                outputs[-1].reshape(1, -1), 
                targets.reshape(-1)
            )
            
            eval_losses.append(loss)
        
        avg_loss = np.mean(eval_losses)
        
        return {
            'eval_loss': avg_loss,
            'num_eval_batches': len(eval_data)
        }
    
    def save_checkpoint(self, current_loss: float):
        """
        Save model checkpoint if it's the best so far.
        
        Args:
            current_loss: Current validation loss.
        """
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            
            self.checkpoint_manager.save(
                model=self.model,
                config=self.config,
                step=self.global_step,
                loss=current_loss,
                metrics={'best_loss': self.best_loss}
            )
    
    def fit(self, train_data: List[Tuple[np.ndarray, np.ndarray]], 
            eval_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
            num_epochs: Optional[int] = None):
        """
        Train the model for multiple epochs.
        
        Args:
            train_data: Training data.
            eval_data: Evaluation data (optional).
            num_epochs: Number of epochs (uses config if not provided).
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model: {self.config.d_model}d, {self.config.num_layers} layers, "
              f"{self.config.num_heads} heads")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_data)
            
            print(f"Epoch {epoch + 1}/{num_epochs}: "
                  f"Train Loss = {train_metrics['avg_loss']:.4f}")
            
            # Evaluate if data provided
            if eval_data is not None and (epoch + 1) % 5 == 0:
                eval_metrics = self.evaluate(eval_data)
                print(f"Eval Loss = {eval_metrics['eval_loss']:.4f}")
                
                # Log evaluation metrics
                combined_metrics = {**train_metrics, **eval_metrics}
                self.experiment_tracker.log_metrics(self.global_step, combined_metrics)
        
        print("Training completed!")
        
        # Save final checkpoint
        final_metrics = self.evaluate(eval_data) if eval_data else {'final_loss': train_metrics['avg_loss']}
        self.checkpoint_manager.save(
            model=self.model,
            config=self.config,
            step=self.global_step,
            loss=final_metrics.get('eval_loss', train_metrics['avg_loss']),
            metrics=final_metrics
        )