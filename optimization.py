"""
Advanced optimization techniques for transformer training including
gradient clipping, learning rate scheduling, and adaptive optimizers.

This module provides sophisticated optimization algorithms and training
utilities for stable and efficient transformer training.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Callable, Tuple
from abc import ABC, abstractmethod
import math


class Optimizer(ABC):
    """Base class for all optimizers."""
    
    def __init__(self, learning_rate: float = 0.001):
        """
        Initialize the optimizer.
        
        Args:
            learning_rate (float): Learning rate for parameter updates.
        """
        self.learning_rate = learning_rate
        self.t = 0  # Time step
    
    @abstractmethod
    def update(self, params: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Update parameters based on gradients.
        
        Args:
            params (Dict[str, np.ndarray]): Current parameters.
            gradients (Dict[str, np.ndarray]): Parameter gradients.
            
        Returns:
            Dict[str, np.ndarray]: Updated parameters.
        """
        pass
    
    def zero_grad(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Reset gradients to zero."""
        return {key: np.zeros_like(grad) for key, grad in gradients.items()}


class SGDOptimizer(Optimizer):
    """Stochastic Gradient Descent optimizer with momentum."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, 
                 weight_decay: float = 0.0):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate (float): Learning rate.
            momentum (float): Momentum factor.
            weight_decay (float): Weight decay (L2 regularization).
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}
    
    def update(self, params: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Update parameters using SGD with momentum."""
        self.t += 1
        updated_params = {}
        
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            
            # Add weight decay
            grad = gradients[key]
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * params[key]
            
            # Update velocity
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grad
            
            # Update parameters
            updated_params[key] = params[key] + self.velocity[key]
        
        return updated_params


class AdamOptimizer(Optimizer):
    """Adam optimizer with bias correction."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate (float): Learning rate.
            beta1 (float): Exponential decay rate for first moment estimates.
            beta2 (float): Exponential decay rate for second moment estimates.
            eps (float): Small constant for numerical stability.
            weight_decay (float): Weight decay (L2 regularization).
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}  # First moment
        self.v = {}  # Second moment
    
    def update(self, params: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Update parameters using Adam algorithm."""
        self.t += 1
        updated_params = {}
        
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Add weight decay
            grad = gradients[key]
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * params[key]
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            updated_params[key] = params[key] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
        
        return updated_params


class AdamWOptimizer(AdamOptimizer):
    """AdamW optimizer with decoupled weight decay."""
    
    def update(self, params: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Update parameters using AdamW algorithm."""
        self.t += 1
        updated_params = {}
        
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            grad = gradients[key]
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters with decoupled weight decay
            updated_params[key] = params[key] - self.learning_rate * (
                m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * params[key]
            )
        
        return updated_params


class LearningRateScheduler(ABC):
    """Base class for learning rate schedulers."""
    
    @abstractmethod
    def get_lr(self, epoch: int, base_lr: float) -> float:
        """
        Get learning rate for the given epoch.
        
        Args:
            epoch (int): Current epoch.
            base_lr (float): Base learning rate.
            
        Returns:
            float: Scheduled learning rate.
        """
        pass


class CosineAnnealingScheduler(LearningRateScheduler):
    """Cosine annealing learning rate scheduler."""
    
    def __init__(self, T_max: int, eta_min: float = 0.0):
        """
        Initialize cosine annealing scheduler.
        
        Args:
            T_max (int): Maximum number of epochs.
            eta_min (float): Minimum learning rate.
        """
        self.T_max = T_max
        self.eta_min = eta_min
    
    def get_lr(self, epoch: int, base_lr: float) -> float:
        """Get cosine annealed learning rate."""
        return self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2


class ExponentialDecayScheduler(LearningRateScheduler):
    """Exponential decay learning rate scheduler."""
    
    def __init__(self, decay_rate: float = 0.95, decay_steps: int = 1000):
        """
        Initialize exponential decay scheduler.
        
        Args:
            decay_rate (float): Decay rate.
            decay_steps (int): Number of steps for each decay.
        """
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
    
    def get_lr(self, epoch: int, base_lr: float) -> float:
        """Get exponentially decayed learning rate."""
        return base_lr * (self.decay_rate ** (epoch // self.decay_steps))


class WarmupScheduler(LearningRateScheduler):
    """Learning rate warmup scheduler."""
    
    def __init__(self, warmup_epochs: int, base_scheduler: Optional[LearningRateScheduler] = None):
        """
        Initialize warmup scheduler.
        
        Args:
            warmup_epochs (int): Number of warmup epochs.
            base_scheduler (LearningRateScheduler, optional): Scheduler to use after warmup.
        """
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
    
    def get_lr(self, epoch: int, base_lr: float) -> float:
        """Get learning rate with warmup."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            return base_lr * (epoch + 1) / self.warmup_epochs
        elif self.base_scheduler:
            # Use base scheduler after warmup
            return self.base_scheduler.get_lr(epoch - self.warmup_epochs, base_lr)
        else:
            # Constant learning rate after warmup
            return base_lr


class GradientClipper:
    """Gradient clipping utilities for stable training."""
    
    @staticmethod
    def clip_by_norm(gradients: Dict[str, np.ndarray], max_norm: float) -> Dict[str, np.ndarray]:
        """
        Clip gradients by global norm.
        
        Args:
            gradients (Dict[str, np.ndarray]): Parameter gradients.
            max_norm (float): Maximum gradient norm.
            
        Returns:
            Dict[str, np.ndarray]: Clipped gradients.
        """
        # Calculate global norm
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)
        
        # Clip if necessary
        if total_norm > max_norm:
            clip_factor = max_norm / total_norm
            clipped_gradients = {}
            for key, grad in gradients.items():
                clipped_gradients[key] = grad * clip_factor
            return clipped_gradients
        
        return gradients
    
    @staticmethod
    def clip_by_value(gradients: Dict[str, np.ndarray], min_value: float, max_value: float) -> Dict[str, np.ndarray]:
        """
        Clip gradients by value.
        
        Args:
            gradients (Dict[str, np.ndarray]): Parameter gradients.
            min_value (float): Minimum gradient value.
            max_value (float): Maximum gradient value.
            
        Returns:
            Dict[str, np.ndarray]: Clipped gradients.
        """
        clipped_gradients = {}
        for key, grad in gradients.items():
            clipped_gradients[key] = np.clip(grad, min_value, max_value)
        return clipped_gradients
    
    @staticmethod
    def get_gradient_norm(gradients: Dict[str, np.ndarray]) -> float:
        """
        Calculate the global gradient norm.
        
        Args:
            gradients (Dict[str, np.ndarray]): Parameter gradients.
            
        Returns:
            float: Global gradient norm.
        """
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += np.sum(grad ** 2)
        return np.sqrt(total_norm)


class TrainingOptimizer:
    """
    High-level training optimizer that combines various optimization techniques.
    
    Integrates optimizer, learning rate scheduler, and gradient clipping
    for comprehensive training optimization.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        scheduler: Optional[LearningRateScheduler] = None,
        gradient_clipping: Optional[Dict[str, Any]] = None,
        accumulation_steps: int = 1
    ):
        """
        Initialize training optimizer.
        
        Args:
            optimizer (Optimizer): Base optimizer.
            scheduler (LearningRateScheduler, optional): Learning rate scheduler.
            gradient_clipping (Dict[str, Any], optional): Gradient clipping configuration.
            accumulation_steps (int): Number of steps to accumulate gradients.
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clipping = gradient_clipping or {}
        self.accumulation_steps = accumulation_steps
        
        # Gradient accumulation
        self.accumulated_gradients = {}
        self.accumulation_count = 0
        
        # Tracking
        self.step_count = 0
        self.epoch_count = 0
        self.gradient_norms = []
        self.learning_rates = []
    
    def step(self, params: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Perform one optimization step.
        
        Args:
            params (Dict[str, np.ndarray]): Current parameters.
            gradients (Dict[str, np.ndarray]): Parameter gradients.
            
        Returns:
            Dict[str, np.ndarray]: Updated parameters.
        """
        # Accumulate gradients
        if not self.accumulated_gradients:
            self.accumulated_gradients = {key: np.zeros_like(grad) for key, grad in gradients.items()}
        
        for key in gradients:
            self.accumulated_gradients[key] += gradients[key] / self.accumulation_steps
        
        self.accumulation_count += 1
        
        # Update parameters when accumulation is complete
        if self.accumulation_count >= self.accumulation_steps:
            # Apply gradient clipping
            if self.gradient_clipping:
                if 'method' in self.gradient_clipping:
                    if self.gradient_clipping['method'] == 'norm':
                        max_norm = self.gradient_clipping.get('max_norm', 1.0)
                        self.accumulated_gradients = GradientClipper.clip_by_norm(
                            self.accumulated_gradients, max_norm
                        )
                    elif self.gradient_clipping['method'] == 'value':
                        min_val = self.gradient_clipping.get('min_value', -1.0)
                        max_val = self.gradient_clipping.get('max_value', 1.0)
                        self.accumulated_gradients = GradientClipper.clip_by_value(
                            self.accumulated_gradients, min_val, max_val
                        )
            
            # Track gradient norm
            grad_norm = GradientClipper.get_gradient_norm(self.accumulated_gradients)
            self.gradient_norms.append(grad_norm)
            
            # Update learning rate
            if self.scheduler:
                current_lr = self.scheduler.get_lr(self.epoch_count, self.optimizer.learning_rate)
                self.optimizer.learning_rate = current_lr
            
            self.learning_rates.append(self.optimizer.learning_rate)
            
            # Update parameters
            updated_params = self.optimizer.update(params, self.accumulated_gradients)
            
            # Reset accumulation
            self.accumulated_gradients = {key: np.zeros_like(grad) for key, grad in gradients.items()}
            self.accumulation_count = 0
            self.step_count += 1
            
            return updated_params
        
        return params
    
    def epoch_end(self):
        """Mark the end of an epoch."""
        self.epoch_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics.
        
        Returns:
            Dict[str, Any]: Optimization statistics.
        """
        return {
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'current_lr': self.optimizer.learning_rate,
            'gradient_norms': self.gradient_norms,
            'learning_rates': self.learning_rates,
            'avg_gradient_norm': np.mean(self.gradient_norms) if self.gradient_norms else 0.0,
            'max_gradient_norm': np.max(self.gradient_norms) if self.gradient_norms else 0.0
        }


class EarlyStopping:
    """Early stopping callback to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = 'min', restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience (int): Number of epochs to wait before stopping.
            min_delta (float): Minimum change to qualify as improvement.
            mode (str): 'min' for decreasing metrics, 'max' for increasing.
            restore_best_weights (bool): Whether to restore best weights on stop.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.stopped = False
    
    def __call__(self, score: float, model_weights: Optional[Dict[str, np.ndarray]] = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            score (float): Current validation score.
            model_weights (Dict[str, np.ndarray], optional): Current model weights.
            
        Returns:
            bool: True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            if model_weights is not None:
                self.best_weights = {k: v.copy() for k, v in model_weights.items()}
        
        improved = False
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            if model_weights is not None:
                self.best_weights = {k: v.copy() for k, v in model_weights.items()}
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.stopped = True
        
        return self.stopped
    
    def get_best_weights(self) -> Optional[Dict[str, np.ndarray]]:
        """Get the best model weights."""
        return self.best_weights


class OptimizerFactory:
    """Factory for creating optimizers and schedulers."""
    
    @staticmethod
    def create_optimizer(optimizer_type: str, **kwargs) -> Optimizer:
        """
        Create an optimizer.
        
        Args:
            optimizer_type (str): Type of optimizer ('sgd', 'adam', 'adamw').
            **kwargs: Optimizer-specific arguments.
            
        Returns:
            Optimizer: Created optimizer.
        """
        if optimizer_type.lower() == 'sgd':
            return SGDOptimizer(**kwargs)
        elif optimizer_type.lower() == 'adam':
            return AdamOptimizer(**kwargs)
        elif optimizer_type.lower() == 'adamw':
            return AdamWOptimizer(**kwargs)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    @staticmethod
    def create_scheduler(scheduler_type: str, **kwargs) -> LearningRateScheduler:
        """
        Create a learning rate scheduler.
        
        Args:
            scheduler_type (str): Type of scheduler.
            **kwargs: Scheduler-specific arguments.
            
        Returns:
            LearningRateScheduler: Created scheduler.
        """
        if scheduler_type.lower() == 'cosine':
            return CosineAnnealingScheduler(**kwargs)
        elif scheduler_type.lower() == 'exponential':
            return ExponentialDecayScheduler(**kwargs)
        elif scheduler_type.lower() == 'warmup':
            return WarmupScheduler(**kwargs)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    @staticmethod
    def create_training_optimizer(config: Dict[str, Any]) -> TrainingOptimizer:
        """
        Create a complete training optimizer from configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary.
            
        Returns:
            TrainingOptimizer: Configured training optimizer.
        """
        # Create base optimizer
        optimizer_config = config.get('optimizer', {})
        optimizer_type = optimizer_config.pop('type', 'adam')
        optimizer = OptimizerFactory.create_optimizer(optimizer_type, **optimizer_config)
        
        # Create scheduler
        scheduler = None
        if 'scheduler' in config:
            scheduler_config = config['scheduler'].copy()
            scheduler_type = scheduler_config.pop('type')
            scheduler = OptimizerFactory.create_scheduler(scheduler_type, **scheduler_config)
        
        # Get other configuration
        gradient_clipping = config.get('gradient_clipping', {})
        accumulation_steps = config.get('accumulation_steps', 1)
        
        return TrainingOptimizer(
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_clipping=gradient_clipping,
            accumulation_steps=accumulation_steps
        )