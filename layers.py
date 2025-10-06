"""
Advanced Transformer Layer Components

This module contains essential transformer building blocks including layer normalization,
feedforward networks, and dropout for building complete transformer architectures.
"""

import numpy as np
from typing import Optional, Tuple


class LayerNormalization:
    """
    Layer Normalization implementation for stabilizing transformer training.
    
    Layer normalization normalizes inputs across the feature dimension rather than
    the batch dimension, which is more suitable for sequence models.
    
    Attributes:
        normalized_shape (int): Size of the feature dimension to normalize.
        eps (float): Small epsilon value for numerical stability.
        gamma (np.ndarray): Learnable scale parameter.
        beta (np.ndarray): Learnable shift parameter.
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        """
        Initialize LayerNormalization.
        
        Args:
            normalized_shape (int): Size of the feature dimension.
            eps (float): Small value for numerical stability.
            
        Raises:
            ValueError: If normalized_shape is not positive or eps is not positive.
        """
        if not isinstance(normalized_shape, int) or normalized_shape <= 0:
            raise ValueError(f"normalized_shape must be a positive integer, got {normalized_shape}")
        
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
            
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Initialize learnable parameters
        self.gamma = np.ones(normalized_shape)  # Scale parameter
        self.beta = np.zeros(normalized_shape)  # Shift parameter
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply layer normalization to input.
        
        Args:
            x (np.ndarray): Input tensor of shape (..., normalized_shape).
            
        Returns:
            np.ndarray: Layer-normalized output with same shape as input.
            
        Raises:
            TypeError: If x is not a numpy array.
            ValueError: If x has incorrect shape.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(x)}")
        
        if x.shape[-1] != self.normalized_shape:
            raise ValueError(f"Last dimension must be {self.normalized_shape}, got {x.shape[-1]}")
        
        # Calculate mean and variance along the last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_normalized = (x - mean) / np.sqrt(var + self.eps)
        
        # Apply learnable parameters
        return self.gamma * x_normalized + self.beta


class FeedForward:
    """
    Position-wise feedforward network used in transformer blocks.
    
    This is a simple two-layer fully connected network with ReLU activation,
    commonly used after attention layers in transformers.
    
    Attributes:
        d_model (int): Input/output dimension.
        d_ff (int): Hidden dimension (typically 4 * d_model).
        W1 (np.ndarray): First linear transformation weight matrix.
        b1 (np.ndarray): First linear transformation bias.
        W2 (np.ndarray): Second linear transformation weight matrix.
        b2 (np.ndarray): Second linear transformation bias.
    """
    
    def __init__(self, d_model: int, d_ff: int):
        """
        Initialize FeedForward network.
        
        Args:
            d_model (int): Input and output dimension.
            d_ff (int): Hidden dimension.
            
        Raises:
            ValueError: If dimensions are not positive integers.
        """
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {d_model}")
        
        if not isinstance(d_ff, int) or d_ff <= 0:
            raise ValueError(f"d_ff must be a positive integer, got {d_ff}")
            
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Xavier initialization for weights
        xavier_std_1 = np.sqrt(2.0 / (d_model + d_ff))
        xavier_std_2 = np.sqrt(2.0 / (d_ff + d_model))
        
        self.W1 = np.random.normal(0, xavier_std_1, (d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.normal(0, xavier_std_2, (d_ff, d_model))
        self.b2 = np.zeros(d_model)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the feedforward network.
        
        Args:
            x (np.ndarray): Input tensor of shape (..., d_model).
            
        Returns:
            np.ndarray: Output tensor of same shape as input.
            
        Raises:
            TypeError: If x is not a numpy array.
            ValueError: If x has incorrect shape.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(x)}")
        
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Last dimension must be {self.d_model}, got {x.shape[-1]}")
        
        # First linear transformation + ReLU
        hidden = np.maximum(0, np.dot(x, self.W1) + self.b1)  # ReLU activation
        
        # Second linear transformation
        output = np.dot(hidden, self.W2) + self.b2
        
        return output


class Dropout:
    """
    Dropout regularization for preventing overfitting.
    
    Randomly sets a fraction of input units to 0 during training,
    which helps prevent overfitting and improves generalization.
    
    Attributes:
        p (float): Dropout probability (fraction of units to drop).
        training (bool): Whether the model is in training mode.
    """
    
    def __init__(self, p: float = 0.1):
        """
        Initialize Dropout layer.
        
        Args:
            p (float): Dropout probability (0.0 to 1.0).
            
        Raises:
            ValueError: If p is not in valid range.
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Dropout probability must be in [0.0, 1.0], got {p}")
            
        self.p = p
        self.training = True
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply dropout to input tensor.
        
        Args:
            x (np.ndarray): Input tensor.
            
        Returns:
            np.ndarray: Output tensor with dropout applied (if training).
            
        Raises:
            TypeError: If x is not a numpy array.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(x)}")
        
        if not self.training or self.p == 0.0:
            return x
        
        # Generate dropout mask
        keep_prob = 1.0 - self.p
        mask = np.random.binomial(1, keep_prob, x.shape)
        
        # Apply mask and scale by keep_prob to maintain expected value
        return x * mask / keep_prob
    
    def eval(self):
        """Set dropout to evaluation mode (no dropout applied)."""
        self.training = False
        
    def train(self):
        """Set dropout to training mode (dropout applied)."""
        self.training = True


def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit (GELU) activation function.
    
    GELU is a smooth activation function that's often used in modern transformers
    instead of ReLU. It provides better gradient flow and performance.
    
    Args:
        x (np.ndarray): Input tensor.
        
    Returns:
        np.ndarray: GELU-activated output.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class FeedForwardGELU(FeedForward):
    """
    Feedforward network with GELU activation instead of ReLU.
    
    This variant uses GELU activation which is commonly used in modern
    transformer architectures like BERT and GPT.
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the feedforward network with GELU activation.
        
        Args:
            x (np.ndarray): Input tensor of shape (..., d_model).
            
        Returns:
            np.ndarray: Output tensor of same shape as input.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(x)}")
        
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Last dimension must be {self.d_model}, got {x.shape[-1]}")
        
        # First linear transformation + GELU
        hidden = gelu(np.dot(x, self.W1) + self.b1)
        
        # Second linear transformation
        output = np.dot(hidden, self.W2) + self.b2
        
        return output