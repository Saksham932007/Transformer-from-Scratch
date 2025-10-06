import numpy as np
from typing import Optional, Tuple, Union
from embed import get_embedding

def softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Apply softmax to normalize attention scores with numerical stability.

    Args:
        scores (np.ndarray): Attention scores.
        axis (int): Axis along which to apply softmax.

    Returns:
        np.ndarray: Normalized attention scores.
        
    Raises:
        TypeError: If scores is not a numpy array.
        ValueError: If scores contains invalid values.
    """
    if not isinstance(scores, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(scores)}")
    
    if scores.size == 0:
        raise ValueError("Input scores cannot be empty")
    
    # Numerical stability: subtract max to prevent overflow
    scores_shifted = scores - np.max(scores, axis=axis, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    return exp_scores / np.sum(exp_scores, axis=axis, keepdims=True)



class SelfAttention:
    """
    Self-Attention mechanism to compute attention scores and apply them to value vectors.
    
    This implementation uses scaled dot-product attention with improved numerical stability
    and better weight initialization for faster convergence.

    Attributes:
        embedding_dim (int): Dimension of the embeddings.
        W_q (np.ndarray): Weight matrix for the Query projection.
        W_k (np.ndarray): Weight matrix for the Key projection.
        W_v (np.ndarray): Weight matrix for the Value projection.
        scale_factor (float): Scaling factor for attention scores (1/sqrt(d_k)).
    """

    def __init__(self, embedding_dim: int):
        """
        Initialize the SelfAttention mechanism with Xavier/Glorot initialization.

        Args:
            embedding_dim (int): Dimension of the embeddings.
            
        Raises:
            ValueError: If embedding_dim is not a positive integer.
        """
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be a positive integer, got {embedding_dim}")
            
        self.embedding_dim = embedding_dim
        self.scale_factor = 1.0 / np.sqrt(embedding_dim)

        # Xavier/Glorot initialization for better convergence
        xavier_std = np.sqrt(2.0 / (embedding_dim + embedding_dim))
        self.W_q = np.random.normal(0, xavier_std, (embedding_dim, embedding_dim))
        self.W_k = np.random.normal(0, xavier_std, (embedding_dim, embedding_dim))
        self.W_v = np.random.normal(0, xavier_std, (embedding_dim, embedding_dim))

    def forward(self, embeddings: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through the Self-Attention mechanism.

        Args:
            embeddings (np.ndarray): Input embeddings of shape (seq_len, embedding_dim).
            mask (np.ndarray, optional): Attention mask of shape (seq_len, seq_len).

        Returns:
            np.ndarray: Output after applying attention to value vectors.
            
        Raises:
            TypeError: If embeddings is not a numpy array.
            ValueError: If embeddings has incorrect shape or mask is incompatible.
        """
        if not isinstance(embeddings, np.ndarray):
            raise TypeError(f"Expected numpy array for embeddings, got {type(embeddings)}")
        
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got {embeddings.ndim}D")
        
        seq_len, emb_dim = embeddings.shape
        if emb_dim != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {emb_dim}")
        
        if mask is not None:
            if not isinstance(mask, np.ndarray):
                raise TypeError(f"Expected numpy array for mask, got {type(mask)}")
            if mask.shape != (seq_len, seq_len):
                raise ValueError(f"Mask shape {mask.shape} incompatible with sequence length {seq_len}")

        # Compute Q, K, V projections
        query = np.dot(embeddings, self.W_q)
        key = np.dot(embeddings, self.W_k)
        values = np.dot(embeddings, self.W_v)

        # Calculate attention scores with proper scaling
        attention_scores = self.calculate_attention_score(query, key)

        # Apply mask if provided
        if mask is not None:
            attention_scores = np.where(mask == 0, -np.inf, attention_scores)

        # Apply softmax to attention scores
        attention_weights = softmax(attention_scores)

        # Compute weighted sum of value vectors
        output = self.values_weighted_sum(attention_weights, values)

        return output

    def calculate_attention_score(self, query: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        Calculate the scaled dot-product attention scores.

        Args:
            query (np.ndarray): Query matrix of shape (seq_len, embedding_dim).
            key (np.ndarray): Key matrix of shape (seq_len, embedding_dim).

        Returns:
            np.ndarray: Scaled attention scores of shape (seq_len, seq_len).
        """
        # Compute dot product and scale
        scores = np.dot(query, key.T) * self.scale_factor
        return scores
    
    def values_weighted_sum(self, weights: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Calculate the weighted sum of value vectors based on attention weights.

        Args:
            weights (np.ndarray): Attention weights of shape (seq_len, seq_len).
            values (np.ndarray): Value vectors of shape (seq_len, embedding_dim).

        Returns:
            np.ndarray: Weighted sum of value vectors of shape (seq_len, embedding_dim).
        """
        return np.dot(weights, values)

class MultiHeadAttention:
    """
    Multi-Head Attention mechanism consisting of multiple self-attention heads.
    
    This implementation provides parallel computation of multiple attention heads
    with improved weight initialization and better error handling.

    Attributes:
        embedding_dim (int): Total embedding dimension.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        attention_heads (list): List of SelfAttention instances.
        W_o (np.ndarray): Final output projection matrix.
    """

    def __init__(self, embedding_dim: int, num_heads: int):
        """
        Initialize the MultiHeadAttention mechanism.

        Args:
            embedding_dim (int): Dimension of the embeddings.
            num_heads (int): Number of attention heads.

        Raises:
            ValueError: If embedding_dim is not divisible by num_heads or invalid parameters.
            TypeError: If parameters are not integers.
        """
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise TypeError(f"embedding_dim must be a positive integer, got {embedding_dim}")
        
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise TypeError(f"num_heads must be a positive integer, got {num_heads}")
        
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Initialize attention heads
        self.attention_heads = [SelfAttention(self.head_dim) for _ in range(num_heads)]
        
        # Xavier initialization for output projection
        xavier_std = np.sqrt(2.0 / (embedding_dim + embedding_dim))
        self.W_o = np.random.normal(0, xavier_std, (embedding_dim, embedding_dim))

    def forward(self, embeddings: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through the Multi-Head Attention mechanism.

        Args:
            embeddings (np.ndarray): Input embeddings of shape (seq_len, embedding_dim).
            mask (np.ndarray, optional): Attention mask of shape (seq_len, seq_len).

        Returns:
            np.ndarray: Output after applying multi-head attention and final transformation.
            
        Raises:
            TypeError: If embeddings is not a numpy array.
            ValueError: If embeddings has incorrect shape.
        """
        if not isinstance(embeddings, np.ndarray):
            raise TypeError(f"Expected numpy array for embeddings, got {type(embeddings)}")
        
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got {embeddings.ndim}D")
        
        seq_len, emb_dim = embeddings.shape
        if emb_dim != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {emb_dim}")

        # Split embeddings into heads more efficiently
        # Reshape to (seq_len, num_heads, head_dim)
        embeddings_reshaped = embeddings.reshape(seq_len, self.num_heads, self.head_dim)

        # Process each head
        head_outputs = []
        for i, head in enumerate(self.attention_heads):
            head_input = embeddings_reshaped[:, i, :]  # Shape: (seq_len, head_dim)
            head_output = head.forward(head_input, mask)
            head_outputs.append(head_output)
        
        # Concatenate outputs of all heads along the last axis
        concatenated_output = np.concatenate(head_outputs, axis=-1)
        
        # Apply final linear transformation
        output = self.linear_transformation(concatenated_output, self.W_o)
        
        return output

    def linear_transformation(self, x: np.ndarray, weight_matrix: np.ndarray) -> np.ndarray:
        """
        Apply a linear transformation to the input.

        Args:
            x (np.ndarray): Input tensor of shape (seq_len, embedding_dim).
            weight_matrix (np.ndarray): Weight matrix of shape (embedding_dim, embedding_dim).

        Returns:
            np.ndarray: Transformed output of shape (seq_len, embedding_dim).
        """
        return np.dot(x, weight_matrix)