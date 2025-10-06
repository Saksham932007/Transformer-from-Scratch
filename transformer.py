import numpy as np
from typing import Optional, List, Tuple
from attention import MultiHeadAttention, softmax
from embed import tokenize_and_embed, add_positional_encoding, embedding_model
from layers import LayerNormalization, FeedForward, Dropout
import random


class TransformerBlock:
    """
    A complete transformer block with multi-head attention, layer normalization,
    and feedforward network with residual connections.
    
    Attributes:
        d_model (int): Model dimension.
        num_heads (int): Number of attention heads.
        d_ff (int): Feedforward network hidden dimension.
        dropout_prob (float): Dropout probability.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: Optional[int] = None, dropout_prob: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            d_model (int): Model dimension.
            num_heads (int): Number of attention heads.
            d_ff (int, optional): Feedforward hidden dimension. Defaults to 4 * d_model.
            dropout_prob (float): Dropout probability.
        """
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Components
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm1 = LayerNormalization(d_model)
        self.feedforward = FeedForward(d_model, d_ff)
        self.layer_norm2 = LayerNormalization(d_model)
        self.dropout = Dropout(dropout_prob)
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through transformer block.
        
        Args:
            x (np.ndarray): Input embeddings of shape (seq_len, d_model).
            mask (np.ndarray, optional): Attention mask.
            
        Returns:
            np.ndarray: Output of transformer block.
        """
        # Multi-head attention with residual connection and layer norm
        attn_output = self.multi_head_attention.forward(x, mask)
        attn_output = self.dropout.forward(attn_output)
        x = self.layer_norm1.forward(x + attn_output)
        
        # Feedforward with residual connection and layer norm
        ff_output = self.feedforward.forward(x)
        ff_output = self.dropout.forward(ff_output)
        output = self.layer_norm2.forward(x + ff_output)
        
        return output


class Transformer:
    """
    Complete transformer model with multiple transformer blocks and improved architecture.
    
    This implementation includes proper layer normalization, residual connections,
    dropout for regularization, and better weight initialization.
    
    Attributes:
        d_model (int): Model dimension (embedding dimension).
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer blocks.
        d_ff (int): Feedforward network hidden dimension.
        dropout_prob (float): Dropout probability.
        max_seq_len (int): Maximum sequence length for positional encoding caching.
    """
    
    def __init__(self, d_model: int, num_heads: int, num_layers: int = 6, 
                 d_ff: Optional[int] = None, dropout_prob: float = 0.1, 
                 max_seq_len: int = 5000):
        """
        Initialize the Transformer model.
        
        Args:
            d_model (int): Model dimension (should match embedding dimension).
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer blocks.
            d_ff (int, optional): Feedforward hidden dimension. Defaults to 4 * d_model.
            dropout_prob (float): Dropout probability.
            max_seq_len (int): Maximum sequence length for positional encoding.
            
        Raises:
            ValueError: If d_model is not divisible by num_heads or other invalid parameters.
        """
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout_prob = dropout_prob
        self.max_seq_len = max_seq_len
        
        # Create transformer blocks
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, d_ff, dropout_prob) 
            for _ in range(num_layers)
        ]
        
        # Final layer normalization
        self.final_layer_norm = LayerNormalization(d_model)
        
        # Output projection for next token prediction
        xavier_std = np.sqrt(2.0 / (d_model + d_model))
        self.output_projection = np.random.normal(0, xavier_std, (d_model, d_model))
        
        # Cache for positional encodings
        self._pos_encoding_cache = None
        
    def _get_causal_mask(self, seq_len: int) -> np.ndarray:
        """
        Generate causal (lower triangular) mask for autoregressive generation.
        
        Args:
            seq_len (int): Sequence length.
            
        Returns:
            np.ndarray: Causal mask of shape (seq_len, seq_len).
        """
        mask = np.tril(np.ones((seq_len, seq_len)))
        return mask
        
    def forward(self, embeddings: np.ndarray, use_causal_mask: bool = True) -> np.ndarray:
        """
        Forward pass through the transformer.
        
        Args:
            embeddings (np.ndarray): Input embeddings of shape (seq_len, d_model).
            use_causal_mask (bool): Whether to use causal masking for autoregressive generation.
            
        Returns:
            np.ndarray: Output logits of shape (seq_len, d_model).
            
        Raises:
            TypeError: If embeddings is not a numpy array.
            ValueError: If embeddings has incorrect shape.
        """
        if not isinstance(embeddings, np.ndarray):
            raise TypeError(f"Expected numpy array for embeddings, got {type(embeddings)}")
        
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got {embeddings.ndim}D")
        
        seq_len, emb_dim = embeddings.shape
        if emb_dim != self.d_model:
            raise ValueError(f"Embedding dimension mismatch: expected {self.d_model}, got {emb_dim}")
        
        # Add positional encoding 
        x = add_positional_encoding(embeddings)
        
        # Generate causal mask if needed
        mask = self._get_causal_mask(seq_len) if use_causal_mask else None
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block.forward(x, mask)
        
        # Final layer normalization
        x = self.final_layer_norm.forward(x)
        
        # Apply output projection
        output = np.dot(x, self.output_projection)
        
        return output

    def predict_next_word(self, sentence: str, temperature: float = 1.0, top_k: int = 5) -> str:
        """
        Predict the next word given a sentence using improved sampling.
        
        Args:
            sentence (str): Input sentence.
            temperature (float): Temperature for controlling randomness (higher = more random).
            top_k (int): Number of top tokens to consider for sampling.
            
        Returns:
            str: Predicted next word.
            
        Raises:
            ValueError: If sentence is empty or temperature is not positive.
        """
        if not sentence.strip():
            raise ValueError("Input sentence cannot be empty")
        
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        
        try:
            # Tokenize and embed input sentence
            embeddings = tokenize_and_embed(sentence, embedding_model)
            
            # Forward pass
            output = self.forward(embeddings, use_causal_mask=True)
            
            # Get logits for the last token
            last_token_logits = output[-1] / temperature
            
            # Apply softmax to get probabilities
            probs = softmax(last_token_logits.reshape(1, -1)).flatten()
            
            # Get top-k indices and their probabilities
            top_k_indices = np.argsort(probs)[-top_k:]
            top_k_probs = probs[top_k_indices]
            
            # Normalize top-k probabilities
            top_k_probs = top_k_probs / np.sum(top_k_probs)
            
            # Sample from top-k distribution
            chosen_index = np.random.choice(top_k_indices, p=top_k_probs)
            
            # Convert index back to word
            if chosen_index < len(embedding_model.index_to_key):
                next_word = embedding_model.index_to_key[chosen_index]
            else:
                next_word = "<UNK>"  # Unknown token
                
            return next_word
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "<UNK>"
    
    def complete_sentence(self, sentence: str, max_length: int = 20, 
                         temperature: float = 1.0, top_k: int = 5) -> str:
        """
        Complete a sentence by generating subsequent words.
        
        Args:
            sentence (str): Initial sentence to complete.
            max_length (int): Maximum number of words in completed sentence.
            temperature (float): Temperature for controlling randomness.
            top_k (int): Number of top tokens to consider for sampling.
            
        Returns:
            str: Completed sentence.
            
        Raises:
            ValueError: If parameters are invalid.
        """
        if not sentence.strip():
            raise ValueError("Input sentence cannot be empty")
        
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        
        words = sentence.strip().split()
        
        for _ in range(max_length - len(words)):
            try:
                current_sentence = " ".join(words)
                next_word = self.predict_next_word(current_sentence, temperature, top_k)
                
                # Stop if we get an end-of-sequence token
                if next_word in ["<EOS>", "<END>", ".", "!", "?"]:
                    if next_word not in ["<EOS>", "<END>"]:
                        words.append(next_word)
                    break
                    
                words.append(next_word)
                
            except Exception as e:
                print(f"Error during generation: {e}")
                break
        
        return " ".join(words)
    
    def set_training_mode(self, training: bool = True):
        """
        Set the model to training or evaluation mode.
        
        Args:
            training (bool): If True, set to training mode; otherwise, evaluation mode.
        """
        for block in self.transformer_blocks:
            if training:
                block.dropout.train()
            else:
                block.dropout.eval()