
import numpy as np
from typing import Union, Optional, Any
import gensim.downloader as api

# Embedding matrix - consider caching for production use
embedding_model = api.load('glove-wiki-gigaword-300')

def get_embedding(word: str, embedding_model: Any) -> np.ndarray:
    """
    Retrieve the embedding vector for a given word.

    Args:
        word (str): The word to be embedded.
        embedding_model: Pre-trained embedding model (e.g., GloVe, Word2Vec).

    Returns:
        np.ndarray: Embedding vector of the word. Returns zero vector if word not found.
        
    Raises:
        TypeError: If word is not a string.
        ValueError: If embedding_model is None or invalid.
    """
    if not isinstance(word, str):
        raise TypeError(f"Expected string, got {type(word)}")
    
    if embedding_model is None:
        raise ValueError("Embedding model cannot be None")
        
    if word in embedding_model:
        return embedding_model[word]
    else: # if word is not in vocab
        return np.zeros(embedding_model.vector_size)

def tokenize_and_embed(sentence: str, embedding_model: Any) -> np.ndarray:
    """
    Tokenize the input sentence and obtain embeddings for each token.

    Args:
        sentence (str): Input sentence to tokenize and embed.
        embedding_model: Pre-trained embedding model.

    Returns:
        np.ndarray: Array of embedding vectors for each token, shape (num_tokens, embedding_dim).
        
    Raises:
        TypeError: If sentence is not a string.
        ValueError: If sentence is empty or embedding_model is None.
    """
    if not isinstance(sentence, str):
        raise TypeError(f"Expected string, got {type(sentence)}")
    
    if not sentence.strip():
        raise ValueError("Input sentence cannot be empty")
        
    if embedding_model is None:
        raise ValueError("Embedding model cannot be None")
    
    tokens = sentence.strip().split()  # split input sentence into words (tokens)
    embeddings = np.array([get_embedding(word, embedding_model) for word in tokens])
    return embeddings

def add_positional_encoding(embeddings: np.ndarray) -> np.ndarray:
    """
    Add positional encoding to the input embeddings using vectorized operations.
    
    This optimized version replaces nested loops with vectorized NumPy operations
    for significantly better performance, especially for longer sequences.

    Args:
        embeddings (np.ndarray): Input embeddings of shape (sequence_len, embedding_dim).

    Returns:
        np.ndarray: Embeddings with added positional encodings.
        
    Raises:
        TypeError: If embeddings is not a numpy array.
        ValueError: If embeddings has incorrect shape or contains invalid values.
    """
    if not isinstance(embeddings, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(embeddings)}")
    
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got {embeddings.ndim}D array")
    
    if embeddings.size == 0:
        raise ValueError("Embeddings array cannot be empty")
        
    sequence_len, embedding_dim = embeddings.shape
    
    if embedding_dim == 0:
        raise ValueError("Embedding dimension cannot be zero")

    # Create position and dimension indices using broadcasting
    positions = np.arange(sequence_len)[:, np.newaxis]  # Shape: (sequence_len, 1)
    dimensions = np.arange(embedding_dim)[np.newaxis, :]  # Shape: (1, embedding_dim)
    
    # Calculate angle rates for all positions and dimensions at once
    angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / embedding_dim)
    angle_rads = positions * angle_rates
    
    # Apply sin to even indices and cos to odd indices
    pos_enc_matrix = np.zeros((sequence_len, embedding_dim))
    pos_enc_matrix[:, 0::2] = np.sin(angle_rads[:, 0::2])  # Even indices
    pos_enc_matrix[:, 1::2] = np.cos(angle_rads[:, 1::2])  # Odd indices

    # Add positional encodings
    return embeddings + pos_enc_matrix

