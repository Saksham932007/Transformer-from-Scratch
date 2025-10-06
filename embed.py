
import numpy as np
import gensim.downloader as api

# Embedding matrix
embedding_model = api.load('glove-wiki-gigaword-300')

def get_embedding(word: str, embedding_model) -> np.ndarray:
    """
    Retrieve the embedding vector for a given word.

    Args:
        word (str): The word to be embedded.
        embedding_model: Pre-trained embedding model.

    Returns:
        np.ndarray: Embedding vector of the word.
    """
    if word in embedding_model:
        return embedding_model[word]
    else: # if word is not in vocab
        return np.zeros(embedding_model.vector_size)

def tokenize_and_embed(word: str, embedding_model) -> list:
    """
    Tokenize the input sentence and obtain embeddings for each token.

    Args:
        word (str): Input sentence.
        embedding_model: Pre-trained embedding model.

    Returns:
        list: List of embedding vectors for each token.
    """
    tokens = word.split()  # split input sentence into words (tokens)
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
    """
    sequence_len, embedding_dim = embeddings.shape

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

