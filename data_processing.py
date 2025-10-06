"""
Advanced Data Processing Pipeline for Transformer Models

This module provides comprehensive data processing capabilities including
tokenization, dataset management, batching, and preprocessing utilities.
"""

import numpy as np
import re
import json
from typing import List, Dict, Tuple, Optional, Iterator, Union, Set
from pathlib import Path
from collections import Counter, defaultdict
import random
from embed import embedding_model


class Tokenizer:
    """
    Advanced tokenizer with support for subword tokenization, special tokens,
    and vocabulary management.
    """
    
    def __init__(self, vocab_size: int = 50000, min_freq: int = 2):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size.
            min_freq: Minimum frequency for tokens to be included.
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1, 
            '<BOS>': 2,  # Beginning of sequence
            '<EOS>': 3,  # End of sequence
            '<MASK>': 4  # Masked token for BERT-style training
        }
        
        # Vocabularies
        self.word_to_id = self.special_tokens.copy()
        self.id_to_word = {v: k for k, v in self.special_tokens.items()}
        self.word_counts = Counter()
        
        # Compiled regex patterns
        self.token_pattern = re.compile(r'\b\w+\b|[^\w\s]')
        self.sentence_pattern = re.compile(r'[.!?]+')
        
    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of text strings to build vocabulary from.
        """
        print(f"Building vocabulary from {len(texts)} texts...")
        
        # Count word frequencies
        for text in texts:
            tokens = self.tokenize_text(text)
            self.word_counts.update(tokens)
        
        print(f"Found {len(self.word_counts)} unique tokens")
        
        # Filter by minimum frequency and sort by frequency
        filtered_words = [(word, count) for word, count in self.word_counts.items() 
                         if count >= self.min_freq]
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        
        # Add to vocabulary (excluding special tokens)
        current_id = len(self.special_tokens)
        for word, count in filtered_words:
            if current_id >= self.vocab_size:
                break
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1
        
        print(f"Final vocabulary size: {len(self.word_to_id)}")
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words and punctuation.
        
        Args:
            text: Input text string.
            
        Returns:
            List of tokens.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected string, got {type(text)}")
        
        # Convert to lowercase and find all tokens
        text = text.lower().strip()
        tokens = self.token_pattern.findall(text)
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True, 
               max_length: Optional[int] = None) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string.
            add_special_tokens: Whether to add BOS/EOS tokens.
            max_length: Maximum sequence length (truncate if longer).
            
        Returns:
            List of token IDs.
        """
        tokens = self.tokenize_text(text)
        
        # Convert tokens to IDs
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.word_to_id['<BOS>'])
        
        for token in tokens:
            token_id = self.word_to_id.get(token, self.word_to_id['<UNK>'])
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.word_to_id['<EOS>'])
        
        # Truncate if necessary
        if max_length is not None and len(token_ids) > max_length:
            if add_special_tokens:
                # Keep BOS, truncate middle, keep EOS
                token_ids = ([token_ids[0]] + 
                           token_ids[1:max_length-1] + 
                           [token_ids[-1]])
            else:
                token_ids = token_ids[:max_length]
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs.
            skip_special_tokens: Whether to skip special tokens in output.
            
        Returns:
            Decoded text string.
        """
        tokens = []
        special_token_ids = set(self.special_tokens.values())
        
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_token_ids:
                continue
            
            token = self.id_to_word.get(token_id, '<UNK>')
            tokens.append(token)
        
        return ' '.join(tokens)
    
    def save(self, filepath: Union[str, Path]):
        """Save tokenizer to file."""
        filepath = Path(filepath)
        vocab_data = {
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'word_counts': dict(self.word_counts),
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq
        }
        
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'Tokenizer':
        """Load tokenizer from file."""
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        
        tokenizer = cls(vocab_data['vocab_size'], vocab_data['min_freq'])
        tokenizer.word_to_id = vocab_data['word_to_id']
        tokenizer.id_to_word = {int(k): v for k, v in vocab_data['id_to_word'].items()}
        tokenizer.word_counts = Counter(vocab_data['word_counts'])
        
        return tokenizer


class TextDataset:
    """
    Dataset class for handling text data with efficient batching and preprocessing.
    """
    
    def __init__(self, texts: List[str], tokenizer: Tokenizer, 
                 max_length: int = 512, for_language_modeling: bool = True):
        """
        Initialize text dataset.
        
        Args:
            texts: List of text strings.
            tokenizer: Tokenizer instance.
            max_length: Maximum sequence length.
            for_language_modeling: Whether to prepare data for language modeling.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.for_language_modeling = for_language_modeling
        
        # Preprocess and encode all texts
        self.encoded_texts = []
        self.process_texts()
    
    def process_texts(self):
        """Process and encode all texts."""
        print(f"Processing {len(self.texts)} texts...")
        
        for text in self.texts:
            # Encode text
            token_ids = self.tokenizer.encode(
                text, 
                add_special_tokens=True, 
                max_length=self.max_length
            )
            
            if len(token_ids) > 1:  # Skip very short sequences
                self.encoded_texts.append(token_ids)
        
        print(f"Processed {len(self.encoded_texts)} valid sequences")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.encoded_texts)
    
    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item.
            
        Returns:
            Tuple of (input_ids, target_ids) for language modeling.
        """
        token_ids = self.encoded_texts[idx]
        
        if self.for_language_modeling:
            # For language modeling: input is sequence[:-1], target is sequence[1:]
            input_ids = token_ids[:-1]
            target_ids = token_ids[1:]
        else:
            # For other tasks, return the full sequence
            input_ids = token_ids
            target_ids = token_ids
        
        return input_ids, target_ids
    
    def get_random_sample(self) -> Tuple[List[int], List[int]]:
        """Get a random sample from the dataset."""
        idx = random.randint(0, len(self) - 1)
        return self[idx]


class DataLoader:
    """
    Data loader with batching, padding, and shuffling capabilities.
    """
    
    def __init__(self, dataset: TextDataset, batch_size: int = 32, 
                 shuffle: bool = True, drop_last: bool = False):
        """
        Initialize data loader.
        
        Args:
            dataset: Text dataset.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle data.
            drop_last: Whether to drop the last incomplete batch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.pad_token_id = dataset.tokenizer.word_to_id['<PAD>']
    
    def collate_fn(self, batch: List[Tuple[List[int], List[int]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collate function to pad sequences in a batch.
        
        Args:
            batch: List of (input_ids, target_ids) tuples.
            
        Returns:
            Tuple of (padded_inputs, padded_targets, attention_mask).
        """
        input_ids_batch = [item[0] for item in batch]
        target_ids_batch = [item[1] for item in batch]
        
        # Find maximum length in batch
        max_len = max(max(len(inp) for inp in input_ids_batch),
                     max(len(tgt) for tgt in target_ids_batch))
        
        # Pad sequences
        padded_inputs = []
        padded_targets = []
        attention_masks = []
        
        for input_ids, target_ids in batch:
            # Pad input
            input_len = len(input_ids)
            padded_input = input_ids + [self.pad_token_id] * (max_len - input_len)
            padded_inputs.append(padded_input)
            
            # Pad target
            target_len = len(target_ids)
            padded_target = target_ids + [self.pad_token_id] * (max_len - target_len)
            padded_targets.append(padded_target)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * input_len + [0] * (max_len - input_len)
            attention_masks.append(attention_mask)
        
        return (np.array(padded_inputs), 
                np.array(padded_targets), 
                np.array(attention_masks))
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Iterate over batches."""
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            # Skip incomplete batch if requested
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            
            # Get batch data
            batch = [self.dataset[idx] for idx in batch_indices]
            
            yield self.collate_fn(batch)
    
    def __len__(self) -> int:
        """Return number of batches."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class DataProcessor:
    """
    High-level data processor for preparing transformer training data.
    """
    
    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        """
        Initialize data processor.
        
        Args:
            tokenizer: Pre-trained tokenizer (optional).
        """
        self.tokenizer = tokenizer
    
    def load_text_file(self, filepath: Union[str, Path]) -> List[str]:
        """
        Load text from file, splitting into sentences or paragraphs.
        
        Args:
            filepath: Path to text file.
            
        Returns:
            List of text strings.
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into sentences or paragraphs
        if '\n\n' in content:
            # Split by paragraphs
            texts = [para.strip() for para in content.split('\n\n') 
                    if para.strip()]
        else:
            # Split by sentences
            sentences = re.split(r'[.!?]+', content)
            texts = [sent.strip() for sent in sentences if sent.strip()]
        
        return texts
    
    def load_json_file(self, filepath: Union[str, Path], text_key: str = 'text') -> List[str]:
        """
        Load text from JSON file.
        
        Args:
            filepath: Path to JSON file.
            text_key: Key containing text in JSON objects.
            
        Returns:
            List of text strings.
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            texts = [item[text_key] for item in data if text_key in item]
        else:
            texts = [data[text_key]] if text_key in data else []
        
        return texts
    
    def prepare_training_data(self, texts: List[str], 
                            vocab_size: int = 50000, 
                            max_length: int = 512,
                            test_split: float = 0.1,
                            val_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader, Tokenizer]:
        """
        Prepare complete training pipeline from raw texts.
        
        Args:
            texts: List of text strings.
            vocab_size: Vocabulary size for tokenizer.
            max_length: Maximum sequence length.
            test_split: Fraction of data for testing.
            val_split: Fraction of data for validation.
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader, tokenizer).
        """
        print(f"Preparing training data from {len(texts)} texts...")
        
        # Build tokenizer if not provided
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(vocab_size=vocab_size)
            self.tokenizer.build_vocab(texts)
        
        # Split data
        random.shuffle(texts)
        n_test = int(len(texts) * test_split)
        n_val = int(len(texts) * val_split)
        
        test_texts = texts[:n_test]
        val_texts = texts[n_test:n_test + n_val]
        train_texts = texts[n_test + n_val:]
        
        print(f"Split: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")
        
        # Create datasets
        train_dataset = TextDataset(train_texts, self.tokenizer, max_length)
        val_dataset = TextDataset(val_texts, self.tokenizer, max_length)
        test_dataset = TextDataset(test_texts, self.tokenizer, max_length)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return train_loader, val_loader, test_loader, self.tokenizer
    
    def create_embedding_matrix(self, tokenizer: Tokenizer, 
                              embedding_dim: int = 300) -> np.ndarray:
        """
        Create embedding matrix from pre-trained embeddings.
        
        Args:
            tokenizer: Tokenizer with vocabulary.
            embedding_dim: Dimension of embeddings.
            
        Returns:
            Embedding matrix of shape (vocab_size, embedding_dim).
        """
        vocab_size = len(tokenizer.word_to_id)
        embedding_matrix = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
        
        # Fill with pre-trained embeddings where available
        for word, idx in tokenizer.word_to_id.items():
            if word in embedding_model:
                embedding_matrix[idx] = embedding_model[word]
        
        return embedding_matrix


def create_sample_dataset(num_samples: int = 1000) -> List[str]:
    """
    Create a sample dataset for testing and demonstration.
    
    Args:
        num_samples: Number of sample texts to generate.
        
    Returns:
        List of sample text strings.
    """
    sample_texts = [
        "The transformer model revolutionized natural language processing.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Deep learning has made significant advances in recent years.",
        "Neural networks can learn complex patterns from data.",
        "Language models generate text by predicting the next word.",
        "Machine learning algorithms can solve many different problems.",
        "Artificial intelligence is transforming various industries.",
        "Data preprocessing is crucial for model performance.",
        "Gradient descent optimizes neural network parameters.",
        "Backpropagation computes gradients for parameter updates."
    ]
    
    # Generate variations and combinations
    texts = []
    for _ in range(num_samples):
        # Select random sentences and combine them
        num_sentences = random.randint(1, 3)
        selected = random.sample(sample_texts, num_sentences)
        combined = ' '.join(selected)
        texts.append(combined)
    
    return texts


# Example usage functions
def example_basic_usage():
    """Example of basic tokenizer and dataset usage."""
    print("=== Basic Tokenizer Usage ===")
    
    # Create sample data
    texts = create_sample_dataset(100)
    
    # Initialize and build tokenizer
    tokenizer = Tokenizer(vocab_size=1000)
    tokenizer.build_vocab(texts)
    
    # Example encoding/decoding
    sample_text = "The transformer model is powerful."
    token_ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(token_ids)
    
    print(f"Original: {sample_text}")
    print(f"Encoded: {token_ids}")
    print(f"Decoded: {decoded_text}")
    
    # Create dataset and data loader
    dataset = TextDataset(texts[:50], tokenizer, max_length=64)
    data_loader = DataLoader(dataset, batch_size=4)
    
    # Example batch
    for batch_inputs, batch_targets, attention_mask in data_loader:
        print(f"Batch shape: {batch_inputs.shape}")
        print(f"First sequence: {batch_inputs[0]}")
        break


if __name__ == "__main__":
    example_basic_usage()