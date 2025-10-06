"""
Focused tests for attention mechanisms.

This module provides specific tests for attention components including
numerical stability, masking, and performance characteristics.
"""

import sys
import os
import numpy as np
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention import SelfAttention, MultiHeadAttention, softmax


def test_attention_basic_functionality():
    """Test basic attention functionality."""
    print("Testing Basic Attention Functionality")
    print("-" * 40)
    
    embedding_dim = 128
    num_heads = 8
    sequence_length = 16
    
    # Create test embeddings
    np.random.seed(42)
    dummy_embeddings = np.random.randn(sequence_length, embedding_dim)
    
    print(f"Input shape: {dummy_embeddings.shape}")
    
    # Test Self Attention
    print("\nTesting Self Attention:")
    self_attention = SelfAttention(embedding_dim)
    self_attention_output = self_attention.forward(dummy_embeddings)
    
    print(f"Self Attention Output Shape: {self_attention_output.shape}")
    print(f"Output range: [{self_attention_output.min():.3f}, {self_attention_output.max():.3f}]")
    
    # Test Multi-Head Attention
    print("\nTesting Multi-Head Attention:")
    multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)
    multi_head_output = multi_head_attention.forward(dummy_embeddings)
    
    print(f"Multi-Head Attention Output Shape: {multi_head_output.shape}")
    print(f"Output range: [{multi_head_output.min():.3f}, {multi_head_output.max():.3f}]")
    
    # Verify shapes are preserved
    assert self_attention_output.shape == dummy_embeddings.shape
    assert multi_head_output.shape == dummy_embeddings.shape
    
    print("✅ Shape preservation test passed!")


def test_attention_with_masking():
    """Test attention with various masking scenarios."""
    print("\nTesting Attention with Masking")
    print("-" * 40)
    
    embedding_dim = 64
    sequence_length = 8
    
    dummy_embeddings = np.random.randn(sequence_length, embedding_dim)
    
    # Create causal mask (lower triangular)
    causal_mask = np.tril(np.ones((sequence_length, sequence_length)))
    
    # Create padding mask (mask last 2 tokens)
    padding_mask = np.ones((sequence_length, sequence_length))
    padding_mask[:, -2:] = 0  # Mask last 2 positions
    
    self_attention = SelfAttention(embedding_dim)
    
    # Test without mask
    output_no_mask = self_attention.forward(dummy_embeddings)
    
    # Test with causal mask
    output_causal = self_attention.forward(dummy_embeddings, mask=causal_mask)
    
    # Test with padding mask
    output_padding = self_attention.forward(dummy_embeddings, mask=padding_mask)
    
    print(f"No mask output range: [{output_no_mask.min():.3f}, {output_no_mask.max():.3f}]")
    print(f"Causal mask output range: [{output_causal.min():.3f}, {output_causal.max():.3f}]")
    print(f"Padding mask output range: [{output_padding.min():.3f}, {output_padding.max():.3f}]")
    
    # Outputs should be different when masks are applied
    assert not np.allclose(output_no_mask, output_causal, atol=1e-6)
    assert not np.allclose(output_no_mask, output_padding, atol=1e-6)
    
    print("✅ Masking test passed!")


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    print("\nTesting Numerical Stability")
    print("-" * 40)
    
    # Test softmax with extreme values
    extreme_scores = np.array([
        [1000, 1001, 999],      # Large positive values
        [-1000, -1001, -999],   # Large negative values
        [0, 0, 0],              # Zero values
        [1e-8, 2e-8, 3e-8]      # Very small values
    ])
    
    for i, scores in enumerate(extreme_scores):
        result = softmax(scores.reshape(1, -1))
        prob_sum = np.sum(result)
        
        print(f"Test {i+1}: Sum of probabilities = {prob_sum:.10f}")
        
        # Check that probabilities sum to 1
        assert abs(prob_sum - 1.0) < 1e-6, f"Probabilities don't sum to 1: {prob_sum}"
        
        # Check no NaN or Inf values
        assert not np.any(np.isnan(result)), "NaN values detected"
        assert not np.any(np.isinf(result)), "Inf values detected"
        
        # Check all probabilities are non-negative
        assert np.all(result >= 0), "Negative probabilities detected"
    
    print("✅ Numerical stability test passed!")


def test_attention_weights_properties():
    """Test properties of attention weights."""
    print("\nTesting Attention Weight Properties")
    print("-" * 40)
    
    embedding_dim = 32
    sequence_length = 6
    
    dummy_embeddings = np.random.randn(sequence_length, embedding_dim)
    
    # Create attention layer
    attention = SelfAttention(embedding_dim)
    
    # Get intermediate values for analysis
    query = np.dot(dummy_embeddings, attention.W_q)
    key = np.dot(dummy_embeddings, attention.W_k)
    
    # Calculate attention scores
    attention_scores = attention.calculate_attention_score(query, key)
    attention_weights = softmax(attention_scores)
    
    print(f"Attention scores shape: {attention_scores.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Test properties of attention weights
    row_sums = np.sum(attention_weights, axis=-1)
    print(f"Row sums (should be ~1.0): {row_sums}")
    
    # Each row should sum to approximately 1
    for i, row_sum in enumerate(row_sums):
        assert abs(row_sum - 1.0) < 1e-6, f"Row {i} sum is {row_sum}, not 1.0"
    
    # All weights should be non-negative
    assert np.all(attention_weights >= 0), "Negative attention weights detected"
    
    print("✅ Attention weight properties test passed!")


def test_multi_head_dimension_consistency():
    """Test dimension consistency in multi-head attention."""
    print("\nTesting Multi-Head Dimension Consistency")
    print("-" * 40)
    
    embedding_dim = 128
    test_head_counts = [1, 2, 4, 8, 16]
    sequence_length = 10
    
    dummy_embeddings = np.random.randn(sequence_length, embedding_dim)
    
    for num_heads in test_head_counts:
        if embedding_dim % num_heads == 0:  # Valid configuration
            print(f"Testing {num_heads} heads with {embedding_dim}D embeddings")
            
            multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)
            output = multi_head_attention.forward(dummy_embeddings)
            
            # Output shape should match input shape
            assert output.shape == dummy_embeddings.shape
            
            # Check head dimension calculation
            expected_head_dim = embedding_dim // num_heads
            assert multi_head_attention.head_dim == expected_head_dim
            
            print(f"  Head dimension: {multi_head_attention.head_dim}")
            print(f"  Output shape: {output.shape}")
        else:
            print(f"Skipping invalid configuration: {num_heads} heads, {embedding_dim}D")
    
    print("✅ Multi-head dimension consistency test passed!")


def benchmark_attention_performance():
    """Benchmark attention performance across different configurations."""
    print("\nBenchmarking Attention Performance")
    print("-" * 40)
    
    configurations = [
        (64, 4, 16),    # Small
        (128, 8, 32),   # Medium
        (256, 8, 64),   # Large
        (512, 16, 128)  # Very large
    ]
    
    for embedding_dim, num_heads, seq_len in configurations:
        print(f"\nConfig: {embedding_dim}D, {num_heads} heads, seq_len={seq_len}")
        
        dummy_embeddings = np.random.randn(seq_len, embedding_dim)
        
        # Benchmark Self Attention
        self_attention = SelfAttention(embedding_dim)
        
        # Warmup
        for _ in range(3):
            _ = self_attention.forward(dummy_embeddings)
        
        # Benchmark
        start_time = time.time()
        num_runs = 10
        
        for _ in range(num_runs):
            _ = self_attention.forward(dummy_embeddings)
        
        self_attn_time = (time.time() - start_time) / num_runs
        
        # Benchmark Multi-Head Attention
        multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)
        
        # Warmup
        for _ in range(3):
            _ = multi_head_attention.forward(dummy_embeddings)
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(num_runs):
            _ = multi_head_attention.forward(dummy_embeddings)
        
        multi_head_time = (time.time() - start_time) / num_runs
        
        print(f"  Self Attention: {self_attn_time:.4f}s")
        print(f"  Multi-Head Attention: {multi_head_time:.4f}s")
        print(f"  Overhead ratio: {multi_head_time/self_attn_time:.2f}x")


def run_attention_tests():
    """Run all attention tests."""
    print("Attention Mechanism Test Suite")
    print("=" * 50)
    
    try:
        test_attention_basic_functionality()
        test_attention_with_masking()
        test_numerical_stability()
        test_attention_weights_properties()
        test_multi_head_dimension_consistency()
        
        print("\n" + "=" * 50)
        benchmark_attention_performance()
        
        print("\n" + "=" * 50)
        print("✅ All attention tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    run_attention_tests()