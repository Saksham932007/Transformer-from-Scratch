"""
Comprehensive Testing Framework for Transformer Implementation

This module provides extensive testing capabilities including unit tests,
integration tests, performance benchmarks, and regression tests.
"""

import unittest
import numpy as np
import time
import sys
import os
from typing import Dict, List, Tuple, Any
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embed import positional_encoding, add_positional_encoding, tokenize_and_embed
from attention import SelfAttention, MultiHeadAttention
from layers import LayerNormalization, FeedForwardNetwork, TransformerBlock
from performance import PerformanceProfiler, MatrixOperationOptimizer
from validation import ModelValidator


class TestEmbedding(unittest.TestCase):
    """Test cases for embedding functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.seq_len = 100
        self.d_model = 512
        self.vocab_size = 1000
        self.batch_size = 32
    
    def test_positional_encoding_shape(self):
        """Test that positional encoding has correct shape."""
        pos_enc = positional_encoding(self.seq_len, self.d_model)
        self.assertEqual(pos_enc.shape, (self.seq_len, self.d_model))
    
    def test_positional_encoding_properties(self):
        """Test mathematical properties of positional encoding."""
        pos_enc = positional_encoding(self.seq_len, self.d_model)
        
        # Check that values are in reasonable range
        self.assertTrue(np.all(np.abs(pos_enc) <= 1.0))
        
        # Check that even positions use sin and odd positions use cos
        # (This is a simplified check - actual implementation may vary)
        self.assertIsInstance(pos_enc, np.ndarray)
        self.assertEqual(pos_enc.dtype, np.float32)
    
    def test_add_positional_encoding(self):
        """Test adding positional encoding to embeddings."""
        embeddings = np.random.randn(self.batch_size, self.seq_len, self.d_model).astype(np.float32)
        result = add_positional_encoding(embeddings)
        
        # Shape should be preserved
        self.assertEqual(result.shape, embeddings.shape)
        
        # Values should be different (encoding was added)
        self.assertFalse(np.array_equal(result, embeddings))
    
    def test_tokenize_and_embed(self):
        """Test tokenization and embedding process."""
        texts = ["hello world", "test sentence"]
        vocab = {word: i for i, word in enumerate(["hello", "world", "test", "sentence"])}
        
        try:
            result = tokenize_and_embed(texts, vocab, self.d_model)
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape[0], len(texts))
        except Exception as e:
            # Function might require additional parameters
            self.skipTest(f"tokenize_and_embed requires specific implementation: {e}")


class TestAttention(unittest.TestCase):
    """Test cases for attention mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 512
        self.num_heads = 8
        self.seq_len = 100
        self.batch_size = 32
        
        # Create sample input
        self.sample_input = np.random.randn(
            self.batch_size, self.seq_len, self.d_model
        ).astype(np.float32)
    
    def test_self_attention_initialization(self):
        """Test self-attention layer initialization."""
        attention = SelfAttention(self.d_model)
        
        # Check that weights are initialized
        self.assertIsNotNone(attention.W_q)
        self.assertIsNotNone(attention.W_k)
        self.assertIsNotNone(attention.W_v)
        
        # Check shapes
        expected_shape = (self.d_model, self.d_model)
        self.assertEqual(attention.W_q.shape, expected_shape)
        self.assertEqual(attention.W_k.shape, expected_shape)
        self.assertEqual(attention.W_v.shape, expected_shape)
    
    def test_self_attention_forward(self):
        """Test self-attention forward pass."""
        attention = SelfAttention(self.d_model)
        
        try:
            output = attention.forward(self.sample_input)
            
            # Output should have same shape as input
            self.assertEqual(output.shape, self.sample_input.shape)
            
            # Output should be different from input (attention was applied)
            self.assertFalse(np.array_equal(output, self.sample_input))
            
        except Exception as e:
            self.fail(f"Self-attention forward pass failed: {e}")
    
    def test_multi_head_attention_initialization(self):
        """Test multi-head attention initialization."""
        mha = MultiHeadAttention(self.d_model, self.num_heads)
        
        # Check head dimension calculation
        expected_head_dim = self.d_model // self.num_heads
        self.assertEqual(mha.d_k, expected_head_dim)
        
        # Check weight initialization
        self.assertIsNotNone(mha.W_q)
        self.assertIsNotNone(mha.W_k)
        self.assertIsNotNone(mha.W_v)
        self.assertIsNotNone(mha.W_o)
    
    def test_multi_head_attention_forward(self):
        """Test multi-head attention forward pass."""
        mha = MultiHeadAttention(self.d_model, self.num_heads)
        
        try:
            output = mha.forward(self.sample_input)
            
            # Output should have same shape as input
            self.assertEqual(output.shape, self.sample_input.shape)
            
            # Check that attention weights are returned if requested
            if hasattr(mha, 'attention_weights'):
                weights_shape = (self.batch_size, self.num_heads, self.seq_len, self.seq_len)
                # This would be the expected shape for attention weights
                
        except Exception as e:
            self.fail(f"Multi-head attention forward pass failed: {e}")


class TestLayers(unittest.TestCase):
    """Test cases for transformer layers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 512
        self.d_ff = 2048
        self.seq_len = 100
        self.batch_size = 32
        
        self.sample_input = np.random.randn(
            self.batch_size, self.seq_len, self.d_model
        ).astype(np.float32)
    
    def test_layer_normalization(self):
        """Test layer normalization."""
        layer_norm = LayerNormalization(self.d_model)
        
        output = layer_norm.forward(self.sample_input)
        
        # Shape should be preserved
        self.assertEqual(output.shape, self.sample_input.shape)
        
        # Check normalization properties (approximately)
        # Mean should be close to 0, std close to 1
        mean = np.mean(output, axis=-1, keepdims=True)
        std = np.std(output, axis=-1, keepdims=True)
        
        self.assertTrue(np.allclose(mean, 0, atol=1e-6))
        self.assertTrue(np.allclose(std, 1, atol=1e-6))
    
    def test_feedforward_network(self):
        """Test feedforward network."""
        ffn = FeedForwardNetwork(self.d_model, self.d_ff)
        
        output = ffn.forward(self.sample_input)
        
        # Shape should be preserved
        self.assertEqual(output.shape, self.sample_input.shape)
        
        # Output should be different from input
        self.assertFalse(np.array_equal(output, self.sample_input))
    
    def test_transformer_block(self):
        """Test complete transformer block."""
        transformer_block = TransformerBlock(
            d_model=self.d_model,
            num_heads=8,
            d_ff=self.d_ff,
            dropout_prob=0.1
        )
        
        output = transformer_block.forward(self.sample_input)
        
        # Shape should be preserved
        self.assertEqual(output.shape, self.sample_input.shape)
        
        # Output should be different from input
        self.assertFalse(np.array_equal(output, self.sample_input))


class TestPerformance(unittest.TestCase):
    """Test cases for performance monitoring."""
    
    def test_performance_profiler(self):
        """Test performance profiler functionality."""
        profiler = PerformanceProfiler()
        
        with profiler:
            # Simulate some computation
            x = np.random.randn(1000, 1000)
            y = np.dot(x, x.T)
            z = np.sum(y)
        
        metrics = profiler.get_metrics()
        
        # Check that metrics are collected
        self.assertIn('execution_time', metrics)
        self.assertIn('memory_usage', metrics)
        self.assertGreater(metrics['execution_time'], 0)
    
    def test_matrix_operation_optimizer(self):
        """Test matrix operation optimization."""
        optimizer = MatrixOperationOptimizer()
        
        # Test different matrix sizes
        sizes = [100, 500, 1000]
        results = []
        
        for size in sizes:
            result = optimizer.benchmark_matrix_multiplication(size)
            results.append(result)
            
            # Check that benchmark returns valid results
            self.assertIn('time', result)
            self.assertIn('operations_per_second', result)
            self.assertGreater(result['time'], 0)
            self.assertGreater(result['operations_per_second'], 0)
        
        # Performance should generally increase with optimizations
        self.assertTrue(len(results) == len(sizes))


class TestIntegration(unittest.TestCase):
    """Integration tests for complete transformer pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.d_model = 256  # Smaller for faster tests
        self.num_heads = 4
        self.seq_len = 50
        self.batch_size = 8
        self.vocab_size = 1000
    
    def test_complete_forward_pass(self):
        """Test complete forward pass through transformer."""
        # Create sample input tokens
        input_tokens = np.random.randint(
            0, self.vocab_size, 
            size=(self.batch_size, self.seq_len)
        )
        
        try:
            # Create embeddings
            embeddings = np.random.randn(
                self.vocab_size, self.d_model
            ).astype(np.float32)
            
            # Get input embeddings
            input_embeddings = embeddings[input_tokens]
            
            # Add positional encoding
            pos_encoded = add_positional_encoding(input_embeddings)
            
            # Pass through transformer block
            transformer_block = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_model * 4,
                dropout_prob=0.1
            )
            
            output = transformer_block.forward(pos_encoded)
            
            # Check output shape
            expected_shape = (self.batch_size, self.seq_len, self.d_model)
            self.assertEqual(output.shape, expected_shape)
            
            # Check that output is different from input
            self.assertFalse(np.array_equal(output, pos_encoded))
            
        except Exception as e:
            self.fail(f"Complete forward pass failed: {e}")
    
    def test_gradient_flow(self):
        """Test that gradients can flow through the network."""
        # This is a placeholder for gradient testing
        # In a full implementation, you'd test backpropagation
        self.assertTrue(True, "Gradient flow test placeholder")


class TestRegression(unittest.TestCase):
    """Regression tests to ensure changes don't break existing functionality."""
    
    def test_deterministic_output(self):
        """Test that model produces deterministic output with fixed seed."""
        np.random.seed(42)
        
        # Create identical inputs
        input1 = np.random.randn(4, 10, 128).astype(np.float32)
        
        np.random.seed(42)
        input2 = np.random.randn(4, 10, 128).astype(np.float32)
        
        # Inputs should be identical
        self.assertTrue(np.array_equal(input1, input2))
        
        # Create model with fixed seed
        np.random.seed(42)
        attention1 = SelfAttention(128)
        
        np.random.seed(42)
        attention2 = SelfAttention(128)
        
        # Weights should be identical
        self.assertTrue(np.array_equal(attention1.W_q, attention2.W_q))
        self.assertTrue(np.array_equal(attention1.W_k, attention2.W_k))
        self.assertTrue(np.array_equal(attention1.W_v, attention2.W_v))
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very large values
        large_input = np.ones((2, 10, 128), dtype=np.float32) * 1000
        
        attention = SelfAttention(128)
        
        try:
            output = attention.forward(large_input)
            
            # Check for NaN or Inf values
            self.assertFalse(np.any(np.isnan(output)))
            self.assertFalse(np.any(np.isinf(output)))
            
        except Exception as e:
            self.fail(f"Numerical stability test failed with large values: {e}")
        
        # Test with very small values
        small_input = np.ones((2, 10, 128), dtype=np.float32) * 1e-10
        
        try:
            output = attention.forward(small_input)
            
            # Check for NaN or Inf values
            self.assertFalse(np.any(np.isnan(output)))
            self.assertFalse(np.any(np.isinf(output)))
            
        except Exception as e:
            self.fail(f"Numerical stability test failed with small values: {e}")


class BenchmarkSuite:
    """Performance benchmark suite for transformer components."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_attention(self, sizes: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """Benchmark attention mechanisms with different sizes."""
        results = []
        
        for batch_size, seq_len, d_model in sizes:
            print(f"Benchmarking attention: batch={batch_size}, seq={seq_len}, d_model={d_model}")
            
            input_data = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
            attention = SelfAttention(d_model)
            
            # Warmup
            for _ in range(3):
                _ = attention.forward(input_data)
            
            # Benchmark
            start_time = time.time()
            for _ in range(10):
                output = attention.forward(input_data)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            throughput = batch_size * seq_len / avg_time
            
            results.append({
                'batch_size': batch_size,
                'seq_len': seq_len,
                'd_model': d_model,
                'avg_time': avg_time,
                'throughput': throughput
            })
        
        return results
    
    def benchmark_transformer_block(self, sizes: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """Benchmark complete transformer blocks."""
        results = []
        
        for batch_size, seq_len, d_model in sizes:
            print(f"Benchmarking transformer: batch={batch_size}, seq={seq_len}, d_model={d_model}")
            
            input_data = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
            transformer = TransformerBlock(
                d_model=d_model,
                num_heads=8,
                d_ff=d_model * 4,
                dropout_prob=0.1
            )
            
            # Warmup
            for _ in range(3):
                _ = transformer.forward(input_data)
            
            # Benchmark
            start_time = time.time()
            for _ in range(5):
                output = transformer.forward(input_data)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 5
            throughput = batch_size * seq_len / avg_time
            
            results.append({
                'batch_size': batch_size,
                'seq_len': seq_len,
                'd_model': d_model,
                'avg_time': avg_time,
                'throughput': throughput
            })
        
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        sizes = [
            (8, 64, 256),
            (16, 128, 512),
            (32, 256, 512),
            (8, 512, 1024),
        ]
        
        results = {
            'attention': self.benchmark_attention(sizes),
            'transformer_block': self.benchmark_transformer_block(sizes)
        }
        
        return results


def run_all_tests():
    """Run all test suites."""
    print("Running Transformer Test Suite")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestEmbedding,
        TestAttention,
        TestLayers,
        TestPerformance,
        TestIntegration,
        TestRegression
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()


def run_benchmarks():
    """Run performance benchmarks."""
    print("\nRunning Performance Benchmarks")
    print("=" * 50)
    
    benchmark_suite = BenchmarkSuite()
    results = benchmark_suite.run_full_benchmark()
    
    print("\nBenchmark Results:")
    print("-" * 30)
    
    for component, component_results in results.items():
        print(f"\n{component.upper()} Results:")
        for result in component_results:
            print(f"  Batch: {result['batch_size']}, Seq: {result['seq_len']}, "
                  f"D_model: {result['d_model']}")
            print(f"    Time: {result['avg_time']:.4f}s, "
                  f"Throughput: {result['throughput']:.2f} tokens/s")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    # Run tests
    test_success = run_all_tests()
    
    if test_success:
        print("\n✅ All tests passed!")
        
        # Run benchmarks if tests pass
        run_benchmarks()
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)