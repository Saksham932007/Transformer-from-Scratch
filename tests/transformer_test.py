"""
Comprehensive test suite for transformer components.

This module provides extensive testing including unit tests, integration tests,
performance benchmarks, and validation tests.
"""

import sys
import os
import time
import numpy as np
import unittest
from typing import List, Tuple, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer import Transformer, TransformerBlock
from attention import MultiHeadAttention, SelfAttention, softmax
from embed import tokenize_and_embed, add_positional_encoding, embedding_model
from layers import LayerNormalization, FeedForward, Dropout, gelu
from config import TransformerConfig, create_default_config, create_small_config
from data_processing import Tokenizer, TextDataset, DataLoader, create_sample_dataset
from training import AdamOptimizer, LearningRateScheduler, LossFunction


class TestTransformerComponents(unittest.TestCase):
    """Test individual transformer components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_dim = 128
        self.num_heads = 4
        self.seq_len = 16
        self.batch_size = 2
        
        # Create test data
        np.random.seed(42)
        self.test_embeddings = np.random.randn(self.seq_len, self.embedding_dim)
        self.test_logits = np.random.randn(self.batch_size, 100)
        self.test_targets = np.random.randint(0, 100, self.batch_size)
    
    def test_softmax_numerical_stability(self):
        """Test softmax function with extreme values."""
        # Test with large positive values
        large_scores = np.array([[1000, 1001, 999]])
        result = softmax(large_scores)
        self.assertTrue(np.allclose(np.sum(result, axis=-1), 1.0))
        self.assertFalse(np.any(np.isnan(result)))
        
        # Test with large negative values
        large_neg_scores = np.array([[-1000, -1001, -999]])
        result = softmax(large_neg_scores)
        self.assertTrue(np.allclose(np.sum(result, axis=-1), 1.0))
        self.assertFalse(np.any(np.isnan(result)))
    
    def test_self_attention_shapes(self):
        """Test self-attention output shapes."""
        attention = SelfAttention(self.embedding_dim)
        output = attention.forward(self.test_embeddings)
        
        self.assertEqual(output.shape, self.test_embeddings.shape)
        self.assertFalse(np.any(np.isnan(output)))
    
    def test_multi_head_attention_shapes(self):
        """Test multi-head attention output shapes."""
        attention = MultiHeadAttention(self.embedding_dim, self.num_heads)
        output = attention.forward(self.test_embeddings)
        
        self.assertEqual(output.shape, self.test_embeddings.shape)
        self.assertFalse(np.any(np.isnan(output)))
    
    def test_layer_normalization(self):
        """Test layer normalization implementation."""
        layer_norm = LayerNormalization(self.embedding_dim)
        output = layer_norm.forward(self.test_embeddings)
        
        # Check shape preservation
        self.assertEqual(output.shape, self.test_embeddings.shape)
        
        # Check normalization (mean ≈ 0, std ≈ 1)
        mean = np.mean(output, axis=-1)
        std = np.std(output, axis=-1)
        
        self.assertTrue(np.allclose(mean, 0, atol=1e-6))
        self.assertTrue(np.allclose(std, 1, atol=1e-6))
    
    def test_feedforward_network(self):
        """Test feedforward network."""
        d_ff = 512
        ff = FeedForward(self.embedding_dim, d_ff)
        output = ff.forward(self.test_embeddings)
        
        self.assertEqual(output.shape, self.test_embeddings.shape)
        self.assertFalse(np.any(np.isnan(output)))
    
    def test_dropout_training_vs_eval(self):
        """Test dropout behavior in training vs evaluation mode."""
        dropout = Dropout(p=0.5)
        
        # Training mode
        dropout.train()
        output_train = dropout.forward(self.test_embeddings)
        
        # Evaluation mode
        dropout.eval()
        output_eval = dropout.forward(self.test_embeddings)
        
        # In eval mode, output should be identical to input
        self.assertTrue(np.allclose(output_eval, self.test_embeddings))
        
        # In training mode, some values should be zeroed
        dropout.train()
        output_train2 = dropout.forward(self.test_embeddings)
        self.assertFalse(np.allclose(output_train2, self.test_embeddings))
    
    def test_gelu_activation(self):
        """Test GELU activation function."""
        x = np.array([-2, -1, 0, 1, 2])
        output = gelu(x)
        
        # GELU should be smooth and approximately 0 at x=0
        self.assertAlmostEqual(output[2], 0, places=6)
        
        # GELU should be monotonically increasing
        self.assertTrue(np.all(np.diff(output) > 0))
    
    def test_positional_encoding_properties(self):
        """Test positional encoding properties."""
        # Test different sequence lengths
        for seq_len in [10, 50, 100]:
            embeddings = np.random.randn(seq_len, self.embedding_dim)
            pos_encoded = add_positional_encoding(embeddings)
            
            # Shape should be preserved
            self.assertEqual(pos_encoded.shape, embeddings.shape)
            
            # Should not contain NaN or Inf
            self.assertFalse(np.any(np.isnan(pos_encoded)))
            self.assertFalse(np.any(np.isinf(pos_encoded)))


class TestTransformerModel(unittest.TestCase):
    """Test complete transformer model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = create_small_config()  # Use small config for faster tests
        self.transformer = Transformer(
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            d_ff=self.config.d_ff,
            dropout_prob=self.config.dropout_prob
        )
        
        # Create test sentence
        self.test_sentence = "this is a test sentence for the transformer"
    
    def test_forward_pass(self):
        """Test complete forward pass."""
        try:
            embeddings = tokenize_and_embed(self.test_sentence, embedding_model)
            
            # Ensure embeddings match model dimension
            if embeddings.shape[1] != self.config.d_model:
                # Pad or truncate to match d_model
                if embeddings.shape[1] > self.config.d_model:
                    embeddings = embeddings[:, :self.config.d_model]
                else:
                    padding = np.zeros((embeddings.shape[0], 
                                      self.config.d_model - embeddings.shape[1]))
                    embeddings = np.concatenate([embeddings, padding], axis=1)
            
            output = self.transformer.forward(embeddings)
            
            # Check output shape
            expected_shape = (embeddings.shape[0], self.config.d_model)
            self.assertEqual(output.shape, expected_shape)
            
            # Check for numerical issues
            self.assertFalse(np.any(np.isnan(output)))
            self.assertFalse(np.any(np.isinf(output)))
            
        except Exception as e:
            self.fail(f"Forward pass failed: {e}")
    
    def test_text_generation(self):
        """Test text generation functionality."""
        try:
            # Test next word prediction
            next_word = self.transformer.predict_next_word(
                self.test_sentence, 
                temperature=1.0, 
                top_k=5
            )
            
            self.assertIsInstance(next_word, str)
            self.assertNotEqual(next_word, "")
            
            # Test sentence completion
            completed = self.transformer.complete_sentence(
                "the quick brown",
                max_length=10,
                temperature=1.0
            )
            
            self.assertIsInstance(completed, str)
            self.assertTrue(len(completed.split()) >= 3)
            
        except Exception as e:
            self.fail(f"Text generation failed: {e}")
    
    def test_causal_masking(self):
        """Test causal masking in attention."""
        seq_len = 5
        mask = self.transformer._get_causal_mask(seq_len)
        
        # Check shape
        self.assertEqual(mask.shape, (seq_len, seq_len))
        
        # Check lower triangular structure
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    self.assertEqual(mask[i, j], 0)
                else:
                    self.assertEqual(mask[i, j], 1)


class TestOptimizationComponents(unittest.TestCase):
    """Test optimization and training components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 64
        self.vocab_size = 100
        
        # Create dummy parameters and gradients
        self.parameters = {
            'weight1': np.random.randn(self.d_model, self.d_model),
            'weight2': np.random.randn(self.d_model, self.vocab_size)
        }
        
        self.gradients = {
            'weight1': np.random.randn(self.d_model, self.d_model) * 0.01,
            'weight2': np.random.randn(self.d_model, self.vocab_size) * 0.01
        }
    
    def test_adam_optimizer(self):
        """Test Adam optimizer implementation."""
        optimizer = AdamOptimizer(learning_rate=0.001)
        
        # Store original parameters
        orig_params = {k: v.copy() for k, v in self.parameters.items()}
        
        # Perform optimization step
        optimizer.step(self.parameters, self.gradients)
        
        # Parameters should have changed
        for key in self.parameters:
            self.assertFalse(np.allclose(self.parameters[key], orig_params[key]))
    
    def test_learning_rate_scheduler(self):
        """Test learning rate scheduler."""
        optimizer = AdamOptimizer(learning_rate=0.001)
        scheduler = LearningRateScheduler(
            optimizer, 
            warmup_steps=100, 
            d_model=self.d_model
        )
        
        initial_lr = scheduler.get_lr()
        
        # Take several steps
        for _ in range(50):
            scheduler.step()
        
        warmup_lr = scheduler.get_lr()
        
        # Learning rate should increase during warmup
        self.assertGreater(warmup_lr, initial_lr)
    
    def test_loss_functions(self):
        """Test loss function implementations."""
        batch_size = 4
        vocab_size = 100
        
        logits = np.random.randn(batch_size, vocab_size)
        targets = np.random.randint(0, vocab_size, batch_size)
        
        # Test cross-entropy loss
        loss, gradients = LossFunction.cross_entropy_loss(logits, targets)
        
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
        self.assertEqual(gradients.shape, logits.shape)
        
        # Test with label smoothing
        loss_smooth, grad_smooth = LossFunction.cross_entropy_loss(
            logits, targets, label_smoothing=0.1
        )
        
        self.assertIsInstance(loss_smooth, float)
        self.assertGreater(loss_smooth, 0)


class TestDataProcessing(unittest.TestCase):
    """Test data processing components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_texts = create_sample_dataset(50)
        self.tokenizer = Tokenizer(vocab_size=1000)
    
    def test_tokenizer_build_vocab(self):
        """Test tokenizer vocabulary building."""
        self.tokenizer.build_vocab(self.sample_texts)
        
        # Check that vocabulary was built
        self.assertGreater(len(self.tokenizer.word_to_id), len(self.tokenizer.special_tokens))
        
        # Check special tokens are present
        for token in self.tokenizer.special_tokens:
            self.assertIn(token, self.tokenizer.word_to_id)
    
    def test_tokenizer_encode_decode(self):
        """Test tokenizer encoding and decoding."""
        self.tokenizer.build_vocab(self.sample_texts)
        
        test_text = "this is a test sentence"
        token_ids = self.tokenizer.encode(test_text)
        decoded_text = self.tokenizer.decode(token_ids)
        
        self.assertIsInstance(token_ids, list)
        self.assertTrue(all(isinstance(x, int) for x in token_ids))
        self.assertIsInstance(decoded_text, str)
    
    def test_text_dataset(self):
        """Test text dataset functionality."""
        self.tokenizer.build_vocab(self.sample_texts)
        dataset = TextDataset(self.sample_texts[:20], self.tokenizer, max_length=64)
        
        # Test dataset length
        self.assertGreater(len(dataset), 0)
        
        # Test data retrieval
        input_ids, target_ids = dataset[0]
        self.assertIsInstance(input_ids, list)
        self.assertIsInstance(target_ids, list)
        self.assertEqual(len(input_ids), len(target_ids))
    
    def test_data_loader(self):
        """Test data loader with batching."""
        self.tokenizer.build_vocab(self.sample_texts)
        dataset = TextDataset(self.sample_texts[:20], self.tokenizer, max_length=32)
        data_loader = DataLoader(dataset, batch_size=4)
        
        # Test batch generation
        for batch_inputs, batch_targets, attention_mask in data_loader:
            self.assertEqual(len(batch_inputs.shape), 2)
            self.assertEqual(len(batch_targets.shape), 2)
            self.assertEqual(len(attention_mask.shape), 2)
            self.assertEqual(batch_inputs.shape, batch_targets.shape)
            self.assertEqual(batch_inputs.shape, attention_mask.shape)
            break  # Just test first batch


class PerformanceBenchmark(unittest.TestCase):
    """Performance benchmark tests."""
    
    def setUp(self):
        """Set up benchmark fixtures."""
        self.config = create_small_config()
        self.transformer = Transformer(
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            num_layers=2,  # Small for faster benchmarks
            dropout_prob=0.0  # Disable for consistent timing
        )
    
    def benchmark_forward_pass(self):
        """Benchmark forward pass performance."""
        # Create test data
        seq_lengths = [16, 32, 64, 128]
        results = {}
        
        for seq_len in seq_lengths:
            embeddings = np.random.randn(seq_len, self.config.d_model)
            
            # Warm up
            for _ in range(3):
                _ = self.transformer.forward(embeddings)
            
            # Benchmark
            start_time = time.time()
            num_runs = 10
            
            for _ in range(num_runs):
                _ = self.transformer.forward(embeddings)
            
            avg_time = (time.time() - start_time) / num_runs
            results[seq_len] = avg_time
            
            print(f"Seq length {seq_len}: {avg_time:.4f}s per forward pass")
        
        # Simple performance check (these are very loose bounds)
        for seq_len, time_taken in results.items():
            self.assertLess(time_taken, 1.0)  # Should complete in under 1 second
    
    def benchmark_attention_scaling(self):
        """Benchmark attention mechanism scaling."""
        attention_dims = [64, 128, 256]
        results = {}
        
        for dim in attention_dims:
            attention = MultiHeadAttention(dim, 4)
            embeddings = np.random.randn(32, dim)
            
            # Warm up
            for _ in range(3):
                _ = attention.forward(embeddings)
            
            # Benchmark
            start_time = time.time()
            num_runs = 20
            
            for _ in range(num_runs):
                _ = attention.forward(embeddings)
            
            avg_time = (time.time() - start_time) / num_runs
            results[dim] = avg_time
            
            print(f"Attention dim {dim}: {avg_time:.4f}s per forward pass")


def run_all_tests():
    """Run all test suites."""
    print("Running Transformer Test Suite")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestTransformerComponents))
    test_suite.addTest(unittest.makeSuite(TestTransformerModel))
    test_suite.addTest(unittest.makeSuite(TestOptimizationComponents))
    test_suite.addTest(unittest.makeSuite(TestDataProcessing))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run benchmarks separately
    print("\n" + "=" * 50)
    print("Running Performance Benchmarks")
    print("=" * 50)
    
    benchmark_suite = unittest.TestSuite()
    benchmark_suite.addTest(unittest.makeSuite(PerformanceBenchmark))
    
    benchmark_runner = unittest.TextTestRunner(verbosity=2)
    benchmark_result = benchmark_runner.run(benchmark_suite)
    
    return result.wasSuccessful() and benchmark_result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)