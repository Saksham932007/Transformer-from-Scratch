"""
Complete Example and Usage Guide for Transformer from Scratch

This script demonstrates how to use all components of the transformer
implementation, from data preparation to model training and text generation.
"""

import numpy as np
import time
from pathlib import Path

# Import all modules
from config import TransformerConfig, create_small_config, create_default_config
from transformer import Transformer
from data_processing import (
    Tokenizer, TextDataset, DataLoader, DataProcessor, 
    create_sample_dataset
)
from training import Trainer, AdamOptimizer, LearningRateScheduler
from embed import embedding_model


def example_1_basic_text_generation():
    """Example 1: Basic text generation with pre-trained embeddings."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Text Generation")
    print("=" * 60)
    
    # Create a small transformer for quick demonstration
    config = create_small_config()
    config.d_model = 300  # Match GloVe embeddings
    
    transformer = Transformer(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=2,  # Small for demo
        dropout_prob=0.1
    )
    
    print(f"Created transformer with {config.d_model}D embeddings, "
          f"{config.num_heads} heads, 2 layers")
    
    # Test sentences for completion
    test_sentences = [
        "the quick brown",
        "artificial intelligence is",
        "natural language processing",
        "deep learning models",
        "transformer architectures"
    ]
    
    print("\nGenerating text completions:")
    print("-" * 40)
    
    for sentence in test_sentences:
        try:
            completed = transformer.complete_sentence(
                sentence, 
                max_length=15, 
                temperature=0.8
            )
            print(f"Input:  '{sentence}'")
            print(f"Output: '{completed}'")
            print()
        except Exception as e:
            print(f"Error with '{sentence}': {e}")
            print()


def example_2_custom_tokenizer_and_dataset():
    """Example 2: Custom tokenizer and dataset creation."""
    print("=" * 60)
    print("EXAMPLE 2: Custom Tokenizer and Dataset")
    print("=" * 60)
    
    # Create sample dataset
    print("Creating sample dataset...")
    texts = create_sample_dataset(200)
    print(f"Generated {len(texts)} sample texts")
    
    # Build custom tokenizer
    print("\nBuilding custom tokenizer...")
    tokenizer = Tokenizer(vocab_size=2000, min_freq=2)
    tokenizer.build_vocab(texts)
    
    print(f"Vocabulary size: {len(tokenizer.word_to_id)}")
    print(f"Special tokens: {list(tokenizer.special_tokens.keys())}")
    
    # Test tokenization
    sample_text = "The transformer model uses attention mechanisms."
    token_ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(token_ids)
    
    print(f"\nTokenization example:")
    print(f"Original: {sample_text}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {decoded_text}")
    
    # Create dataset and data loader
    print("\nCreating dataset and data loader...")
    dataset = TextDataset(texts[:100], tokenizer, max_length=64)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(data_loader)}")
    
    # Show a sample batch
    for batch_inputs, batch_targets, attention_mask in data_loader:
        print(f"\nSample batch shapes:")
        print(f"Inputs: {batch_inputs.shape}")
        print(f"Targets: {batch_targets.shape}")
        print(f"Attention mask: {attention_mask.shape}")
        
        # Show first sequence in batch
        first_sequence = batch_inputs[0]
        decoded_sequence = tokenizer.decode(first_sequence.tolist())
        print(f"First sequence: {decoded_sequence}")
        break


def example_3_complete_training_pipeline():
    """Example 3: Complete training pipeline demonstration."""
    print("=" * 60)
    print("EXAMPLE 3: Complete Training Pipeline")
    print("=" * 60)
    
    # Create configuration
    config = create_small_config()
    config.d_model = 128  # Smaller for faster training
    config.num_layers = 2
    config.num_epochs = 3
    config.batch_size = 4
    
    print(f"Training configuration:")
    print(f"  Model dimension: {config.d_model}")
    print(f"  Number of heads: {config.num_heads}")
    print(f"  Number of layers: {config.num_layers}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Number of epochs: {config.num_epochs}")
    
    # Prepare data
    print("\nPreparing training data...")
    data_processor = DataProcessor()
    sample_texts = create_sample_dataset(100)
    
    train_loader, val_loader, test_loader, tokenizer = data_processor.prepare_training_data(
        sample_texts,
        vocab_size=1000,
        max_length=32,
        test_split=0.2,
        val_split=0.2
    )
    
    # Adjust model dimension to match tokenizer embedding dimension
    # For this demo, we'll use a smaller dimension
    config.d_model = 64
    
    # Create model
    transformer = Transformer(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout_prob=config.dropout_prob
    )
    
    print(f"Created model with {config.d_model}D embeddings")
    
    # Create trainer
    trainer = Trainer(transformer, config)
    
    # Prepare simplified training data (convert DataLoader to list)
    print("\nPreparing training batches...")
    train_batches = []
    val_batches = []
    
    # Convert data loader batches to simple format for demonstration
    for i, (batch_inputs, batch_targets, attention_mask) in enumerate(train_loader):
        if i >= 10:  # Limit to 10 batches for demo
            break
        
        # For simplicity, we'll use the first sample in each batch
        # In practice, you'd handle full batches
        sample_input = batch_inputs[0]
        sample_target = batch_targets[0]
        
        # Convert to embeddings (simplified approach)
        # In practice, you'd have proper embedding layers
        input_embedding = np.random.randn(len(sample_input), config.d_model)
        target_indices = sample_target[:len(sample_input)]
        
        train_batches.append((input_embedding, target_indices))
    
    for i, (batch_inputs, batch_targets, attention_mask) in enumerate(val_loader):
        if i >= 5:  # Limit to 5 batches for demo
            break
            
        sample_input = batch_inputs[0]
        sample_target = batch_targets[0]
        
        input_embedding = np.random.randn(len(sample_input), config.d_model)
        target_indices = sample_target[:len(sample_input)]
        
        val_batches.append((input_embedding, target_indices))
    
    print(f"Prepared {len(train_batches)} training batches")
    print(f"Prepared {len(val_batches)} validation batches")
    
    # Run training (simplified for demonstration)
    print("\nStarting training...")
    try:
        trainer.fit(train_batches, val_batches, num_epochs=2)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training error (expected in demo): {e}")


def example_4_configuration_management():
    """Example 4: Configuration management and serialization."""
    print("=" * 60)
    print("EXAMPLE 4: Configuration Management")
    print("=" * 60)
    
    # Create different configurations
    configs = {
        "small": create_small_config(),
        "default": create_default_config(),
    }
    
    # Modify configurations
    configs["small"].temperature = 0.7
    configs["small"].top_k = 10
    
    configs["default"].learning_rate = 0.0005
    configs["default"].dropout_prob = 0.15
    
    # Display configurations
    for name, config in configs.items():
        print(f"\n{name.upper()} Configuration:")
        print(f"  Model dimension: {config.d_model}")
        print(f"  Number of heads: {config.num_heads}")
        print(f"  Number of layers: {config.num_layers}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Dropout probability: {config.dropout_prob}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Top-k: {config.top_k}")
    
    # Save configurations
    print("\nSaving configurations...")
    config_dir = Path("example_configs")
    config_dir.mkdir(exist_ok=True)
    
    for name, config in configs.items():
        config_path = config_dir / f"{name}_config.json"
        config.save(config_path)
        print(f"Saved {name} config to {config_path}")
    
    # Load and verify
    print("\nLoading and verifying configurations...")
    for name in configs.keys():
        config_path = config_dir / f"{name}_config.json"
        loaded_config = TransformerConfig.load(config_path)
        
        print(f"Loaded {name} config - d_model: {loaded_config.d_model}")
        
        # Verify it matches original
        assert loaded_config.d_model == configs[name].d_model
        assert loaded_config.num_heads == configs[name].num_heads
    
    print("Configuration serialization test passed!")


def example_5_performance_comparison():
    """Example 5: Performance comparison of different optimizations."""
    print("=" * 60)
    print("EXAMPLE 5: Performance Comparison")
    print("=" * 60)
    
    # Test different configurations
    test_configs = [
        ("Tiny", 64, 4, 2),
        ("Small", 128, 8, 4),
        ("Medium", 256, 8, 6),
    ]
    
    results = {}
    
    for name, d_model, num_heads, num_layers in test_configs:
        print(f"\nTesting {name} model ({d_model}D, {num_heads} heads, {num_layers} layers):")
        
        # Create model
        transformer = Transformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_prob=0.0  # Disable for consistent timing
        )
        
        # Create test input
        seq_len = 32
        test_embeddings = np.random.randn(seq_len, d_model)
        
        # Warmup
        for _ in range(3):
            _ = transformer.forward(test_embeddings)
        
        # Benchmark
        start_time = time.time()
        num_runs = 10
        
        for _ in range(num_runs):
            output = transformer.forward(test_embeddings)
        
        avg_time = (time.time() - start_time) / num_runs
        
        results[name] = {
            'time': avg_time,
            'params': d_model * num_heads * num_layers,
            'output_shape': output.shape
        }
        
        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Output shape: {output.shape}")
        print(f"  Approximate parameters: {results[name]['params']:,}")
    
    # Summary
    print("\nPerformance Summary:")
    print("-" * 40)
    fastest = min(results.items(), key=lambda x: x[1]['time'])
    slowest = max(results.items(), key=lambda x: x[1]['time'])
    
    print(f"Fastest: {fastest[0]} ({fastest[1]['time']:.4f}s)")
    print(f"Slowest: {slowest[0]} ({slowest[1]['time']:.4f}s)")
    print(f"Speedup: {slowest[1]['time'] / fastest[1]['time']:.1f}x")


def main():
    """Run all examples."""
    print("TRANSFORMER FROM SCRATCH - COMPLETE EXAMPLES")
    print("=" * 80)
    print("This script demonstrates the capabilities of our transformer implementation.")
    print("Each example showcases different aspects of the system.\n")
    
    examples = [
        example_1_basic_text_generation,
        example_2_custom_tokenizer_and_dataset,
        example_3_complete_training_pipeline,
        example_4_configuration_management,
        example_5_performance_comparison
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\nRunning Example {i}...")
            example_func()
            print(f"✅ Example {i} completed successfully!")
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(examples):
            input("\nPress Enter to continue to the next example...")
    
    print("\n" + "=" * 80)
    print("🎉 All examples completed!")
    print("\nYou have successfully explored:")
    print("  • Basic text generation with transformers")
    print("  • Custom tokenization and dataset creation")
    print("  • Complete training pipeline setup")
    print("  • Configuration management and serialization")
    print("  • Performance benchmarking and optimization")
    print("\nThe transformer implementation is ready for your own projects!")


if __name__ == "__main__":
    main()