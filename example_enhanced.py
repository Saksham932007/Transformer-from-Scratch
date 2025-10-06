#!/usr/bin/env python3
"""
Comprehensive example demonstrating the enhanced transformer implementation.

This script showcases all the improvements made to the transformer codebase
including performance monitoring, visualization, checkpointing, and more.
"""

import numpy as np
import os
import sys
from typing import Dict, Any

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformer import Transformer
from attention import MultiHeadAttention
from embed import tokenize_and_embed, embedding_model
from layers import TransformerBlock
from performance import global_profiler, ModelPerformanceTracker
from checkpoint import ModelStateManager
from visualization import VisualizationManager
from validation import ModelValidator, validate_transformer_model
from optimization import OptimizerFactory


def demonstrate_basic_functionality():
    """Demonstrate basic transformer functionality with improvements."""
    print("=" * 60)
    print("BASIC TRANSFORMER FUNCTIONALITY DEMONSTRATION")
    print("=" * 60)
    
    # Initialize transformer with enhanced error handling
    try:
        embedding_dim = 300
        num_heads = 6
        transformer = Transformer(embedding_dim, num_heads)
        print(f"✓ Created transformer with {embedding_dim}D embeddings and {num_heads} heads")
    except Exception as e:
        print(f"✗ Failed to create transformer: {e}")
        return
    
    # Test sentence completion with performance monitoring
    test_sentence = "the quick brown fox"
    print(f"\nInput sentence: '{test_sentence}'")
    
    with global_profiler.profile("sentence_completion", 1):
        try:
            completed = transformer.complete_sentence(test_sentence, max_length=10, temperature=0.8)
            print(f"Completed sentence: '{completed}'")
        except Exception as e:
            print(f"✗ Sentence completion failed: {e}")
            return
    
    # Show performance metrics
    summary = global_profiler.get_summary("sentence_completion")
    if "sentence_completion" in summary:
        stats = summary["sentence_completion"]
        print(f"Performance: {stats['avg_execution_time']:.4f}s, "
              f"{stats['avg_memory_usage_mb']:.2f}MB")


def demonstrate_validation():
    """Demonstrate model validation capabilities."""
    print("\n" + "=" * 60)
    print("MODEL VALIDATION DEMONSTRATION")
    print("=" * 60)
    
    # Create a model configuration
    model_config = {
        'embedding_dim': 300,
        'num_heads': 6,
        'max_sequence_length': 512,
        'num_layers': 1
    }
    
    # Create and validate transformer
    transformer = Transformer(model_config['embedding_dim'], model_config['num_heads'])
    
    # Run comprehensive validation
    validation_results = validate_transformer_model(transformer, model_config)
    
    # Print validation report
    validator = ModelValidator()
    validator.print_validation_report(validation_results)


def demonstrate_optimization():
    """Demonstrate advanced optimization features."""
    print("\n" + "=" * 60)
    print("ADVANCED OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Create optimizer configuration
    optimizer_config = {
        'optimizer': {
            'type': 'adamw',
            'learning_rate': 0.001,
            'weight_decay': 0.01
        },
        'scheduler': {
            'type': 'cosine',
            'T_max': 100
        },
        'gradient_clipping': {
            'method': 'norm',
            'max_norm': 1.0
        },
        'accumulation_steps': 4
    }
    
    try:
        training_optimizer = OptimizerFactory.create_training_optimizer(optimizer_config)
        print("✓ Created advanced training optimizer with:")
        print("  - AdamW optimizer with weight decay")
        print("  - Cosine annealing learning rate schedule")
        print("  - Gradient clipping by norm")
        print("  - Gradient accumulation over 4 steps")
    except Exception as e:
        print(f"✗ Failed to create optimizer: {e}")


def demonstrate_checkpointing():
    """Demonstrate model checkpointing and state management."""
    print("\n" + "=" * 60)
    print("MODEL CHECKPOINTING DEMONSTRATION")
    print("=" * 60)
    
    # Create model and state manager
    transformer = Transformer(128, 4)  # Smaller model for demo
    state_manager = ModelStateManager()
    
    try:
        # Save model
        save_path = "demo_checkpoint"
        checkpoint_path = state_manager.save_model(
            transformer, 
            save_path,
            metadata={'epoch': 10, 'loss': 0.5, 'demo': True}
        )
        print(f"✓ Saved model checkpoint to: {os.path.basename(checkpoint_path)}")
        
        # Create new model and load checkpoint
        new_transformer = Transformer(128, 4)
        success = state_manager.load_model(new_transformer, checkpoint_path)
        
        if success:
            print("✓ Successfully loaded model from checkpoint")
        else:
            print("✗ Failed to load model from checkpoint")
            
    except Exception as e:
        print(f"✗ Checkpointing demo failed: {e}")


def demonstrate_performance_tracking():
    """Demonstrate performance tracking capabilities."""
    print("\n" + "=" * 60)
    print("PERFORMANCE TRACKING DEMONSTRATION")
    print("=" * 60)
    
    # Create performance tracker
    tracker = ModelPerformanceTracker()
    
    # Simulate training epochs
    transformer = Transformer(256, 8)
    test_sentence = "artificial intelligence and machine learning"
    
    for epoch in range(3):
        tracker.start_epoch()
        
        # Simulate forward passes
        for step in range(5):
            with global_profiler.profile(f"forward_pass_epoch_{epoch}", 1):
                _ = transformer.complete_sentence(test_sentence, max_length=8, temperature=1.0)
        
        # Simulate loss calculation
        loss = 2.0 - epoch * 0.3  # Decreasing loss
        tracker.end_epoch(num_samples=5, loss=loss)
    
    # Get training summary
    summary = tracker.get_training_summary()
    print("Training Summary:")
    print(f"  - Total Epochs: {summary.get('total_epochs', 0)}")
    print(f"  - Average Epoch Time: {summary.get('avg_epoch_time', 0):.4f}s")
    print(f"  - Final Loss: {summary.get('final_loss', 0):.4f}")
    print(f"  - Average Throughput: {summary.get('avg_throughput', 0):.2f} samples/sec")


def demonstrate_visualization():
    """Demonstrate visualization capabilities."""
    print("\n" + "=" * 60)
    print("VISUALIZATION DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Create visualization manager
        viz_manager = VisualizationManager("./demo_visualizations")
        
        # Generate sample data for visualization
        seq_len = 8
        embedding_dim = 64
        
        # Create sample attention weights (multiple heads)
        attention_weights = []
        for head in range(4):
            # Create more realistic attention pattern
            weights = np.random.rand(seq_len, seq_len)
            # Make it causal (lower triangular)
            weights = np.tril(weights)
            # Normalize to sum to 1
            weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-8)
            attention_weights.append(weights)
        
        # Create sample embeddings
        embeddings = np.random.randn(seq_len, embedding_dim)
        
        # Create sample training history
        training_history = {
            'train_losses': [2.5, 2.1, 1.8, 1.5, 1.3, 1.1, 1.0],
            'val_losses': [2.6, 2.2, 1.9, 1.6, 1.4, 1.2, 1.1],
            'learning_rates': [0.001, 0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001]
        }
        
        model_data = {
            'attention_weights': attention_weights,
            'tokens': ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog'],
            'embeddings': embeddings,
            'embedding_labels': [f'token_{i}' for i in range(seq_len)],
            'training_history': training_history,
            'embedding_dim': embedding_dim,
            'num_heads': 4,
            'seq_len': seq_len
        }
        
        # Create comprehensive visualization report
        report_dir = viz_manager.create_model_report(model_data, "demo_analysis")
        print(f"✓ Created visualization report in: {os.path.basename(report_dir)}")
        print("  - Attention heatmaps for all heads")
        print("  - Embedding PCA visualization")
        print("  - Training progress curves")
        print("  - HTML summary report")
        
    except Exception as e:
        print(f"✗ Visualization demo failed: {e}")


def demonstrate_data_pipeline():
    """Demonstrate data processing pipeline."""
    print("\n" + "=" * 60)
    print("DATA PROCESSING PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Create sample text data
        sample_texts = [
            "The transformer architecture revolutionized natural language processing.",
            "Attention mechanisms allow models to focus on relevant parts of input.",
            "Self-attention computes relationships between all positions in a sequence.",
            "Multi-head attention captures different types of relationships.",
            "Positional encoding provides sequence order information to the model."
        ]
        
        # Process the texts using existing functionality
        print("Processing sample texts...")
        processed_data = []
        for i, text in enumerate(sample_texts):
            try:
                # Tokenize and embed
                embeddings = tokenize_and_embed(text, embedding_model)
                processed_data.append({
                    'text': text,
                    'embeddings': embeddings,
                    'length': len(embeddings)
                })
                print(f"  ✓ Text {i+1}: {len(embeddings)} tokens")
            except Exception as e:
                print(f"  ✗ Text {i+1} failed: {e}")
        
        print(f"\n✓ Successfully processed {len(processed_data)} texts")
        print(f"Average sequence length: {np.mean([d['length'] for d in processed_data]):.1f}")
        
    except Exception as e:
        print(f"✗ Data pipeline demo failed: {e}")


def run_complete_demonstration():
    """Run the complete demonstration of all features."""
    print("TRANSFORMER FROM SCRATCH - ENHANCED VERSION")
    print("Comprehensive Feature Demonstration")
    print()
    
    # Run all demonstrations
    demonstrate_basic_functionality()
    demonstrate_validation()
    demonstrate_optimization()
    demonstrate_checkpointing()
    demonstrate_performance_tracking()
    demonstrate_visualization()
    demonstrate_data_pipeline()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nFeatures demonstrated:")
    print("✓ Enhanced transformer architecture with error handling")
    print("✓ Comprehensive model validation")
    print("✓ Advanced optimization techniques")
    print("✓ Model checkpointing and state management")
    print("✓ Performance monitoring and profiling")
    print("✓ Visualization tools for analysis")
    print("✓ Data processing pipeline")
    print("\nFor more details, check the individual module files:")
    print("- transformer.py: Core transformer implementation")
    print("- attention.py: Attention mechanisms")
    print("- layers.py: Layer normalization and feedforward networks")
    print("- optimization.py: Advanced optimizers and schedulers")
    print("- checkpoint.py: Model persistence")
    print("- performance.py: Performance monitoring")
    print("- visualization.py: Visualization tools")
    print("- validation.py: Model validation")
    print("- data_pipeline.py: Data processing utilities")


if __name__ == "__main__":
    # Allow running specific demonstrations
    if len(sys.argv) > 1:
        demo_map = {
            'basic': demonstrate_basic_functionality,
            'validation': demonstrate_validation,
            'optimization': demonstrate_optimization,
            'checkpoint': demonstrate_checkpointing,
            'performance': demonstrate_performance_tracking,
            'visualization': demonstrate_visualization,
            'data': demonstrate_data_pipeline
        }
        
        demo_name = sys.argv[1].lower()
        if demo_name in demo_map:
            demo_map[demo_name]()
        else:
            print(f"Unknown demo: {demo_name}")
            print(f"Available demos: {', '.join(demo_map.keys())}")
    else:
        run_complete_demonstration()