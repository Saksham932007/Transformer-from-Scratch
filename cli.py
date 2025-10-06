#!/usr/bin/env python3
"""
Command Line Interface for Transformer from Scratch.

This CLI provides easy access to common transformer operations including
training, text generation, and model analysis.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

# Import our modules
from config import TransformerConfig, create_small_config, create_default_config
from transformer import Transformer
from data_processing import DataProcessor, create_sample_dataset
from training import Trainer
from utils import ModelAnalyzer, PerformanceProfiler, save_model_summary


def cmd_generate(args):
    """Generate text using a transformer model."""
    print(f"🤖 Generating text completion for: '{args.text}'")
    
    # Load or create configuration
    if args.config:
        config = TransformerConfig.load(args.config)
    else:
        config = create_small_config() if args.small else create_default_config()
    
    # Create model
    model = Transformer(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout_prob=0.0  # Disable dropout for inference
    )
    
    # Set to evaluation mode
    model.set_training_mode(False)
    
    # Generate text
    try:
        completed_text = model.complete_sentence(
            args.text,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        print(f"📝 Completed text: '{completed_text}'")
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(completed_text)
            print(f"💾 Saved to: {args.output}")
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return 1
    
    return 0


def cmd_train(args):
    """Train a transformer model."""
    print("🏋️ Starting transformer training...")
    
    # Load configuration
    if args.config:
        config = TransformerConfig.load(args.config)
    else:
        config = create_small_config() if args.small else create_default_config()
    
    # Override config with command line arguments
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    
    print(f"📊 Configuration: {config.d_model}D, {config.num_heads} heads, {config.num_layers} layers")
    
    # Prepare data
    processor = DataProcessor()
    
    if args.data:
        # Load custom data
        data_path = Path(args.data)
        if data_path.suffix == '.json':
            texts = processor.load_json_file(data_path)
        else:
            texts = processor.load_text_file(data_path)
    else:
        # Use sample data
        texts = create_sample_dataset(args.num_samples)
        print(f"📚 Using {len(texts)} sample texts")
    
    # Prepare training data
    train_loader, val_loader, test_loader, tokenizer = processor.prepare_training_data(
        texts,
        vocab_size=config.vocab_size,
        max_length=config.max_seq_len,
        test_split=0.1,
        val_split=0.1
    )
    
    # Create model
    model = Transformer(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout_prob=config.dropout_prob
    )
    
    # Create trainer
    trainer = Trainer(model, config)
    
    # Simplified training for CLI demonstration
    print("⚠️ Note: CLI training uses simplified setup for demonstration")
    print("📖 For full training, see examples.py or use the API directly")
    
    try:
        # Convert to simplified format for demo
        train_batches = []
        for i, (batch_inputs, batch_targets, attention_mask) in enumerate(train_loader):
            if i >= 10:  # Limit for CLI demo
                break
            sample_input = batch_inputs[0]
            sample_target = batch_targets[0]
            input_embedding = np.random.randn(len(sample_input), config.d_model)
            target_indices = sample_target[:len(sample_input)]
            train_batches.append((input_embedding, target_indices))
        
        trainer.fit(train_batches, num_epochs=min(config.num_epochs, 3))
        print("✅ Training completed!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1
    
    return 0


def cmd_analyze(args):
    """Analyze a transformer model."""
    print("🔍 Analyzing transformer model...")
    
    # Load configuration
    if args.config:
        config = TransformerConfig.load(args.config)
    else:
        config = create_small_config() if args.small else create_default_config()
    
    # Create model
    model = Transformer(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers
    )
    
    # Analyze model
    analyzer = ModelAnalyzer(model)
    
    # Parameter analysis
    param_counts = analyzer.count_parameters()
    print(f"📊 Parameter Analysis:")
    for component, count in param_counts.items():
        print(f"  {component}: {count:,} parameters")
    
    # FLOP analysis
    flops = analyzer.compute_model_flops(args.seq_len)
    print(f"\n⚡ FLOP Analysis (seq_len={args.seq_len}):")
    print(f"  Attention per layer: {flops['attention_per_layer']:,}")
    print(f"  Feedforward per layer: {flops['feedforward_per_layer']:,}")
    print(f"  Total model: {flops['total_model']:,}")
    
    # Performance profiling
    if args.benchmark:
        print(f"\n🏃 Performance Benchmarking...")
        profiler = PerformanceProfiler()
        
        import numpy as np
        test_input = np.random.randn(args.seq_len, config.d_model)
        
        timing_results = profiler.profile_forward_pass(model, test_input)
        print(f"  Forward pass time: {timing_results['total_forward_pass']:.4f}s")
        print(f"  Average block time: {timing_results.get('avg_block_time', 'N/A')}")
    
    # Save summary if requested
    if args.output:
        save_model_summary(model, args.output)
        print(f"💾 Analysis saved to: {args.output}")
    
    return 0


def cmd_config(args):
    """Create or modify configuration files."""
    print("⚙️ Configuration management...")
    
    if args.create:
        # Create new configuration
        if args.template == 'small':
            config = create_small_config()
        elif args.template == 'large':
            config = TransformerConfig(
                d_model=1024,
                num_heads=16,
                num_layers=12,
                d_ff=4096
            )
        else:
            config = create_default_config()
        
        # Save configuration
        config.save(args.output)
        print(f"✅ Created {args.template} configuration: {args.output}")
        
    elif args.show:
        # Show existing configuration
        config = TransformerConfig.load(args.show)
        config_dict = config.to_dict()
        
        print(f"📋 Configuration from {args.show}:")
        for key, value in config_dict.items():
            print(f"  {key}: {value}")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Transformer from Scratch - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s generate "The future of AI" --temperature 0.8
  %(prog)s train --data my_text.txt --epochs 10
  %(prog)s analyze --benchmark --seq-len 128
  %(prog)s config --create --template small -o small_config.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate text using transformer')
    gen_parser.add_argument('text', help='Input text to complete')
    gen_parser.add_argument('--config', '-c', help='Configuration file to use')
    gen_parser.add_argument('--small', action='store_true', help='Use small model configuration')
    gen_parser.add_argument('--temperature', '-t', type=float, default=1.0, help='Generation temperature')
    gen_parser.add_argument('--max-length', '-l', type=int, default=20, help='Maximum generation length')
    gen_parser.add_argument('--output', '-o', help='Output file to save result')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train transformer model')
    train_parser.add_argument('--config', '-c', help='Configuration file to use')
    train_parser.add_argument('--small', action='store_true', help='Use small model configuration')
    train_parser.add_argument('--data', '-d', help='Training data file (text or JSON)')
    train_parser.add_argument('--epochs', '-e', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch-size', '-b', type=int, help='Training batch size')
    train_parser.add_argument('--learning-rate', '-lr', type=float, help='Learning rate')
    train_parser.add_argument('--num-samples', type=int, default=500, help='Number of sample texts if no data file')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze transformer model')
    analyze_parser.add_argument('--config', '-c', help='Configuration file to use')
    analyze_parser.add_argument('--small', action='store_true', help='Use small model configuration')
    analyze_parser.add_argument('--seq-len', type=int, default=64, help='Sequence length for analysis')
    analyze_parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    analyze_parser.add_argument('--output', '-o', help='Output file for analysis results')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configuration files')
    config_group = config_parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument('--create', action='store_true', help='Create new configuration')
    config_group.add_argument('--show', help='Show existing configuration file')
    config_parser.add_argument('--template', choices=['small', 'default', 'large'], 
                              default='default', help='Configuration template')
    config_parser.add_argument('--output', '-o', default='config.json', help='Output configuration file')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Import numpy here to avoid slow startup for help
    global np
    import numpy as np
    
    # Execute command
    try:
        if args.command == 'generate':
            return cmd_generate(args)
        elif args.command == 'train':
            return cmd_train(args)
        elif args.command == 'analyze':
            return cmd_analyze(args)
        elif args.command == 'config':
            return cmd_config(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())