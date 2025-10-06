"""
Utility functions for transformer model development and analysis.

This module provides helpful utilities for model analysis, visualization,
debugging, and performance monitoring.
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt


class ModelAnalyzer:
    """
    Analyzer for transformer model behavior and performance.
    """
    
    def __init__(self, model):
        """
        Initialize model analyzer.
        
        Args:
            model: Transformer model to analyze.
        """
        self.model = model
        self.analysis_results = {}
    
    def analyze_attention_patterns(self, embeddings: np.ndarray, 
                                 layer_idx: int = 0, head_idx: int = 0) -> np.ndarray:
        """
        Analyze attention patterns for a given layer and head.
        
        Args:
            embeddings: Input embeddings.
            layer_idx: Layer index to analyze.
            head_idx: Attention head index.
            
        Returns:
            Attention weight matrix.
        """
        if layer_idx >= len(self.model.transformer_blocks):
            raise ValueError(f"Layer index {layer_idx} out of range")
        
        transformer_block = self.model.transformer_blocks[layer_idx]
        attention_layer = transformer_block.multi_head_attention
        
        if head_idx >= len(attention_layer.attention_heads):
            raise ValueError(f"Head index {head_idx} out of range")
        
        # Get attention weights for specific head
        attention_head = attention_layer.attention_heads[head_idx]
        
        # Forward pass through attention head
        seq_len = embeddings.shape[0]
        head_dim = attention_head.embedding_dim
        
        # Extract embeddings for this head
        embeddings_head = embeddings[:, head_idx * head_dim:(head_idx + 1) * head_dim]
        
        # Compute Q, K, V
        query = np.dot(embeddings_head, attention_head.W_q)
        key = np.dot(embeddings_head, attention_head.W_k)
        
        # Compute attention scores
        attention_scores = attention_head.calculate_attention_score(query, key)
        
        # Apply softmax to get attention weights
        from attention import softmax
        attention_weights = softmax(attention_scores)
        
        return attention_weights
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count the number of parameters in the model.
        
        Returns:
            Dictionary with parameter counts by component.
        """
        param_counts = {}
        total_params = 0
        
        # Count transformer block parameters
        if hasattr(self.model, 'transformer_blocks'):
            block_params = 0
            for block in self.model.transformer_blocks:
                # Multi-head attention parameters
                for head in block.multi_head_attention.attention_heads:
                    block_params += head.W_q.size
                    block_params += head.W_k.size
                    block_params += head.W_v.size
                
                # Output projection
                block_params += block.multi_head_attention.W_o.size
                
                # Layer normalization parameters
                block_params += block.layer_norm1.gamma.size
                block_params += block.layer_norm1.beta.size
                block_params += block.layer_norm2.gamma.size
                block_params += block.layer_norm2.beta.size
                
                # Feedforward parameters
                block_params += block.feedforward.W1.size
                block_params += block.feedforward.b1.size
                block_params += block.feedforward.W2.size
                block_params += block.feedforward.b2.size
            
            param_counts['transformer_blocks'] = block_params
            total_params += block_params
        
        # Count output projection parameters
        if hasattr(self.model, 'output_projection'):
            output_params = self.model.output_projection.size
            param_counts['output_projection'] = output_params
            total_params += output_params
        
        param_counts['total'] = total_params
        
        return param_counts
    
    def analyze_gradient_flow(self, gradients: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze gradient flow in the model.
        
        Args:
            gradients: Dictionary of gradients by parameter name.
            
        Returns:
            Dictionary with gradient statistics.
        """
        grad_stats = {}
        
        for name, grad in gradients.items():
            grad_norm = np.linalg.norm(grad)
            grad_mean = np.mean(np.abs(grad))
            grad_std = np.std(grad)
            grad_max = np.max(np.abs(grad))
            
            grad_stats[name] = {
                'norm': grad_norm,
                'mean_abs': grad_mean,
                'std': grad_std,
                'max_abs': grad_max
            }
        
        return grad_stats
    
    def compute_model_flops(self, seq_len: int) -> Dict[str, int]:
        """
        Compute floating-point operations for the model.
        
        Args:
            seq_len: Input sequence length.
            
        Returns:
            Dictionary with FLOP counts by component.
        """
        flops = {}
        d_model = self.model.d_model
        num_heads = self.model.num_heads
        num_layers = len(self.model.transformer_blocks)
        
        # Attention FLOPs per layer
        # Q, K, V projections: 3 * seq_len * d_model^2
        # Attention scores: seq_len^2 * d_model
        # Attention output: seq_len^2 * d_model
        # Output projection: seq_len * d_model^2
        attention_flops = (3 * seq_len * d_model**2 + 
                          2 * seq_len**2 * d_model + 
                          seq_len * d_model**2)
        
        # Feedforward FLOPs per layer
        d_ff = self.model.transformer_blocks[0].d_ff if num_layers > 0 else 4 * d_model
        ff_flops = 2 * seq_len * d_model * d_ff
        
        # Total per layer
        layer_flops = attention_flops + ff_flops
        
        flops['attention_per_layer'] = attention_flops
        flops['feedforward_per_layer'] = ff_flops
        flops['total_per_layer'] = layer_flops
        flops['total_model'] = layer_flops * num_layers
        
        return flops


class PerformanceProfiler:
    """
    Profiler for measuring model performance and bottlenecks.
    """
    
    def __init__(self):
        """Initialize profiler."""
        self.timings = {}
        self.memory_usage = {}
        
    def profile_forward_pass(self, model, embeddings: np.ndarray, 
                           num_runs: int = 10) -> Dict[str, float]:
        """
        Profile forward pass performance.
        
        Args:
            model: Model to profile.
            embeddings: Input embeddings.
            num_runs: Number of runs for averaging.
            
        Returns:
            Dictionary with timing results.
        """
        # Warmup
        for _ in range(3):
            _ = model.forward(embeddings)
        
        # Profile components
        results = {}
        
        # Total forward pass
        start_time = time.time()
        for _ in range(num_runs):
            output = model.forward(embeddings)
        total_time = (time.time() - start_time) / num_runs
        results['total_forward_pass'] = total_time
        
        # Profile individual transformer blocks
        if hasattr(model, 'transformer_blocks'):
            block_times = []
            x = embeddings
            
            for i, block in enumerate(model.transformer_blocks):
                start_time = time.time()
                for _ in range(num_runs):
                    _ = block.forward(x)
                block_time = (time.time() - start_time) / num_runs
                block_times.append(block_time)
                
                # Update x for next block (simplified)
                x = block.forward(x)
            
            results['transformer_blocks'] = block_times
            results['avg_block_time'] = np.mean(block_times)
        
        return results
    
    def profile_attention_scaling(self, model_class, d_model_range: List[int], 
                                seq_len: int = 64) -> Dict[int, float]:
        """
        Profile attention mechanism scaling with model dimension.
        
        Args:
            model_class: Model class to instantiate.
            d_model_range: List of model dimensions to test.
            seq_len: Sequence length for testing.
            
        Returns:
            Dictionary mapping d_model to timing.
        """
        results = {}
        
        for d_model in d_model_range:
            num_heads = min(8, d_model // 64)  # Reasonable number of heads
            if d_model % num_heads != 0:
                continue
                
            try:
                model = model_class(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_layers=2
                )
                
                embeddings = np.random.randn(seq_len, d_model)
                
                # Profile
                timing_results = self.profile_forward_pass(model, embeddings, num_runs=5)
                results[d_model] = timing_results['total_forward_pass']
                
            except Exception as e:
                print(f"Failed to profile d_model={d_model}: {e}")
        
        return results


def visualize_attention_weights(attention_weights: np.ndarray, 
                              tokens: Optional[List[str]] = None,
                              save_path: Optional[str] = None):
    """
    Visualize attention weight matrix as a heatmap.
    
    Args:
        attention_weights: Attention weight matrix.
        tokens: List of token strings for labeling.
        save_path: Path to save the visualization.
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set labels
        if tokens:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_yticklabels(tokens)
        
        ax.set_xlabel('Key Tokens')
        ax.set_ylabel('Query Tokens')
        ax.set_title('Attention Weights Heatmap')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention visualization saved to {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")


def debug_tensor_shapes(tensors: Dict[str, np.ndarray], 
                       expected_shapes: Optional[Dict[str, Tuple]] = None):
    """
    Debug tensor shapes and print information.
    
    Args:
        tensors: Dictionary of tensors to debug.
        expected_shapes: Expected shapes for validation.
    """
    print("Tensor Shape Debug Information")
    print("-" * 40)
    
    for name, tensor in tensors.items():
        print(f"{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Range: [{tensor.min():.4f}, {tensor.max():.4f}]")
        print(f"  Mean: {tensor.mean():.4f}")
        print(f"  Std: {tensor.std():.4f}")
        
        # Check for problematic values
        if np.any(np.isnan(tensor)):
            print(f"  ⚠️  Contains NaN values!")
        if np.any(np.isinf(tensor)):
            print(f"  ⚠️  Contains Inf values!")
        
        # Validate expected shape
        if expected_shapes and name in expected_shapes:
            expected = expected_shapes[name]
            if tensor.shape != expected:
                print(f"  ❌ Shape mismatch! Expected {expected}")
            else:
                print(f"  ✅ Shape matches expected")
        
        print()


def save_model_summary(model, filepath: str):
    """
    Save a comprehensive model summary to file.
    
    Args:
        model: Model to summarize.
        filepath: Path to save summary.
    """
    analyzer = ModelAnalyzer(model)
    
    summary = {
        'model_type': model.__class__.__name__,
        'parameters': analyzer.count_parameters(),
        'architecture': {
            'd_model': getattr(model, 'd_model', 'Unknown'),
            'num_heads': getattr(model, 'num_heads', 'Unknown'),
            'num_layers': len(getattr(model, 'transformer_blocks', [])),
        }
    }
    
    # Add FLOP estimates for common sequence lengths
    if hasattr(model, 'd_model'):
        flop_estimates = {}
        for seq_len in [32, 64, 128, 256, 512]:
            try:
                flops = analyzer.compute_model_flops(seq_len)
                flop_estimates[seq_len] = flops['total_model']
            except:
                pass
        summary['flop_estimates'] = flop_estimates
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Model summary saved to {filepath}")


def compare_models(models: Dict[str, Any], test_input: np.ndarray) -> Dict[str, Dict]:
    """
    Compare multiple models on the same input.
    
    Args:
        models: Dictionary of model name to model object.
        test_input: Test input for comparison.
        
    Returns:
        Dictionary with comparison results.
    """
    profiler = PerformanceProfiler()
    results = {}
    
    for name, model in models.items():
        print(f"Analyzing model: {name}")
        
        try:
            # Performance profiling
            timing_results = profiler.profile_forward_pass(model, test_input)
            
            # Parameter count
            analyzer = ModelAnalyzer(model)
            param_counts = analyzer.count_parameters()
            
            # Model output
            output = model.forward(test_input)
            
            results[name] = {
                'timing': timing_results,
                'parameters': param_counts,
                'output_shape': output.shape,
                'output_stats': {
                    'mean': float(np.mean(output)),
                    'std': float(np.std(output)),
                    'min': float(np.min(output)),
                    'max': float(np.max(output))
                }
            }
            
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Model Utilities - Example Usage")
    print("=" * 40)
    
    # This would typically be run with actual models
    print("Import this module and use the utilities with your transformer models!")
    print("\nExample:")
    print("from utils import ModelAnalyzer, PerformanceProfiler")
    print("analyzer = ModelAnalyzer(your_model)")
    print("param_counts = analyzer.count_parameters()")