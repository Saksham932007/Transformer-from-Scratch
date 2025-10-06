"""
Performance benchmarks for transformer components.

This module provides benchmarking utilities to measure and compare
the performance of different transformer implementations.
"""

import sys
import os
import numpy as np
import time
from typing import Dict, List, Tuple, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention import SelfAttention, MultiHeadAttention
from embed import add_positional_encoding
from layers import LayerNormalization, FeedForwardNetwork
from performance import PerformanceProfiler, MatrixOperationOptimizer


class TransformerBenchmark:
    """Comprehensive benchmark suite for transformer components."""
    
    def __init__(self):
        """Initialize the benchmark suite."""
        self.profiler = PerformanceProfiler()
        self.optimizer = MatrixOperationOptimizer()
        self.results = {}
    
    def benchmark_attention_scaling(self) -> Dict[str, List[float]]:
        """Benchmark attention mechanism with different sequence lengths."""
        print("Benchmarking Attention Scaling Performance...")
        
        embedding_dim = 512
        num_heads = 8
        sequence_lengths = [16, 32, 64, 128, 256, 512]
        
        results = {
            'sequence_lengths': sequence_lengths,
            'execution_times': [],
            'memory_usage': [],
            'throughput': []
        }
        
        attention = MultiHeadAttention(embedding_dim, num_heads)
        
        for seq_len in sequence_lengths:
            embeddings = np.random.randn(seq_len, embedding_dim)
            
            # Warm up
            for _ in range(3):
                _ = attention.forward(embeddings)
            
            # Benchmark
            with self.profiler.profile(f"attention_seq_{seq_len}", 10):
                for _ in range(10):
                    _ = attention.forward(embeddings)
            
            summary = self.profiler.get_summary(f"attention_seq_{seq_len}")
            stats = summary[f"attention_seq_{seq_len}"]
            
            results['execution_times'].append(stats['avg_execution_time'])
            results['memory_usage'].append(stats['avg_memory_usage_mb'])
            results['throughput'].append(stats['avg_throughput'])
            
            print(f"Seq Len {seq_len:3d}: {stats['avg_execution_time']:.4f}s, "
                  f"{stats['avg_memory_usage_mb']:.2f}MB, "
                  f"{stats['avg_throughput']:.2f} ops/sec")
        
        self.results['attention_scaling'] = results
        return results
    
    def benchmark_positional_encoding(self) -> Dict[str, List[float]]:
        """Benchmark positional encoding with different sequence lengths."""
        print("\nBenchmarking Positional Encoding Performance...")
        
        embedding_dim = 512
        sequence_lengths = [64, 128, 256, 512, 1024, 2048]
        
        results = {
            'sequence_lengths': sequence_lengths,
            'execution_times': [],
            'memory_usage': []
        }
        
        for seq_len in sequence_lengths:
            embeddings = np.random.randn(seq_len, embedding_dim)
            
            # Benchmark
            with self.profiler.profile(f"pos_enc_seq_{seq_len}", 20):
                for _ in range(20):
                    _ = add_positional_encoding(embeddings)
            
            summary = self.profiler.get_summary(f"pos_enc_seq_{seq_len}")
            stats = summary[f"pos_enc_seq_{seq_len}"]
            
            results['execution_times'].append(stats['avg_execution_time'])
            results['memory_usage'].append(stats['avg_memory_usage_mb'])
            
            print(f"Seq Len {seq_len:4d}: {stats['avg_execution_time']:.6f}s, "
                  f"{stats['avg_memory_usage_mb']:.2f}MB")
        
        self.results['positional_encoding'] = results
        return results
    
    def benchmark_layer_normalization(self) -> Dict[str, List[float]]:
        """Benchmark layer normalization with different dimensions."""
        print("\nBenchmarking Layer Normalization Performance...")
        
        embedding_dims = [128, 256, 512, 768, 1024, 1536]
        seq_len = 128
        
        results = {
            'embedding_dims': embedding_dims,
            'execution_times': [],
            'memory_usage': []
        }
        
        for emb_dim in embedding_dims:
            layer_norm = LayerNormalization(emb_dim)
            embeddings = np.random.randn(seq_len, emb_dim)
            
            # Benchmark
            with self.profiler.profile(f"layer_norm_dim_{emb_dim}", 50):
                for _ in range(50):
                    _ = layer_norm.forward(embeddings)
            
            summary = self.profiler.get_summary(f"layer_norm_dim_{emb_dim}")
            stats = summary[f"layer_norm_dim_{emb_dim}"]
            
            results['execution_times'].append(stats['avg_execution_time'])
            results['memory_usage'].append(stats['avg_memory_usage_mb'])
            
            print(f"Emb Dim {emb_dim:4d}: {stats['avg_execution_time']:.6f}s, "
                  f"{stats['avg_memory_usage_mb']:.2f}MB")
        
        self.results['layer_normalization'] = results
        return results
    
    def benchmark_feedforward_network(self) -> Dict[str, List[float]]:
        """Benchmark feedforward network with different dimensions."""
        print("\nBenchmarking Feedforward Network Performance...")
        
        embedding_dim = 512
        hidden_multipliers = [1, 2, 4, 6, 8]
        seq_len = 128
        
        results = {
            'hidden_multipliers': hidden_multipliers,
            'execution_times': [],
            'memory_usage': []
        }
        
        for multiplier in hidden_multipliers:
            hidden_dim = embedding_dim * multiplier
            ffn = FeedForwardNetwork(embedding_dim, hidden_dim)
            embeddings = np.random.randn(seq_len, embedding_dim)
            
            # Benchmark
            with self.profiler.profile(f"ffn_mult_{multiplier}", 20):
                for _ in range(20):
                    _ = ffn.forward(embeddings)
            
            summary = self.profiler.get_summary(f"ffn_mult_{multiplier}")
            stats = summary[f"ffn_mult_{multiplier}"]
            
            results['execution_times'].append(stats['avg_execution_time'])
            results['memory_usage'].append(stats['avg_memory_usage_mb'])
            
            print(f"Hidden Mult {multiplier}x: {stats['avg_execution_time']:.4f}s, "
                  f"{stats['avg_memory_usage_mb']:.2f}MB")
        
        self.results['feedforward_network'] = results
        return results
    
    def benchmark_multi_head_comparison(self) -> Dict[str, List[float]]:
        """Compare performance across different numbers of attention heads."""
        print("\nBenchmarking Multi-Head Attention Head Count...")
        
        embedding_dim = 512
        seq_len = 128
        head_counts = [1, 2, 4, 8, 16]
        
        results = {
            'head_counts': head_counts,
            'execution_times': [],
            'memory_usage': []
        }
        
        for num_heads in head_counts:
            if embedding_dim % num_heads != 0:
                continue
                
            attention = MultiHeadAttention(embedding_dim, num_heads)
            embeddings = np.random.randn(seq_len, embedding_dim)
            
            # Benchmark
            with self.profiler.profile(f"heads_{num_heads}", 20):
                for _ in range(20):
                    _ = attention.forward(embeddings)
            
            summary = self.profiler.get_summary(f"heads_{num_heads}")
            stats = summary[f"heads_{num_heads}"]
            
            results['execution_times'].append(stats['avg_execution_time'])
            results['memory_usage'].append(stats['avg_memory_usage_mb'])
            
            print(f"Heads {num_heads:2d}: {stats['avg_execution_time']:.4f}s, "
                  f"{stats['avg_memory_usage_mb']:.2f}MB")
        
        self.results['multi_head_comparison'] = results
        return results
    
    def run_complete_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        print("=" * 60)
        print("TRANSFORMER PERFORMANCE BENCHMARK SUITE")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all benchmarks
        self.benchmark_attention_scaling()
        self.benchmark_positional_encoding()
        self.benchmark_layer_normalization()
        self.benchmark_feedforward_network()
        self.benchmark_multi_head_comparison()
        
        total_time = time.time() - start_time
        
        print(f"\n" + "=" * 60)
        print(f"BENCHMARK COMPLETE - Total Time: {total_time:.2f}s")
        print("=" * 60)
        
        return self.results
    
    def generate_performance_report(self, output_file: str = "performance_report.txt"):
        """Generate a detailed performance report."""
        if not self.results:
            print("No benchmark results available. Run benchmarks first.")
            return
        
        with open(output_file, 'w') as f:
            f.write("TRANSFORMER PERFORMANCE BENCHMARK REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            for benchmark_name, results in self.results.items():
                f.write(f"{benchmark_name.upper().replace('_', ' ')}\n")
                f.write("-" * len(benchmark_name) + "\n")
                
                if 'sequence_lengths' in results:
                    for i, seq_len in enumerate(results['sequence_lengths']):
                        f.write(f"Seq Len {seq_len:4d}: {results['execution_times'][i]:.6f}s\n")
                elif 'embedding_dims' in results:
                    for i, dim in enumerate(results['embedding_dims']):
                        f.write(f"Emb Dim {dim:4d}: {results['execution_times'][i]:.6f}s\n")
                elif 'hidden_multipliers' in results:
                    for i, mult in enumerate(results['hidden_multipliers']):
                        f.write(f"Hidden {mult}x: {results['execution_times'][i]:.6f}s\n")
                elif 'head_counts' in results:
                    for i, heads in enumerate(results['head_counts']):
                        f.write(f"Heads {heads:2d}: {results['execution_times'][i]:.6f}s\n")
                
                f.write("\n")
        
        print(f"Performance report saved to {output_file}")


def run_quick_benchmark():
    """Run a quick benchmark for basic performance validation."""
    print("Running Quick Performance Validation...")
    
    benchmark = TransformerBenchmark()
    
    # Quick attention test
    embedding_dim = 256
    num_heads = 8
    seq_len = 64
    
    attention = MultiHeadAttention(embedding_dim, num_heads)
    embeddings = np.random.randn(seq_len, embedding_dim)
    
    start_time = time.time()
    for _ in range(100):
        _ = attention.forward(embeddings)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    
    print(f"Quick Benchmark Results:")
    print(f"  - Embedding Dim: {embedding_dim}")
    print(f"  - Sequence Length: {seq_len}")
    print(f"  - Number of Heads: {num_heads}")
    print(f"  - Average Time per Forward Pass: {avg_time:.6f}s")
    print(f"  - Throughput: {1/avg_time:.2f} ops/sec")
    
    return avg_time


if __name__ == "__main__":
    # Choose between quick benchmark or full suite
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_benchmark()
    else:
        benchmark = TransformerBenchmark()
        results = benchmark.run_complete_benchmark()
        benchmark.generate_performance_report()
        
        # Print summary
        print("\nBENCHMARK SUMMARY:")
        print("- Attention scaling performance measured")
        print("- Positional encoding optimization validated")
        print("- Layer normalization efficiency checked")
        print("- Feedforward network performance analyzed")
        print("- Multi-head attention scaling evaluated")
        print("- Report saved to performance_report.txt")