"""
Performance monitoring and profiling utilities for transformer training and inference.

This module provides tools for tracking model performance, memory usage,
and execution time for optimization purposes.
"""

import numpy as np
import time
import psutil
import os
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_percent: float
    operations_count: int
    throughput: float  # operations per second


class PerformanceProfiler:
    """
    Comprehensive performance profiler for transformer operations.
    
    Tracks execution time, memory usage, and computational throughput
    for different components of the transformer model.
    """
    
    def __init__(self):
        """Initialize the performance profiler."""
        self.metrics_history: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.current_operation = None
        self.start_time = None
        self.start_memory = None
        self.operations_count = 0
        
    @contextmanager
    def profile(self, operation_name: str, operations_count: int = 1):
        """
        Context manager for profiling a specific operation.
        
        Args:
            operation_name (str): Name of the operation being profiled.
            operations_count (int): Number of operations performed.
            
        Yields:
            None
        """
        # Start profiling
        self.current_operation = operation_name
        self.operations_count = operations_count
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        
        process = psutil.Process(os.getpid())
        start_cpu_time = process.cpu_times()
        
        try:
            yield
        finally:
            # End profiling
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu_time = process.cpu_times()
            
            execution_time = end_time - self.start_time
            memory_usage = end_memory - self.start_memory
            peak_memory = max(self.start_memory, end_memory)
            
            # Calculate CPU usage percentage
            total_cpu_time = ((end_cpu_time.user - start_cpu_time.user) + 
                            (end_cpu_time.system - start_cpu_time.system))
            cpu_percent = (total_cpu_time / execution_time) * 100 if execution_time > 0 else 0
            
            # Calculate throughput
            throughput = operations_count / execution_time if execution_time > 0 else 0
            
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                peak_memory_mb=peak_memory,
                cpu_percent=cpu_percent,
                operations_count=operations_count,
                throughput=throughput
            )
            
            self.metrics_history[operation_name].append(metrics)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    def get_summary(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Args:
            operation_name (str, optional): Specific operation to analyze.
                                          If None, returns summary for all operations.
        
        Returns:
            Dict[str, Any]: Summary statistics.
        """
        if operation_name:
            operations = {operation_name: self.metrics_history[operation_name]}
        else:
            operations = self.metrics_history
        
        summary = {}
        
        for op_name, metrics_list in operations.items():
            if not metrics_list:
                continue
                
            execution_times = [m.execution_time for m in metrics_list]
            memory_usages = [m.memory_usage_mb for m in metrics_list]
            throughputs = [m.throughput for m in metrics_list]
            
            summary[op_name] = {
                'count': len(metrics_list),
                'avg_execution_time': np.mean(execution_times),
                'min_execution_time': np.min(execution_times),
                'max_execution_time': np.max(execution_times),
                'std_execution_time': np.std(execution_times),
                'avg_memory_usage_mb': np.mean(memory_usages),
                'max_memory_usage_mb': np.max(memory_usages),
                'avg_throughput': np.mean(throughputs),
                'max_throughput': np.max(throughputs),
                'total_operations': sum(m.operations_count for m in metrics_list)
            }
        
        return summary
    
    def reset(self):
        """Reset all collected metrics."""
        self.metrics_history.clear()
    
    def export_metrics(self, filepath: str):
        """
        Export metrics to a file.
        
        Args:
            filepath (str): Path to save the metrics.
        """
        summary = self.get_summary()
        
        with open(filepath, 'w') as f:
            f.write("Performance Metrics Summary\n")
            f.write("===========================\n\n")
            
            for op_name, stats in summary.items():
                f.write(f"Operation: {op_name}\n")
                f.write(f"  Count: {stats['count']}\n")
                f.write(f"  Avg Execution Time: {stats['avg_execution_time']:.4f}s\n")
                f.write(f"  Min Execution Time: {stats['min_execution_time']:.4f}s\n")
                f.write(f"  Max Execution Time: {stats['max_execution_time']:.4f}s\n")
                f.write(f"  Std Execution Time: {stats['std_execution_time']:.4f}s\n")
                f.write(f"  Avg Memory Usage: {stats['avg_memory_usage_mb']:.2f} MB\n")
                f.write(f"  Max Memory Usage: {stats['max_memory_usage_mb']:.2f} MB\n")
                f.write(f"  Avg Throughput: {stats['avg_throughput']:.2f} ops/sec\n")
                f.write(f"  Max Throughput: {stats['max_throughput']:.2f} ops/sec\n")
                f.write(f"  Total Operations: {stats['total_operations']}\n\n")


class MatrixOperationOptimizer:
    """
    Optimizer for common matrix operations used in transformers.
    
    Provides optimized implementations and benchmarking for key operations
    like matrix multiplication, softmax, and layer normalization.
    """
    
    def __init__(self):
        """Initialize the optimizer."""
        self.profiler = PerformanceProfiler()
    
    def benchmark_matrix_multiplication(self, sizes: List[Tuple[int, int, int]], 
                                      num_runs: int = 10) -> Dict[str, Any]:
        """
        Benchmark matrix multiplication with different sizes.
        
        Args:
            sizes (List[Tuple[int, int, int]]): List of (m, n, k) matrix dimensions.
            num_runs (int): Number of runs for each size.
            
        Returns:
            Dict[str, Any]: Benchmark results.
        """
        results = {}
        
        for m, n, k in sizes:
            size_key = f"{m}x{n}x{k}"
            times = []
            
            for _ in range(num_runs):
                A = np.random.randn(m, n)
                B = np.random.randn(n, k)
                
                with self.profiler.profile(f"matmul_{size_key}", 1):
                    _ = np.dot(A, B)
            
            summary = self.profiler.get_summary(f"matmul_{size_key}")
            results[size_key] = summary[f"matmul_{size_key}"]
            
        return results
    
    def benchmark_softmax(self, shapes: List[Tuple[int, ...]], 
                         num_runs: int = 10) -> Dict[str, Any]:
        """
        Benchmark softmax operation with different shapes.
        
        Args:
            shapes (List[Tuple[int, ...]]): List of tensor shapes.
            num_runs (int): Number of runs for each shape.
            
        Returns:
            Dict[str, Any]: Benchmark results.
        """
        from attention import softmax
        
        results = {}
        
        for shape in shapes:
            shape_key = "x".join(map(str, shape))
            
            for _ in range(num_runs):
                x = np.random.randn(*shape)
                
                with self.profiler.profile(f"softmax_{shape_key}", 1):
                    _ = softmax(x)
            
            summary = self.profiler.get_summary(f"softmax_{shape_key}")
            results[shape_key] = summary[f"softmax_{shape_key}"]
            
        return results
    
    def optimize_attention_computation(self, seq_len: int, embedding_dim: int,
                                     num_heads: int) -> Dict[str, float]:
        """
        Optimize attention computation for given parameters.
        
        Args:
            seq_len (int): Sequence length.
            embedding_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            
        Returns:
            Dict[str, float]: Optimization recommendations.
        """
        head_dim = embedding_dim // num_heads
        
        # Estimate memory requirements
        qkv_memory = 3 * seq_len * embedding_dim * 8  # 8 bytes per float64
        attention_scores_memory = seq_len * seq_len * num_heads * 8
        total_memory_mb = (qkv_memory + attention_scores_memory) / (1024 * 1024)
        
        # Estimate computational complexity
        qkv_flops = 3 * seq_len * embedding_dim * embedding_dim
        attention_flops = num_heads * seq_len * seq_len * head_dim
        total_flops = qkv_flops + attention_flops
        
        return {
            'estimated_memory_mb': total_memory_mb,
            'estimated_flops': total_flops,
            'recommended_batch_size': max(1, int(1000 / total_memory_mb)),  # Rough estimate
            'memory_per_sample_mb': total_memory_mb
        }


class ModelPerformanceTracker:
    """
    High-level performance tracker for transformer models.
    
    Tracks training and inference performance with detailed breakdowns
    by model components.
    """
    
    def __init__(self):
        """Initialize the performance tracker."""
        self.profiler = PerformanceProfiler()
        self.epoch_metrics = []
        self.component_times = defaultdict(list)
    
    def start_epoch(self):
        """Start tracking a new epoch."""
        self.epoch_start_time = time.time()
        self.epoch_start_memory = self.profiler._get_memory_usage()
    
    def end_epoch(self, num_samples: int, loss: float):
        """
        End epoch tracking.
        
        Args:
            num_samples (int): Number of samples processed.
            loss (float): Epoch loss.
        """
        epoch_time = time.time() - self.epoch_start_time
        epoch_memory = self.profiler._get_memory_usage() - self.epoch_start_memory
        
        self.epoch_metrics.append({
            'epoch_time': epoch_time,
            'memory_usage_mb': epoch_memory,
            'samples_per_second': num_samples / epoch_time,
            'loss': loss,
            'timestamp': time.time()
        })
    
    def track_component(self, component_name: str, execution_time: float):
        """
        Track execution time for a specific component.
        
        Args:
            component_name (str): Name of the component.
            execution_time (float): Execution time in seconds.
        """
        self.component_times[component_name].append(execution_time)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training performance summary.
        
        Returns:
            Dict[str, Any]: Training performance summary.
        """
        if not self.epoch_metrics:
            return {}
        
        epoch_times = [m['epoch_time'] for m in self.epoch_metrics]
        memory_usages = [m['memory_usage_mb'] for m in self.epoch_metrics]
        throughputs = [m['samples_per_second'] for m in self.epoch_metrics]
        losses = [m['loss'] for m in self.epoch_metrics]
        
        component_summary = {}
        for comp_name, times in self.component_times.items():
            component_summary[comp_name] = {
                'avg_time': np.mean(times),
                'total_time': np.sum(times),
                'percentage': (np.sum(times) / np.sum(epoch_times)) * 100
            }
        
        return {
            'total_epochs': len(self.epoch_metrics),
            'avg_epoch_time': np.mean(epoch_times),
            'total_training_time': np.sum(epoch_times),
            'avg_memory_usage_mb': np.mean(memory_usages),
            'peak_memory_usage_mb': np.max(memory_usages),
            'avg_throughput': np.mean(throughputs),
            'final_loss': losses[-1] if losses else 0,
            'component_breakdown': component_summary
        }


# Global profiler instance for easy access
global_profiler = PerformanceProfiler()