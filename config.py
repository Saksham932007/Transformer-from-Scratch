"""
Configuration Management for Transformer Models

This module provides comprehensive configuration management for transformer models,
including hyperparameter settings, model serialization, and experiment tracking.
"""

import json
import pickle
import numpy as np
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from pathlib import Path
import time


@dataclass
class TransformerConfig:
    """
    Configuration dataclass for transformer model hyperparameters.
    
    This centralizes all model configuration in a type-safe, serializable format.
    """
    # Model architecture
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 5000
    vocab_size: int = 50000
    
    # Training parameters
    dropout_prob: float = 0.1
    learning_rate: float = 0.0001
    batch_size: int = 32
    num_epochs: int = 100
    warmup_steps: int = 4000
    
    # Generation parameters
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    max_gen_length: int = 100
    
    # Regularization
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.1
    
    # Embedding settings
    embedding_model_name: str = "glove-wiki-gigaword-300"
    use_pretrained_embeddings: bool = True
    freeze_embeddings: bool = False
    
    # Training settings
    use_mixed_precision: bool = False
    accumulate_grad_batches: int = 1
    save_every_n_steps: int = 1000
    eval_every_n_steps: int = 500
    
    # Paths
    model_save_path: str = "checkpoints/"
    data_path: str = "data/"
    log_path: str = "logs/"
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self.validate()
    
    def validate(self):
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid.
        """
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")
        
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        
        if not 0.0 <= self.dropout_prob <= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1], got {self.dropout_prob}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, filepath: Union[str, Path]):
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save the configuration.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'TransformerConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to the configuration file.
            
        Returns:
            TransformerConfig: Loaded configuration.
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def update(self, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Parameters to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        
        self.validate()


class ModelCheckpoint:
    """
    Model checkpoint management for saving and loading transformer models.
    """
    
    def __init__(self, checkpoint_dir: Union[str, Path] = "checkpoints/"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, model, config: TransformerConfig, step: int, 
             loss: float, metrics: Optional[Dict[str, float]] = None):
        """
        Save model checkpoint.
        
        Args:
            model: Transformer model to save.
            config: Model configuration.
            step: Training step.
            loss: Current loss value.
            metrics: Additional metrics to save.
        """
        checkpoint_data = {
            'step': step,
            'loss': loss,
            'metrics': metrics or {},
            'config': config.to_dict(),
            'timestamp': time.time()
        }
        
        # Save model weights (simplified - in practice you'd save actual model state)
        model_weights = {}
        if hasattr(model, 'transformer_blocks'):
            model_weights['num_blocks'] = len(model.transformer_blocks)
            model_weights['output_projection'] = model.output_projection.tolist()
        
        checkpoint_data['model_weights'] = model_weights
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pkl"
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Save config separately for easy access
        config.save(self.checkpoint_dir / f"config_step_{step}.json")
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """
        Load the latest checkpoint.
        
        Returns:
            Dictionary containing checkpoint data or None if no checkpoints found.
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pkl"))
        
        if not checkpoints:
            return None
        
        # Sort by step number
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
        
        with open(latest_checkpoint, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        return checkpoint_data
    
    def load_step(self, step: int) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from specific step.
        
        Args:
            step: Training step to load.
            
        Returns:
            Dictionary containing checkpoint data or None if not found.
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pkl"
        
        if not checkpoint_path.exists():
            return None
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        return checkpoint_data
    
    def list_checkpoints(self) -> List[int]:
        """
        List all available checkpoint steps.
        
        Returns:
            List of step numbers for available checkpoints.
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pkl"))
        steps = [int(cp.stem.split('_')[-1]) for cp in checkpoints]
        return sorted(steps)


class ExperimentTracker:
    """
    Simple experiment tracking for monitoring training progress and hyperparameters.
    """
    
    def __init__(self, experiment_name: str, log_dir: Union[str, Path] = "logs/"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment.
            log_dir: Directory to save logs.
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"{experiment_name}.jsonl"
        self.metrics_history = []
        
    def log_metrics(self, step: int, metrics: Dict[str, float]):
        """
        Log metrics for a training step.
        
        Args:
            step: Training step.
            metrics: Dictionary of metric values.
        """
        log_entry = {
            'step': step,
            'timestamp': time.time(),
            'metrics': metrics
        }
        
        self.metrics_history.append(log_entry)
        
        # Append to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_config(self, config: TransformerConfig):
        """
        Log experiment configuration.
        
        Args:
            config: Model configuration.
        """
        config_entry = {
            'timestamp': time.time(),
            'config': config.to_dict(),
            'experiment_name': self.experiment_name
        }
        
        config_file = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_entry, f, indent=2)
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete metrics history.
        
        Returns:
            List of all logged metrics.
        """
        return self.metrics_history.copy()
    
    def get_best_metric(self, metric_name: str, mode: str = 'min') -> Optional[Dict[str, Any]]:
        """
        Get the best value for a specific metric.
        
        Args:
            metric_name: Name of the metric.
            mode: 'min' for lowest value, 'max' for highest value.
            
        Returns:
            Dictionary containing the best metric entry.
        """
        if not self.metrics_history:
            return None
        
        valid_entries = [entry for entry in self.metrics_history 
                        if metric_name in entry['metrics']]
        
        if not valid_entries:
            return None
        
        if mode == 'min':
            best_entry = min(valid_entries, key=lambda x: x['metrics'][metric_name])
        else:
            best_entry = max(valid_entries, key=lambda x: x['metrics'][metric_name])
        
        return best_entry


def create_default_config() -> TransformerConfig:
    """
    Create a default transformer configuration.
    
    Returns:
        TransformerConfig: Default configuration.
    """
    return TransformerConfig()


def create_small_config() -> TransformerConfig:
    """
    Create a small transformer configuration for testing/debugging.
    
    Returns:
        TransformerConfig: Small model configuration.
    """
    return TransformerConfig(
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        batch_size=16,
        max_seq_len=512
    )


def create_large_config() -> TransformerConfig:
    """
    Create a large transformer configuration.
    
    Returns:
        TransformerConfig: Large model configuration.
    """
    return TransformerConfig(
        d_model=1024,
        num_heads=16,
        num_layers=12,
        d_ff=4096,
        batch_size=8,
        max_seq_len=2048
    )