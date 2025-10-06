"""
Model checkpointing and state management utilities for transformer models.

This module provides functionality for saving, loading, and managing model states
during training and inference, including support for resuming training and 
model versioning.
"""

import numpy as np
import pickle
import json
import os
import hashlib
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import shutil


@dataclass
class ModelMetadata:
    """Metadata for saved model checkpoints."""
    model_name: str
    version: str
    save_timestamp: str
    epoch: int
    loss: float
    embedding_dim: int
    num_heads: int
    num_layers: int
    vocab_size: int
    max_sequence_length: int
    hyperparameters: Dict[str, Any]
    file_hash: str
    file_size_bytes: int


class ModelCheckpoint:
    """
    Comprehensive model checkpointing system for transformer models.
    
    Supports saving and loading model states, optimizer states, and training metadata
    with integrity verification and version management.
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir (str): Directory to store checkpoints.
        """
        self.checkpoint_dir = checkpoint_dir
        self.metadata_file = "checkpoint_metadata.json"
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_name: Optional[str] = None
    ) -> str:
        """
        Save a model checkpoint with metadata.
        
        Args:
            model_state (Dict[str, Any]): Model state dictionary.
            optimizer_state (Dict[str, Any], optional): Optimizer state.
            metadata (Dict[str, Any], optional): Additional metadata.
            checkpoint_name (str, optional): Custom checkpoint name.
            
        Returns:
            str: Path to the saved checkpoint.
            
        Raises:
            ValueError: If model_state is empty or invalid.
            IOError: If saving fails.
        """
        if not model_state:
            raise ValueError("Model state cannot be empty")
        
        # Generate checkpoint name if not provided
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_{timestamp}"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pkl")
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'metadata': metadata or {},
            'save_timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        try:
            # Save checkpoint
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Calculate file hash for integrity verification
            file_hash = self._calculate_file_hash(checkpoint_path)
            file_size = os.path.getsize(checkpoint_path)
            
            # Update metadata
            checkpoint_metadata = {
                'checkpoint_name': checkpoint_name,
                'file_path': checkpoint_path,
                'file_hash': file_hash,
                'file_size_bytes': file_size,
                'save_timestamp': checkpoint_data['save_timestamp'],
                'metadata': metadata or {}
            }
            
            self._update_metadata_registry(checkpoint_metadata)
            
            return checkpoint_path
            
        except Exception as e:
            # Clean up partial file if saving failed
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            raise IOError(f"Failed to save checkpoint: {str(e)}")
    
    def load_checkpoint(self, checkpoint_path: str, verify_integrity: bool = True) -> Dict[str, Any]:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
            verify_integrity (bool): Whether to verify file integrity.
            
        Returns:
            Dict[str, Any]: Loaded checkpoint data.
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            ValueError: If checkpoint is corrupted or invalid.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if verify_integrity:
            if not self._verify_checkpoint_integrity(checkpoint_path):
                raise ValueError(f"Checkpoint integrity verification failed: {checkpoint_path}")
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Validate checkpoint structure
            required_keys = ['model_state', 'save_timestamp', 'version']
            for key in required_keys:
                if key not in checkpoint_data:
                    raise ValueError(f"Invalid checkpoint format: missing {key}")
            
            return checkpoint_data
            
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {str(e)}")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with metadata.
        
        Returns:
            List[Dict[str, Any]]: List of checkpoint information.
        """
        metadata_path = os.path.join(self.checkpoint_dir, self.metadata_file)
        
        if not os.path.exists(metadata_path):
            return []
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata.get('checkpoints', [])
        except Exception:
            return []
    
    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """
        Delete a checkpoint and its metadata.
        
        Args:
            checkpoint_name (str): Name of the checkpoint to delete.
            
        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pkl")
        
        try:
            # Remove checkpoint file
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            
            # Remove from metadata registry
            self._remove_from_metadata_registry(checkpoint_name)
            
            return True
            
        except Exception:
            return False
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the path to the most recent checkpoint.
        
        Returns:
            Optional[str]: Path to latest checkpoint, None if no checkpoints exist.
        """
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return None
        
        # Sort by timestamp and return the most recent
        checkpoints.sort(key=lambda x: x['save_timestamp'], reverse=True)
        return checkpoints[0]['file_path']
    
    def cleanup_old_checkpoints(self, keep_latest: int = 5) -> int:
        """
        Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_latest (int): Number of latest checkpoints to keep.
            
        Returns:
            int: Number of checkpoints deleted.
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_latest:
            return 0
        
        # Sort by timestamp
        checkpoints.sort(key=lambda x: x['save_timestamp'], reverse=True)
        
        # Delete old checkpoints
        deleted_count = 0
        for checkpoint in checkpoints[keep_latest:]:
            if self.delete_checkpoint(checkpoint['checkpoint_name']):
                deleted_count += 1
        
        return deleted_count
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _verify_checkpoint_integrity(self, checkpoint_path: str) -> bool:
        """Verify checkpoint file integrity using stored hash."""
        try:
            current_hash = self._calculate_file_hash(checkpoint_path)
            
            # Get stored hash from metadata
            checkpoints = self.list_checkpoints()
            for checkpoint in checkpoints:
                if checkpoint['file_path'] == checkpoint_path:
                    return current_hash == checkpoint['file_hash']
            
            return True  # If no hash stored, assume valid
            
        except Exception:
            return False
    
    def _update_metadata_registry(self, checkpoint_metadata: Dict[str, Any]):
        """Update the checkpoint metadata registry."""
        metadata_path = os.path.join(self.checkpoint_dir, self.metadata_file)
        
        # Load existing metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {'checkpoints': []}
        
        # Add new checkpoint metadata
        metadata['checkpoints'].append(checkpoint_metadata)
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _remove_from_metadata_registry(self, checkpoint_name: str):
        """Remove checkpoint from metadata registry."""
        metadata_path = os.path.join(self.checkpoint_dir, self.metadata_file)
        
        if not os.path.exists(metadata_path):
            return
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Filter out the checkpoint
        metadata['checkpoints'] = [
            cp for cp in metadata['checkpoints'] 
            if cp['checkpoint_name'] != checkpoint_name
        ]
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


class ModelStateManager:
    """
    High-level model state management for transformer models.
    
    Provides utilities for extracting, saving, and restoring model states
    from transformer objects.
    """
    
    def __init__(self, checkpoint_manager: Optional[ModelCheckpoint] = None):
        """
        Initialize the state manager.
        
        Args:
            checkpoint_manager (ModelCheckpoint, optional): Checkpoint manager instance.
        """
        self.checkpoint_manager = checkpoint_manager or ModelCheckpoint()
    
    def extract_model_state(self, model) -> Dict[str, Any]:
        """
        Extract state from a transformer model.
        
        Args:
            model: Transformer model instance.
            
        Returns:
            Dict[str, Any]: Model state dictionary.
        """
        state = {}
        
        # Extract basic attributes
        if hasattr(model, 'embedding_dim'):
            state['embedding_dim'] = model.embedding_dim
        if hasattr(model, 'num_heads'):
            state['num_heads'] = model.num_heads
        if hasattr(model, 'num_layers'):
            state['num_layers'] = getattr(model, 'num_layers', 1)
        
        # Extract weight matrices
        state['weights'] = {}
        
        # Handle different model types
        if hasattr(model, 'multi_head_attention'):
            state['weights']['attention'] = self._extract_attention_weights(model.multi_head_attention)
        
        if hasattr(model, 'output_projection'):
            state['weights']['output_projection'] = model.output_projection.copy()
        
        if hasattr(model, 'layers') and isinstance(model.layers, list):
            state['weights']['layers'] = []
            for layer in model.layers:
                layer_weights = self._extract_layer_weights(layer)
                state['weights']['layers'].append(layer_weights)
        
        return state
    
    def restore_model_state(self, model, state: Dict[str, Any]) -> bool:
        """
        Restore state to a transformer model.
        
        Args:
            model: Transformer model instance.
            state (Dict[str, Any]): Model state dictionary.
            
        Returns:
            bool: True if restoration was successful.
        """
        try:
            weights = state.get('weights', {})
            
            # Restore attention weights
            if 'attention' in weights and hasattr(model, 'multi_head_attention'):
                self._restore_attention_weights(model.multi_head_attention, weights['attention'])
            
            # Restore output projection
            if 'output_projection' in weights and hasattr(model, 'output_projection'):
                model.output_projection = weights['output_projection'].copy()
            
            # Restore layer weights
            if 'layers' in weights and hasattr(model, 'layers'):
                for i, layer_weights in enumerate(weights['layers']):
                    if i < len(model.layers):
                        self._restore_layer_weights(model.layers[i], layer_weights)
            
            return True
            
        except Exception as e:
            print(f"Failed to restore model state: {e}")
            return False
    
    def save_model(self, model, save_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a complete model with metadata.
        
        Args:
            model: Transformer model instance.
            save_path (str): Path to save the model.
            metadata (Dict[str, Any], optional): Additional metadata.
            
        Returns:
            str: Path to the saved checkpoint.
        """
        model_state = self.extract_model_state(model)
        
        # Add model-specific metadata
        model_metadata = {
            'model_type': type(model).__name__,
            'parameters': self._count_parameters(model_state),
            'save_path': save_path
        }
        
        if metadata:
            model_metadata.update(metadata)
        
        return self.checkpoint_manager.save_checkpoint(
            model_state=model_state,
            metadata=model_metadata,
            checkpoint_name=os.path.basename(save_path)
        )
    
    def load_model(self, model, checkpoint_path: str) -> bool:
        """
        Load a model from checkpoint.
        
        Args:
            model: Transformer model instance to load into.
            checkpoint_path (str): Path to the checkpoint.
            
        Returns:
            bool: True if loading was successful.
        """
        try:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)
            model_state = checkpoint_data['model_state']
            
            return self.restore_model_state(model, model_state)
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def _extract_attention_weights(self, attention_module) -> Dict[str, Any]:
        """Extract weights from attention module."""
        weights = {}
        
        if hasattr(attention_module, 'attention_heads'):
            weights['heads'] = []
            for head in attention_module.attention_heads:
                head_weights = {
                    'W_q': head.W_q.copy(),
                    'W_k': head.W_k.copy(),
                    'W_v': head.W_v.copy()
                }
                weights['heads'].append(head_weights)
        
        if hasattr(attention_module, 'W_o'):
            weights['W_o'] = attention_module.W_o.copy()
        
        return weights
    
    def _restore_attention_weights(self, attention_module, weights: Dict[str, Any]):
        """Restore weights to attention module."""
        if 'heads' in weights and hasattr(attention_module, 'attention_heads'):
            for i, head_weights in enumerate(weights['heads']):
                if i < len(attention_module.attention_heads):
                    head = attention_module.attention_heads[i]
                    head.W_q = head_weights['W_q'].copy()
                    head.W_k = head_weights['W_k'].copy()
                    head.W_v = head_weights['W_v'].copy()
        
        if 'W_o' in weights and hasattr(attention_module, 'W_o'):
            attention_module.W_o = weights['W_o'].copy()
    
    def _extract_layer_weights(self, layer) -> Dict[str, Any]:
        """Extract weights from a transformer layer."""
        weights = {}
        
        # Extract layer normalization weights
        if hasattr(layer, 'norm1'):
            weights['norm1'] = {
                'gamma': layer.norm1.gamma.copy(),
                'beta': layer.norm1.beta.copy()
            }
        
        if hasattr(layer, 'norm2'):
            weights['norm2'] = {
                'gamma': layer.norm2.gamma.copy(),
                'beta': layer.norm2.beta.copy()
            }
        
        # Extract feedforward weights
        if hasattr(layer, 'ffn'):
            weights['ffn'] = {
                'W1': layer.ffn.W1.copy(),
                'b1': layer.ffn.b1.copy(),
                'W2': layer.ffn.W2.copy(),
                'b2': layer.ffn.b2.copy()
            }
        
        return weights
    
    def _restore_layer_weights(self, layer, weights: Dict[str, Any]):
        """Restore weights to a transformer layer."""
        if 'norm1' in weights and hasattr(layer, 'norm1'):
            layer.norm1.gamma = weights['norm1']['gamma'].copy()
            layer.norm1.beta = weights['norm1']['beta'].copy()
        
        if 'norm2' in weights and hasattr(layer, 'norm2'):
            layer.norm2.gamma = weights['norm2']['gamma'].copy()
            layer.norm2.beta = weights['norm2']['beta'].copy()
        
        if 'ffn' in weights and hasattr(layer, 'ffn'):
            layer.ffn.W1 = weights['ffn']['W1'].copy()
            layer.ffn.b1 = weights['ffn']['b1'].copy()
            layer.ffn.W2 = weights['ffn']['W2'].copy()
            layer.ffn.b2 = weights['ffn']['b2'].copy()
    
    def _count_parameters(self, model_state: Dict[str, Any]) -> int:
        """Count the total number of parameters in the model."""
        total_params = 0
        
        def count_array_params(obj):
            nonlocal total_params
            if isinstance(obj, np.ndarray):
                total_params += obj.size
            elif isinstance(obj, dict):
                for value in obj.values():
                    count_array_params(value)
            elif isinstance(obj, list):
                for item in obj:
                    count_array_params(item)
        
        count_array_params(model_state)
        return total_params