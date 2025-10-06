"""
Model Export and Serialization Utilities

This module provides utilities for exporting transformer models to various formats
including ONNX, TensorFlow, and custom serialization formats.
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)


class ModelExporter:
    """Export transformer models to various formats for deployment."""
    
    def __init__(self, model_dir: str = "exported_models"):
        """Initialize model exporter."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.supported_formats = {
            'numpy': self._export_numpy,
            'json': self._export_json,
            'pickle': self._export_pickle,
            'onnx': self._export_onnx,
            'tensorflow': self._export_tensorflow
        }
    
    def export_model(self, 
                    model_state: Dict[str, Any],
                    model_config: Dict[str, Any],
                    format_type: str = 'numpy',
                    model_name: str = 'transformer_model',
                    include_metadata: bool = True) -> str:
        """
        Export model to specified format.
        
        Args:
            model_state: Dictionary containing model weights and parameters
            model_config: Model configuration dictionary
            format_type: Export format ('numpy', 'json', 'pickle', 'onnx', 'tensorflow')
            model_name: Name for the exported model
            include_metadata: Whether to include metadata in export
            
        Returns:
            Path to exported model file
        """
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}. Supported: {list(self.supported_formats.keys())}")
        
        logger.info(f"Exporting model '{model_name}' to {format_type} format...")
        
        # Prepare export data
        export_data = {
            'model_state': model_state,
            'config': model_config
        }
        
        if include_metadata:
            export_data['metadata'] = self._create_metadata(model_name, format_type)
        
        # Call appropriate export function
        export_path = self.supported_formats[format_type](export_data, model_name)
        
        logger.info(f"Model exported successfully to: {export_path}")
        return export_path
    
    def _export_numpy(self, export_data: Dict[str, Any], model_name: str) -> str:
        """Export model using NumPy's native format."""
        export_dir = self.model_dir / f"{model_name}_numpy"
        export_dir.mkdir(exist_ok=True)
        
        # Save weights as .npy files
        weights_dir = export_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        
        model_state = export_data['model_state']
        for key, value in model_state.items():
            if isinstance(value, np.ndarray):
                weight_file = weights_dir / f"{key}.npy"
                np.save(weight_file, value)
            elif isinstance(value, dict):
                # Handle nested dictionaries (e.g., layer weights)
                layer_dir = weights_dir / key
                layer_dir.mkdir(exist_ok=True)
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        sub_weight_file = layer_dir / f"{sub_key}.npy"
                        np.save(sub_weight_file, sub_value)
        
        # Save configuration and metadata as JSON
        config_file = export_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'config': export_data['config'],
                'metadata': export_data.get('metadata', {})
            }, f, indent=2)
        
        # Create loading script
        self._create_loading_script(export_dir, 'numpy')
        
        return str(export_dir)
    
    def _export_json(self, export_data: Dict[str, Any], model_name: str) -> str:
        """Export model as JSON (weights converted to lists)."""
        export_file = self.model_dir / f"{model_name}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_data = self._convert_numpy_to_lists(export_data)
        
        with open(export_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        return str(export_file)
    
    def _export_pickle(self, export_data: Dict[str, Any], model_name: str) -> str:
        """Export model using pickle."""
        export_file = self.model_dir / f"{model_name}.pkl"
        
        with open(export_file, 'wb') as f:
            pickle.dump(export_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return str(export_file)
    
    def _export_onnx(self, export_data: Dict[str, Any], model_name: str) -> str:
        """Export model to ONNX format (placeholder implementation)."""
        logger.warning("ONNX export requires additional dependencies and model tracing")
        
        # Create placeholder ONNX export directory
        export_dir = self.model_dir / f"{model_name}_onnx"
        export_dir.mkdir(exist_ok=True)
        
        # Save model info for ONNX conversion
        info_file = export_dir / "onnx_conversion_info.json"
        with open(info_file, 'w') as f:
            json.dump({
                'model_name': model_name,
                'config': export_data['config'],
                'metadata': export_data.get('metadata', {}),
                'note': 'ONNX conversion requires implementing model tracing and ONNX dependencies'
            }, f, indent=2)
        
        return str(export_dir)
    
    def _export_tensorflow(self, export_data: Dict[str, Any], model_name: str) -> str:
        """Export model to TensorFlow format (placeholder implementation)."""
        logger.warning("TensorFlow export requires TensorFlow dependencies and model conversion")
        
        # Create placeholder TensorFlow export directory
        export_dir = self.model_dir / f"{model_name}_tensorflow"
        export_dir.mkdir(exist_ok=True)
        
        # Save model info for TensorFlow conversion
        info_file = export_dir / "tf_conversion_info.json"
        with open(info_file, 'w') as f:
            json.dump({
                'model_name': model_name,
                'config': export_data['config'],
                'metadata': export_data.get('metadata', {}),
                'note': 'TensorFlow conversion requires implementing TF model creation and dependencies'
            }, f, indent=2)
        
        return str(export_dir)
    
    def _create_metadata(self, model_name: str, format_type: str) -> Dict[str, Any]:
        """Create metadata for the exported model."""
        return {
            'model_name': model_name,
            'export_format': format_type,
            'export_timestamp': time.time(),
            'export_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'exporter_version': '1.0.0',
            'numpy_version': np.__version__,
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        }
    
    def _convert_numpy_to_lists(self, data: Any) -> Any:
        """Recursively convert numpy arrays to lists for JSON serialization."""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {key: self._convert_numpy_to_lists(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_numpy_to_lists(item) for item in data]
        elif isinstance(data, (np.int32, np.int64)):
            return int(data)
        elif isinstance(data, (np.float32, np.float64)):
            return float(data)
        else:
            return data
    
    def _create_loading_script(self, export_dir: Path, format_type: str):
        """Create a loading script for the exported model."""
        if format_type == 'numpy':
            script_content = '''#!/usr/bin/env python3
"""
Auto-generated model loading script for NumPy format export.
"""

import numpy as np
import json
from pathlib import Path


def load_model(model_dir):
    """Load model from NumPy format export."""
    model_dir = Path(model_dir)
    
    # Load configuration
    config_file = model_dir / "config.json"
    with open(config_file, 'r') as f:
        data = json.load(f)
    
    config = data['config']
    metadata = data.get('metadata', {})
    
    # Load weights
    weights_dir = model_dir / "weights"
    model_state = {}
    
    for npy_file in weights_dir.glob("**/*.npy"):
        # Get relative path and convert to key
        rel_path = npy_file.relative_to(weights_dir)
        key_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
        
        # Create nested dictionary structure
        current = model_state
        for part in key_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Load the numpy array
        current[key_parts[-1]] = np.load(npy_file)
    
    return {
        'model_state': model_state,
        'config': config,
        'metadata': metadata
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python load_model.py <model_directory>")
        sys.exit(1)
    
    model_data = load_model(sys.argv[1])
    print("Model loaded successfully!")
    print(f"Model name: {model_data['metadata'].get('model_name', 'Unknown')}")
    print(f"Export date: {model_data['metadata'].get('export_date', 'Unknown')}")
    print(f"Configuration: {model_data['config']}")
'''
            
            script_file = export_dir / "load_model.py"
            with open(script_file, 'w') as f:
                f.write(script_content)
            
            # Make script executable
            os.chmod(script_file, 0o755)
    
    def list_exported_models(self) -> List[Dict[str, Any]]:
        """List all exported models."""
        models = []
        
        for item in self.model_dir.iterdir():
            if item.is_file():
                # Single file exports (JSON, pickle)
                models.append({
                    'name': item.stem,
                    'format': item.suffix[1:] if item.suffix else 'unknown',
                    'path': str(item),
                    'size': item.stat().st_size,
                    'modified': item.stat().st_mtime
                })
            elif item.is_dir():
                # Directory exports (NumPy, ONNX, TensorFlow)
                format_type = 'unknown'
                if '_numpy' in item.name:
                    format_type = 'numpy'
                elif '_onnx' in item.name:
                    format_type = 'onnx'
                elif '_tensorflow' in item.name:
                    format_type = 'tensorflow'
                
                models.append({
                    'name': item.name,
                    'format': format_type,
                    'path': str(item),
                    'size': sum(f.stat().st_size for f in item.rglob('*') if f.is_file()),
                    'modified': item.stat().st_mtime
                })
        
        return sorted(models, key=lambda x: x['modified'], reverse=True)


class ModelLoader:
    """Load exported models from various formats."""
    
    def __init__(self):
        self.loaders = {
            'numpy': self._load_numpy,
            'json': self._load_json,
            'pickle': self._load_pickle
        }
    
    def load_model(self, model_path: str, format_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Load model from exported format.
        
        Args:
            model_path: Path to exported model
            format_type: Format type (auto-detected if None)
            
        Returns:
            Dictionary containing model data
        """
        model_path = Path(model_path)
        
        if format_type is None:
            format_type = self._detect_format(model_path)
        
        if format_type not in self.loaders:
            raise ValueError(f"Unsupported format: {format_type}")
        
        logger.info(f"Loading model from {model_path} (format: {format_type})")
        return self.loaders[format_type](model_path)
    
    def _detect_format(self, model_path: Path) -> str:
        """Auto-detect model format from path."""
        if model_path.is_file():
            suffix = model_path.suffix.lower()
            if suffix == '.json':
                return 'json'
            elif suffix == '.pkl':
                return 'pickle'
        elif model_path.is_dir():
            if '_numpy' in model_path.name or (model_path / 'weights').exists():
                return 'numpy'
            elif '_onnx' in model_path.name:
                return 'onnx'
            elif '_tensorflow' in model_path.name:
                return 'tensorflow'
        
        raise ValueError(f"Could not detect format for: {model_path}")
    
    def _load_numpy(self, model_path: Path) -> Dict[str, Any]:
        """Load model from NumPy format."""
        # Load configuration
        config_file = model_path / "config.json"
        with open(config_file, 'r') as f:
            data = json.load(f)
        
        # Load weights
        weights_dir = model_path / "weights"
        model_state = {}
        
        for npy_file in weights_dir.glob("**/*.npy"):
            rel_path = npy_file.relative_to(weights_dir)
            key_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
            
            current = model_state
            for part in key_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[key_parts[-1]] = np.load(npy_file)
        
        return {
            'model_state': model_state,
            'config': data['config'],
            'metadata': data.get('metadata', {})
        }
    
    def _load_json(self, model_path: Path) -> Dict[str, Any]:
        """Load model from JSON format."""
        with open(model_path, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        return self._convert_lists_to_numpy(data)
    
    def _load_pickle(self, model_path: Path) -> Dict[str, Any]:
        """Load model from pickle format."""
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def _convert_lists_to_numpy(self, data: Any) -> Any:
        """Recursively convert lists to numpy arrays."""
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (int, float, list)):
            try:
                return np.array(data, dtype=np.float32)
            except (ValueError, TypeError):
                return [self._convert_lists_to_numpy(item) for item in data]
        elif isinstance(data, dict):
            return {key: self._convert_lists_to_numpy(value) for key, value in data.items()}
        else:
            return data


def export_demo_model():
    """Create and export a demo model for testing."""
    # Create sample model state
    model_state = {
        'embedding_weights': np.random.randn(1000, 512).astype(np.float32),
        'attention_weights': {
            'W_q': np.random.randn(512, 512).astype(np.float32),
            'W_k': np.random.randn(512, 512).astype(np.float32),
            'W_v': np.random.randn(512, 512).astype(np.float32),
            'W_o': np.random.randn(512, 512).astype(np.float32)
        },
        'layer_norm_weights': {
            'gamma': np.ones(512, dtype=np.float32),
            'beta': np.zeros(512, dtype=np.float32)
        }
    }
    
    # Create sample configuration
    model_config = {
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 6,
        'vocab_size': 1000,
        'max_seq_length': 512
    }
    
    # Export in multiple formats
    exporter = ModelExporter()
    
    formats = ['numpy', 'json', 'pickle']
    exported_paths = {}
    
    for fmt in formats:
        try:
            path = exporter.export_model(
                model_state=model_state,
                model_config=model_config,
                format_type=fmt,
                model_name='demo_transformer'
            )
            exported_paths[fmt] = path
            print(f"✅ Exported to {fmt}: {path}")
        except Exception as e:
            print(f"❌ Failed to export to {fmt}: {e}")
    
    return exported_paths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model export utilities")
    parser.add_argument("--demo", action="store_true", help="Export demo model")
    parser.add_argument("--list", action="store_true", help="List exported models")
    parser.add_argument("--format", choices=['numpy', 'json', 'pickle'], default='numpy', help="Export format")
    
    args = parser.parse_args()
    
    if args.demo:
        print("Creating and exporting demo model...")
        export_demo_model()
    
    if args.list:
        exporter = ModelExporter()
        models = exporter.list_exported_models()
        
        print("\nExported Models:")
        print("-" * 50)
        for model in models:
            size_mb = model['size'] / (1024 * 1024)
            modified = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model['modified']))
            print(f"📦 {model['name']} ({model['format']}) - {size_mb:.2f} MB - {modified}")
    
    if not any([args.demo, args.list]):
        print("Use --demo to create demo model or --list to show exported models")