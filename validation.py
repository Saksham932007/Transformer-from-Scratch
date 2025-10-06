"""
Model validation utilities for transformer architectures.

This module provides comprehensive validation tools for checking model
correctness, numerical stability, and architectural integrity.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Container for validation results."""
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    severity: str = "info"  # "info", "warning", "error"


class ModelValidator:
    """
    Comprehensive validator for transformer model components.
    
    Performs various checks including numerical stability, gradient flow,
    and architectural consistency validation.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize the model validator.
        
        Args:
            tolerance (float): Numerical tolerance for validation checks.
        """
        self.tolerance = tolerance
        self.validation_results = []
    
    def validate_attention_weights(self, attention_weights: np.ndarray) -> ValidationResult:
        """
        Validate attention weight matrices.
        
        Args:
            attention_weights (np.ndarray): Attention weights to validate.
            
        Returns:
            ValidationResult: Validation result.
        """
        if attention_weights.ndim != 2:
            return ValidationResult(
                passed=False,
                message=f"Attention weights must be 2D, got {attention_weights.ndim}D",
                severity="error"
            )
        
        if not np.all(attention_weights >= 0):
            return ValidationResult(
                passed=False,
                message="Attention weights contain negative values",
                severity="error"
            )
        
        # Check if rows sum to approximately 1 (after softmax)
        row_sums = np.sum(attention_weights, axis=1)
        if not np.allclose(row_sums, 1.0, atol=self.tolerance):
            max_deviation = np.max(np.abs(row_sums - 1.0))
            return ValidationResult(
                passed=False,
                message=f"Attention weights don't sum to 1, max deviation: {max_deviation:.6f}",
                details={"row_sums": row_sums.tolist(), "max_deviation": max_deviation},
                severity="warning" if max_deviation < 0.01 else "error"
            )
        
        # Check for NaN or infinity
        if np.any(np.isnan(attention_weights)) or np.any(np.isinf(attention_weights)):
            return ValidationResult(
                passed=False,
                message="Attention weights contain NaN or infinity values",
                severity="error"
            )
        
        return ValidationResult(
            passed=True,
            message="Attention weights validation passed",
            details={"shape": attention_weights.shape, "range": [np.min(attention_weights), np.max(attention_weights)]}
        )
    
    def validate_embeddings(self, embeddings: np.ndarray, expected_dim: Optional[int] = None) -> ValidationResult:
        """
        Validate embedding matrices.
        
        Args:
            embeddings (np.ndarray): Embeddings to validate.
            expected_dim (int, optional): Expected embedding dimension.
            
        Returns:
            ValidationResult: Validation result.
        """
        if embeddings.ndim != 2:
            return ValidationResult(
                passed=False,
                message=f"Embeddings must be 2D, got {embeddings.ndim}D",
                severity="error"
            )
        
        if expected_dim and embeddings.shape[1] != expected_dim:
            return ValidationResult(
                passed=False,
                message=f"Expected embedding dimension {expected_dim}, got {embeddings.shape[1]}",
                severity="error"
            )
        
        # Check for NaN or infinity
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            return ValidationResult(
                passed=False,
                message="Embeddings contain NaN or infinity values",
                severity="error"
            )
        
        # Check embedding magnitude distribution
        norms = np.linalg.norm(embeddings, axis=1)
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        # Warn if embeddings have very different magnitudes
        if std_norm > mean_norm:
            return ValidationResult(
                passed=True,
                message="Embedding magnitudes vary significantly",
                details={"mean_norm": mean_norm, "std_norm": std_norm},
                severity="warning"
            )
        
        return ValidationResult(
            passed=True,
            message="Embeddings validation passed",
            details={
                "shape": embeddings.shape,
                "mean_norm": mean_norm,
                "std_norm": std_norm,
                "range": [np.min(embeddings), np.max(embeddings)]
            }
        )
    
    def validate_weight_initialization(self, weights: Dict[str, np.ndarray]) -> List[ValidationResult]:
        """
        Validate weight initialization across model components.
        
        Args:
            weights (Dict[str, np.ndarray]): Dictionary of weight matrices.
            
        Returns:
            List[ValidationResult]: List of validation results.
        """
        results = []
        
        for name, weight in weights.items():
            # Check for NaN or infinity
            if np.any(np.isnan(weight)) or np.any(np.isinf(weight)):
                results.append(ValidationResult(
                    passed=False,
                    message=f"Weight matrix '{name}' contains NaN or infinity values",
                    severity="error"
                ))
                continue
            
            # Check weight magnitude
            weight_std = np.std(weight)
            weight_mean = np.abs(np.mean(weight))
            
            # Weights should generally have small mean (close to zero) and reasonable std
            if weight_mean > 0.1:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Weight matrix '{name}' has large mean: {weight_mean:.6f}",
                    details={"mean": weight_mean, "std": weight_std},
                    severity="warning"
                ))
            
            # Check for degenerate initialization
            if weight_std < 1e-8:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Weight matrix '{name}' has very small variance: {weight_std:.8f}",
                    details={"std": weight_std},
                    severity="error"
                ))
            elif weight_std > 10.0:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Weight matrix '{name}' has large variance: {weight_std:.6f}",
                    details={"std": weight_std},
                    severity="warning"
                ))
            else:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Weight matrix '{name}' initialization looks good",
                    details={"mean": weight_mean, "std": weight_std}
                ))
        
        return results
    
    def validate_gradient_flow(self, gradients: Dict[str, np.ndarray]) -> List[ValidationResult]:
        """
        Validate gradient flow through the model.
        
        Args:
            gradients (Dict[str, np.ndarray]): Dictionary of gradients.
            
        Returns:
            List[ValidationResult]: List of validation results.
        """
        results = []
        
        for name, grad in gradients.items():
            # Check for NaN or infinity
            if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                results.append(ValidationResult(
                    passed=False,
                    message=f"Gradient for '{name}' contains NaN or infinity values",
                    severity="error"
                ))
                continue
            
            # Check gradient magnitude
            grad_norm = np.linalg.norm(grad)
            grad_mean = np.abs(np.mean(grad))
            
            # Check for vanishing gradients
            if grad_norm < 1e-8:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Gradient for '{name}' is vanishing: norm={grad_norm:.8f}",
                    details={"grad_norm": grad_norm},
                    severity="warning"
                ))
            # Check for exploding gradients
            elif grad_norm > 100.0:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Gradient for '{name}' is exploding: norm={grad_norm:.6f}",
                    details={"grad_norm": grad_norm},
                    severity="error"
                ))
            else:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Gradient for '{name}' looks healthy",
                    details={"grad_norm": grad_norm, "grad_mean": grad_mean}
                ))
        
        return results
    
    def validate_model_architecture(self, model_config: Dict[str, Any]) -> List[ValidationResult]:
        """
        Validate model architecture configuration.
        
        Args:
            model_config (Dict[str, Any]): Model configuration dictionary.
            
        Returns:
            List[ValidationResult]: List of validation results.
        """
        results = []
        
        # Check required parameters
        required_params = ['embedding_dim', 'num_heads']
        for param in required_params:
            if param not in model_config:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Missing required parameter: {param}",
                    severity="error"
                ))
        
        if 'embedding_dim' in model_config and 'num_heads' in model_config:
            embedding_dim = model_config['embedding_dim']
            num_heads = model_config['num_heads']
            
            # Check divisibility
            if embedding_dim % num_heads != 0:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Embedding dimension ({embedding_dim}) must be divisible by number of heads ({num_heads})",
                    severity="error"
                ))
            
            # Check reasonable head dimension
            head_dim = embedding_dim // num_heads
            if head_dim < 16:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Head dimension ({head_dim}) is quite small, consider fewer heads",
                    details={"head_dim": head_dim},
                    severity="warning"
                ))
            elif head_dim > 256:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Head dimension ({head_dim}) is large, consider more heads",
                    details={"head_dim": head_dim},
                    severity="warning"
                ))
        
        # Check sequence length constraints
        if 'max_sequence_length' in model_config:
            max_seq_len = model_config['max_sequence_length']
            if max_seq_len <= 0:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Maximum sequence length must be positive, got {max_seq_len}",
                    severity="error"
                ))
        
        return results
    
    def validate_numerical_stability(self, values: np.ndarray, name: str = "values") -> ValidationResult:
        """
        Check numerical stability of computed values.
        
        Args:
            values (np.ndarray): Values to check.
            name (str): Name for reporting.
            
        Returns:
            ValidationResult: Validation result.
        """
        # Check for NaN
        if np.any(np.isnan(values)):
            nan_count = np.sum(np.isnan(values))
            return ValidationResult(
                passed=False,
                message=f"{name} contains {nan_count} NaN values",
                details={"nan_count": nan_count, "total_elements": values.size},
                severity="error"
            )
        
        # Check for infinity
        if np.any(np.isinf(values)):
            inf_count = np.sum(np.isinf(values))
            return ValidationResult(
                passed=False,
                message=f"{name} contains {inf_count} infinity values",
                details={"inf_count": inf_count, "total_elements": values.size},
                severity="error"
            )
        
        # Check for very large values that might cause overflow
        max_val = np.max(np.abs(values))
        if max_val > 1e6:
            return ValidationResult(
                passed=True,
                message=f"{name} contains very large values (max: {max_val:.2e})",
                details={"max_abs_value": max_val},
                severity="warning"
            )
        
        # Check for very small values that might cause underflow
        min_val = np.min(np.abs(values[values != 0])) if np.any(values != 0) else 0
        if min_val > 0 and min_val < 1e-10:
            return ValidationResult(
                passed=True,
                message=f"{name} contains very small values (min: {min_val:.2e})",
                details={"min_nonzero_value": min_val},
                severity="warning"
            )
        
        return ValidationResult(
            passed=True,
            message=f"{name} numerical stability check passed",
            details={"range": [np.min(values), np.max(values)]}
        )
    
    def run_comprehensive_validation(
        self,
        model_data: Dict[str, Any],
        model_config: Dict[str, Any]
    ) -> Dict[str, List[ValidationResult]]:
        """
        Run comprehensive validation on model data and configuration.
        
        Args:
            model_data (Dict[str, Any]): Model data including weights, activations, etc.
            model_config (Dict[str, Any]): Model configuration.
            
        Returns:
            Dict[str, List[ValidationResult]]: Categorized validation results.
        """
        results = {
            'architecture': [],
            'weights': [],
            'embeddings': [],
            'attention': [],
            'gradients': [],
            'numerical': []
        }
        
        # Validate architecture
        results['architecture'] = self.validate_model_architecture(model_config)
        
        # Validate weights if available
        if 'weights' in model_data:
            results['weights'] = self.validate_weight_initialization(model_data['weights'])
        
        # Validate embeddings if available
        if 'embeddings' in model_data:
            expected_dim = model_config.get('embedding_dim')
            results['embeddings'] = [self.validate_embeddings(model_data['embeddings'], expected_dim)]
        
        # Validate attention weights if available
        if 'attention_weights' in model_data:
            for i, attn_weights in enumerate(model_data['attention_weights']):
                result = self.validate_attention_weights(attn_weights)
                result.message = f"Head {i}: {result.message}"
                results['attention'].append(result)
        
        # Validate gradients if available
        if 'gradients' in model_data:
            results['gradients'] = self.validate_gradient_flow(model_data['gradients'])
        
        # Validate numerical stability for all arrays
        for key, value in model_data.items():
            if isinstance(value, np.ndarray):
                results['numerical'].append(self.validate_numerical_stability(value, key))
        
        return results
    
    def print_validation_report(self, results: Dict[str, List[ValidationResult]]):
        """
        Print a formatted validation report.
        
        Args:
            results (Dict[str, List[ValidationResult]]): Validation results.
        """
        print("MODEL VALIDATION REPORT")
        print("=" * 50)
        
        total_checks = sum(len(category_results) for category_results in results.values())
        passed_checks = sum(
            sum(1 for result in category_results if result.passed)
            for category_results in results.values()
        )
        
        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {total_checks - passed_checks}")
        print()
        
        for category, category_results in results.items():
            if not category_results:
                continue
                
            print(f"{category.upper()} VALIDATION:")
            print("-" * 30)
            
            for result in category_results:
                status = "✓" if result.passed else "✗"
                severity_marker = {
                    "info": "",
                    "warning": "⚠",
                    "error": "❌"
                }
                
                print(f"{status} {severity_marker.get(result.severity, '')} {result.message}")
                
                if result.details and result.severity != "info":
                    print(f"    Details: {result.details}")
            
            print()


def validate_transformer_model(model, model_config: Dict[str, Any]) -> Dict[str, List[ValidationResult]]:
    """
    Convenience function to validate a complete transformer model.
    
    Args:
        model: Transformer model instance.
        model_config (Dict[str, Any]): Model configuration.
        
    Returns:
        Dict[str, List[ValidationResult]]: Validation results.
    """
    validator = ModelValidator()
    
    # Extract model data
    model_data = {}
    
    # Try to extract weights
    try:
        if hasattr(model, 'multi_head_attention'):
            weights = {}
            if hasattr(model.multi_head_attention, 'attention_heads'):
                for i, head in enumerate(model.multi_head_attention.attention_heads):
                    weights[f'head_{i}_W_q'] = head.W_q
                    weights[f'head_{i}_W_k'] = head.W_k
                    weights[f'head_{i}_W_v'] = head.W_v
            if hasattr(model.multi_head_attention, 'W_o'):
                weights['W_o'] = model.multi_head_attention.W_o
            model_data['weights'] = weights
    except Exception as e:
        print(f"Warning: Could not extract model weights: {e}")
    
    return validator.run_comprehensive_validation(model_data, model_config)