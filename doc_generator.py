"""
Automatic Documentation Generator for Transformer Project

This module generates comprehensive documentation by analyzing the codebase,
extracting docstrings, and creating formatted documentation files.
"""

import os
import ast
import inspect
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import time


class CodeAnalyzer:
    """Analyzes Python code to extract documentation information."""
    
    def __init__(self):
        self.classes = []
        self.functions = []
        self.constants = []
        self.imports = []
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Python file and extract documentation information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            file_info = {
                'path': file_path,
                'docstring': ast.get_docstring(tree),
                'classes': [],
                'functions': [],
                'constants': [],
                'imports': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node)
                    file_info['classes'].append(class_info)
                
                elif isinstance(node, ast.FunctionDef):
                    # Only top-level functions
                    if isinstance(node.parent if hasattr(node, 'parent') else None, ast.Module):
                        func_info = self._analyze_function(node)
                        file_info['functions'].append(func_info)
                
                elif isinstance(node, ast.Assign):
                    # Constants (uppercase variables)
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            const_info = {
                                'name': target.id,
                                'line': node.lineno,
                                'value': self._get_node_value(node.value)
                            }
                            file_info['constants'].append(const_info)
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = self._analyze_import(node)
                    file_info['imports'].append(import_info)
            
            return file_info
            
        except Exception as e:
            return {
                'path': file_path,
                'error': str(e),
                'classes': [],
                'functions': [],
                'constants': [],
                'imports': []
            }
    
    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze a class definition."""
        methods = []
        attributes = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_function(item, is_method=True)
                methods.append(method_info)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attr_info = {
                            'name': target.id,
                            'line': item.lineno,
                            'value': self._get_node_value(item.value)
                        }
                        attributes.append(attr_info)
        
        return {
            'name': node.name,
            'line': node.lineno,
            'docstring': ast.get_docstring(node),
            'bases': [self._get_node_name(base) for base in node.bases],
            'methods': methods,
            'attributes': attributes
        }
    
    def _analyze_function(self, node: ast.FunctionDef, is_method: bool = False) -> Dict[str, Any]:
        """Analyze a function definition."""
        args = []
        for arg in node.args.args:
            arg_info = {'name': arg.arg}
            if arg.annotation:
                arg_info['type'] = self._get_node_name(arg.annotation)
            args.append(arg_info)
        
        return_type = None
        if node.returns:
            return_type = self._get_node_name(node.returns)
        
        return {
            'name': node.name,
            'line': node.lineno,
            'docstring': ast.get_docstring(node),
            'args': args,
            'return_type': return_type,
            'is_method': is_method,
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'decorators': [self._get_node_name(dec) for dec in node.decorator_list]
        }
    
    def _analyze_import(self, node) -> Dict[str, Any]:
        """Analyze an import statement."""
        if isinstance(node, ast.Import):
            return {
                'type': 'import',
                'modules': [alias.name for alias in node.names],
                'line': node.lineno
            }
        elif isinstance(node, ast.ImportFrom):
            return {
                'type': 'from_import',
                'module': node.module,
                'names': [alias.name for alias in node.names],
                'line': node.lineno
            }
    
    def _get_node_name(self, node) -> str:
        """Get the name of an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_node_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._get_node_name(node.value)}[{self._get_node_name(node.slice)}]"
        else:
            return str(type(node).__name__)
    
    def _get_node_value(self, node) -> str:
        """Get a string representation of a node's value."""
        try:
            if isinstance(node, ast.Constant):
                return repr(node.value)
            elif isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return f"{self._get_node_name(node.value)}.{node.attr}"
            else:
                return f"<{type(node).__name__}>"
        except:
            return "<unknown>"


class DocumentationGenerator:
    """Generates documentation for the transformer project."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.analyzer = CodeAnalyzer()
        self.files_info = {}
    
    def analyze_project(self) -> Dict[str, Any]:
        """Analyze all Python files in the project."""
        python_files = list(self.project_root.glob("*.py"))
        
        project_info = {
            'name': self.project_root.name,
            'path': str(self.project_root),
            'files': {},
            'summary': {
                'total_files': len(python_files),
                'total_classes': 0,
                'total_functions': 0,
                'total_lines': 0
            }
        }
        
        for py_file in python_files:
            if py_file.name.startswith('.'):
                continue
                
            file_info = self.analyzer.analyze_file(str(py_file))
            project_info['files'][py_file.name] = file_info
            
            # Update summary
            project_info['summary']['total_classes'] += len(file_info.get('classes', []))
            project_info['summary']['total_functions'] += len(file_info.get('functions', []))
            
            try:
                with open(py_file, 'r') as f:
                    lines = len(f.readlines())
                project_info['summary']['total_lines'] += lines
            except:
                pass
        
        self.files_info = project_info
        return project_info
    
    def generate_api_documentation(self) -> str:
        """Generate API documentation in markdown format."""
        if not self.files_info:
            self.analyze_project()
        
        doc_lines = [
            "# Transformer API Documentation",
            "",
            f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Project Overview",
            "",
            f"- **Project**: {self.files_info['name']}",
            f"- **Total Files**: {self.files_info['summary']['total_files']}",
            f"- **Total Classes**: {self.files_info['summary']['total_classes']}",
            f"- **Total Functions**: {self.files_info['summary']['total_functions']}",
            f"- **Total Lines of Code**: {self.files_info['summary']['total_lines']}",
            "",
            "## Table of Contents",
            ""
        ]
        
        # Generate table of contents
        for filename in sorted(self.files_info['files'].keys()):
            doc_lines.append(f"- [{filename}](#{filename.replace('.', '').replace('_', '-')})")
        
        doc_lines.extend(["", "---", ""])
        
        # Generate documentation for each file
        for filename in sorted(self.files_info['files'].keys()):
            file_info = self.files_info['files'][filename]
            doc_lines.extend(self._generate_file_documentation(filename, file_info))
        
        return '\n'.join(doc_lines)
    
    def _generate_file_documentation(self, filename: str, file_info: Dict[str, Any]) -> List[str]:
        """Generate documentation for a single file."""
        doc_lines = [
            f"## {filename}",
            ""
        ]
        
        # File docstring
        if file_info.get('docstring'):
            doc_lines.extend([
                file_info['docstring'],
                ""
            ])
        
        # Error handling
        if 'error' in file_info:
            doc_lines.extend([
                f"⚠️ **Error analyzing file**: {file_info['error']}",
                ""
            ])
            return doc_lines
        
        # Imports
        if file_info.get('imports'):
            doc_lines.extend(["### Imports", ""])
            for imp in file_info['imports']:
                if imp['type'] == 'import':
                    doc_lines.append(f"- `import {', '.join(imp['modules'])}`")
                else:
                    doc_lines.append(f"- `from {imp['module']} import {', '.join(imp['names'])}`")
            doc_lines.append("")
        
        # Constants
        if file_info.get('constants'):
            doc_lines.extend(["### Constants", ""])
            for const in file_info['constants']:
                doc_lines.append(f"- **{const['name']}**: `{const['value']}`")
            doc_lines.append("")
        
        # Classes
        if file_info.get('classes'):
            doc_lines.extend(["### Classes", ""])
            for cls in file_info['classes']:
                doc_lines.extend(self._generate_class_documentation(cls))
        
        # Functions
        if file_info.get('functions'):
            doc_lines.extend(["### Functions", ""])
            for func in file_info['functions']:
                doc_lines.extend(self._generate_function_documentation(func))
        
        doc_lines.extend(["---", ""])
        return doc_lines
    
    def _generate_class_documentation(self, cls_info: Dict[str, Any]) -> List[str]:
        """Generate documentation for a class."""
        doc_lines = [f"#### {cls_info['name']}", ""]
        
        # Inheritance
        if cls_info.get('bases'):
            bases_str = ', '.join(cls_info['bases'])
            doc_lines.append(f"**Inherits from**: {bases_str}")
            doc_lines.append("")
        
        # Class docstring
        if cls_info.get('docstring'):
            doc_lines.extend([
                cls_info['docstring'],
                ""
            ])
        
        # Attributes
        if cls_info.get('attributes'):
            doc_lines.extend(["**Attributes:**", ""])
            for attr in cls_info['attributes']:
                doc_lines.append(f"- `{attr['name']}`: {attr.get('value', 'N/A')}")
            doc_lines.append("")
        
        # Methods
        if cls_info.get('methods'):
            doc_lines.extend(["**Methods:**", ""])
            for method in cls_info['methods']:
                doc_lines.extend(self._generate_method_documentation(method))
        
        return doc_lines
    
    def _generate_function_documentation(self, func_info: Dict[str, Any]) -> List[str]:
        """Generate documentation for a function."""
        return self._generate_method_documentation(func_info)
    
    def _generate_method_documentation(self, func_info: Dict[str, Any]) -> List[str]:
        """Generate documentation for a method or function."""
        # Build signature
        args_str = ', '.join([
            f"{arg['name']}: {arg.get('type', 'Any')}" if 'type' in arg 
            else arg['name'] 
            for arg in func_info.get('args', [])
        ])
        
        return_type = func_info.get('return_type', 'None')
        signature = f"{func_info['name']}({args_str}) -> {return_type}"
        
        doc_lines = [
            f"##### `{signature}`",
            ""
        ]
        
        # Function docstring
        if func_info.get('docstring'):
            doc_lines.extend([
                func_info['docstring'],
                ""
            ])
        
        # Decorators
        if func_info.get('decorators'):
            decorators_str = ', '.join(func_info['decorators'])
            doc_lines.extend([
                f"**Decorators**: {decorators_str}",
                ""
            ])
        
        return doc_lines
    
    def generate_architecture_guide(self) -> str:
        """Generate an architecture guide based on the code structure."""
        if not self.files_info:
            self.analyze_project()
        
        guide_lines = [
            "# Transformer Architecture Guide",
            "",
            f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview",
            "",
            "This guide provides an overview of the transformer implementation architecture.",
            "",
            "## Core Components",
            ""
        ]
        
        # Analyze component relationships
        components = {
            'embed.py': 'Embedding and Positional Encoding',
            'attention.py': 'Attention Mechanisms',
            'layers.py': 'Transformer Layers and Blocks',
            'transformer.py': 'Complete Transformer Model',
            'config.py': 'Configuration Management',
            'performance.py': 'Performance Monitoring',
            'visualization.py': 'Visualization Tools',
            'optimization.py': 'Training Optimization',
            'checkpoint.py': 'Model Checkpointing',
            'validation.py': 'Model Validation'
        }
        
        for filename, description in components.items():
            if filename in self.files_info['files']:
                file_info = self.files_info['files'][filename]
                guide_lines.extend([
                    f"### {description} (`{filename}`)",
                    ""
                ])
                
                if file_info.get('docstring'):
                    guide_lines.extend([
                        file_info['docstring'],
                        ""
                    ])
                
                # List main classes
                if file_info.get('classes'):
                    guide_lines.append("**Main Classes:**")
                    for cls in file_info['classes']:
                        guide_lines.append(f"- `{cls['name']}`: {cls.get('docstring', 'No description').split('.')[0]}")
                    guide_lines.append("")
        
        guide_lines.extend([
            "## Data Flow",
            "",
            "1. **Input Processing**: Text tokens are converted to embeddings (`embed.py`)",
            "2. **Positional Encoding**: Position information is added to embeddings",
            "3. **Attention**: Multi-head self-attention processes the sequence (`attention.py`)",
            "4. **Layer Processing**: Multiple transformer layers apply attention and feed-forward networks (`layers.py`)",
            "5. **Output Generation**: Final layer produces output predictions",
            "",
            "## Configuration",
            "",
            "The system uses a comprehensive configuration management system (`config.py`) that handles:",
            "- Model hyperparameters",
            "- Training settings", 
            "- Experiment configuration",
            "",
            "## Performance Optimization",
            "",
            "Several optimization techniques are implemented:",
            "- Vectorized operations for efficiency",
            "- Memory-efficient attention computation",
            "- Gradient clipping and advanced optimizers",
            "- Performance monitoring and profiling tools",
        ])
        
        return '\n'.join(guide_lines)
    
    def generate_usage_examples(self) -> str:
        """Generate usage examples documentation."""
        examples = [
            "# Usage Examples",
            "",
            f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Basic Usage",
            "",
            "### 1. Simple Forward Pass",
            "",
            "```python",
            "import numpy as np",
            "from embed import add_positional_encoding",
            "from attention import SelfAttention",
            "from layers import TransformerBlock",
            "",
            "# Create sample input",
            "batch_size, seq_len, d_model = 2, 10, 512",
            "embeddings = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)",
            "",
            "# Add positional encoding",
            "pos_encoded = add_positional_encoding(embeddings)",
            "",
            "# Apply transformer block",
            "transformer = TransformerBlock(d_model=512, num_heads=8, d_ff=2048)",
            "output = transformer.forward(pos_encoded)",
            "```",
            "",
            "### 2. Using Configuration System",
            "",
            "```python",
            "from config import get_config",
            "",
            "# Get default configuration",
            "config = get_config('default')",
            "",
            "# Modify parameters",
            "config.model.d_model = 256",
            "config.training.learning_rate = 0.001",
            "",
            "# Save configuration",
            "config.save_to_file('my_config.json')",
            "```",
            "",
            "### 3. Performance Monitoring",
            "",
            "```python",
            "from performance import PerformanceProfiler",
            "",
            "profiler = PerformanceProfiler()",
            "",
            "with profiler:",
            "    # Your transformer operations here",
            "    output = transformer.forward(input_data)",
            "",
            "metrics = profiler.get_metrics()",
            "print(f'Execution time: {metrics[\"execution_time\"]:.4f}s')",
            "```",
            "",
            "### 4. Model Checkpointing",
            "",
            "```python",
            "from checkpoint import ModelCheckpoint",
            "",
            "checkpoint = ModelCheckpoint('models/')",
            "",
            "# Save model state",
            "model_state = {'weights': weights, 'config': config}",
            "checkpoint.save_checkpoint(model_state, epoch=10, loss=0.5)",
            "",
            "# Load latest checkpoint",
            "loaded_state = checkpoint.load_latest_checkpoint()",
            "```",
            "",
            "### 5. Visualization",
            "",
            "```python",
            "from visualization import AttentionVisualizer",
            "",
            "visualizer = AttentionVisualizer()",
            "",
            "# Visualize attention weights",
            "attention_weights = np.random.rand(8, 10, 10)  # num_heads, seq_len, seq_len",
            "tokens = ['hello', 'world', 'this', 'is', 'test']",
            "",
            "visualizer.plot_attention_heatmap(attention_weights[0], tokens, tokens)",
            "```",
            "",
            "## Advanced Usage",
            "",
            "### Custom Optimizer Configuration",
            "",
            "```python",
            "from optimization import get_optimizer, get_scheduler",
            "",
            "# Create Adam optimizer with custom settings",
            "optimizer = get_optimizer('adam', lr=0.001, weight_decay=0.01)",
            "",
            "# Create learning rate scheduler",
            "scheduler = get_scheduler('cosine', optimizer, warmup_steps=1000)",
            "```",
            "",
            "### Model Validation",
            "",
            "```python",
            "from validation import ModelValidator",
            "",
            "validator = ModelValidator()",
            "",
            "# Validate model architecture",
            "is_valid = validator.validate_transformer_architecture(",
            "    d_model=512, num_heads=8, num_layers=6",
            ")",
            "",
            "# Check numerical stability",
            "stability_report = validator.check_numerical_stability(model)",
            "```",
            "",
            "## Testing and Deployment",
            "",
            "### Running Tests",
            "",
            "```bash",
            "# Run full test suite",
            "python test_framework.py",
            "",
            "# Run specific test category",
            "python -m unittest test_framework.TestAttention",
            "```",
            "",
            "### Deployment",
            "",
            "```bash",
            "# Deploy with default configuration",
            "./deploy.py",
            "",
            "# Deploy with custom configuration",
            "./deploy.py --config large --target-dir /path/to/production",
            "",
            "# Skip tests and benchmarks for faster deployment",
            "./deploy.py --skip-tests --skip-benchmarks",
            "```",
        ]
        
        return '\n'.join(examples)
    
    def generate_complete_documentation(self, output_dir: str = "docs") -> List[str]:
        """Generate complete documentation package."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        generated_files = []
        
        # Generate API documentation
        api_doc = self.generate_api_documentation()
        api_file = output_path / "api_reference.md"
        with open(api_file, 'w', encoding='utf-8') as f:
            f.write(api_doc)
        generated_files.append(str(api_file))
        
        # Generate architecture guide
        arch_guide = self.generate_architecture_guide()
        arch_file = output_path / "architecture_guide.md"
        with open(arch_file, 'w', encoding='utf-8') as f:
            f.write(arch_guide)
        generated_files.append(str(arch_file))
        
        # Generate usage examples
        usage_examples = self.generate_usage_examples()
        usage_file = output_path / "usage_examples.md"
        with open(usage_file, 'w', encoding='utf-8') as f:
            f.write(usage_examples)
        generated_files.append(str(usage_file))
        
        # Generate project analysis JSON
        analysis_file = output_path / "project_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(self.files_info, f, indent=2)
        generated_files.append(str(analysis_file))
        
        return generated_files


def main():
    """Generate documentation for the current project."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate transformer project documentation")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output-dir", default="docs", help="Output directory for documentation")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze project, don't generate docs")
    
    args = parser.parse_args()
    
    # Create documentation generator
    doc_gen = DocumentationGenerator(args.project_root)
    
    print("🔍 Analyzing project...")
    project_info = doc_gen.analyze_project()
    
    print(f"📊 Analysis complete:")
    print(f"  - Files analyzed: {project_info['summary']['total_files']}")
    print(f"  - Classes found: {project_info['summary']['total_classes']}")
    print(f"  - Functions found: {project_info['summary']['total_functions']}")
    print(f"  - Lines of code: {project_info['summary']['total_lines']}")
    
    if args.analyze_only:
        return
    
    print(f"\n📝 Generating documentation in {args.output_dir}...")
    generated_files = doc_gen.generate_complete_documentation(args.output_dir)
    
    print("✅ Documentation generated:")
    for file_path in generated_files:
        print(f"  - {file_path}")
    
    print(f"\n🎉 Documentation complete! Check the '{args.output_dir}' directory.")


if __name__ == "__main__":
    main()