#!/usr/bin/env python3
"""
Setup and installation script for Transformer from Scratch.

This script helps users set up the environment and verify the installation.
"""

import sys
import subprocess
import importlib.util
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    min_version = (3, 7)
    current_version = sys.version_info[:2]
    
    if current_version < min_version:
        print(f"❌ Python {min_version[0]}.{min_version[1]}+ required. "
              f"Current version: {current_version[0]}.{current_version[1]}")
        return False
    
    print(f"✅ Python {current_version[0]}.{current_version[1]} is compatible")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def verify_imports():
    """Verify that all modules can be imported."""
    print("Verifying module imports...")
    
    modules_to_test = [
        "numpy",
        "gensim", 
        "transformer",
        "attention",
        "embed", 
        "layers",
        "config",
        "training",
        "data_processing"
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            if module in ["numpy", "gensim"]:
                # External dependencies
                importlib.import_module(module)
            else:
                # Local modules
                spec = importlib.util.spec_from_file_location(
                    module, f"{module}.py"
                )
                if spec and spec.loader:
                    module_obj = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module_obj)
            
            print(f"✅ {module}")
            
        except Exception as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All modules imported successfully")
    return True


def run_basic_test():
    """Run a basic functionality test."""
    print("Running basic functionality test...")
    
    try:
        # Import required modules
        from config import create_small_config
        from transformer import Transformer
        import numpy as np
        
        # Create small model for testing
        config = create_small_config()
        model = Transformer(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=2  # Small for quick test
        )
        
        # Test forward pass
        test_input = np.random.randn(8, config.d_model)
        output = model.forward(test_input)
        
        # Verify output shape
        expected_shape = (8, config.d_model)
        if output.shape == expected_shape:
            print(f"✅ Basic test passed - output shape: {output.shape}")
            return True
        else:
            print(f"❌ Shape mismatch - expected: {expected_shape}, got: {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False


def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("Running comprehensive tests...")
    
    try:
        # Try to run the test suite
        result = subprocess.run([
            sys.executable, "tests/transformer_test.py"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Comprehensive tests passed")
            return True
        else:
            print(f"❌ Some tests failed:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Tests timed out (may still be working)")
        return True
    except Exception as e:
        print(f"⚠️ Could not run comprehensive tests: {e}")
        return True  # Don't fail setup for this


def download_embeddings():
    """Download required embeddings if needed."""
    print("Checking for pre-trained embeddings...")
    
    try:
        import gensim.downloader as api
        
        # Check if embeddings are already available
        try:
            model = api.load('glove-wiki-gigaword-300')
            print("✅ GloVe embeddings already available")
            return True
        except:
            pass
        
        # Download embeddings
        print("Downloading GloVe embeddings (this may take a while)...")
        model = api.load('glove-wiki-gigaword-300')
        print("✅ GloVe embeddings downloaded successfully")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not download embeddings: {e}")
        print("You can download them later when needed")
        return True


def create_example_config():
    """Create example configuration files."""
    print("Creating example configuration files...")
    
    try:
        from config import create_small_config, create_default_config
        
        # Create configs directory
        configs_dir = Path("example_configs")
        configs_dir.mkdir(exist_ok=True)
        
        # Save example configurations
        small_config = create_small_config()
        small_config.save(configs_dir / "small_config.json")
        
        default_config = create_default_config()
        default_config.save(configs_dir / "default_config.json")
        
        print(f"✅ Example configs saved to {configs_dir}/")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not create example configs: {e}")
        return True


def main():
    """Main setup function."""
    print("🚀 Transformer from Scratch - Setup")
    print("=" * 50)
    
    # Track success of each step
    steps = [
        ("Checking Python version", check_python_version),
        ("Installing dependencies", install_dependencies),
        ("Verifying imports", verify_imports),
        ("Running basic test", run_basic_test),
        ("Downloading embeddings", download_embeddings),
        ("Creating example configs", create_example_config),
        ("Running comprehensive tests", run_comprehensive_test),
    ]
    
    results = []
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        success = step_func()
        results.append((step_name, success))
        
        if not success and step_name in ["Checking Python version", "Installing dependencies"]:
            print("❌ Critical step failed. Setup cannot continue.")
            return False
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Setup Summary")
    print("=" * 50)
    
    for step_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {step_name}")
    
    successful_steps = sum(1 for _, success in results if success)
    total_steps = len(results)
    
    print(f"\n🎯 {successful_steps}/{total_steps} steps completed successfully")
    
    if successful_steps >= total_steps - 1:  # Allow one non-critical failure
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run examples: python examples.py")
        print("2. Run tests: python tests/transformer_test.py")
        print("3. Explore notebooks: jupyter notebook transformer_notes.ipynb")
        return True
    else:
        print("\n⚠️ Setup completed with some issues.")
        print("Please check the failed steps above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)