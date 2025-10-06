#!/usr/bin/env python3
"""
Production Deployment Script for Transformer Implementation

This script provides automated deployment capabilities including:
- Environment setup and validation
- Dependency installation
- Model testing and validation
- Performance benchmarking
- Production configuration
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeploymentManager:
    """Manages the deployment process for the transformer implementation."""
    
    def __init__(self, target_dir: str = "."):
        """Initialize deployment manager."""
        self.target_dir = Path(target_dir).resolve()
        self.required_files = [
            "embed.py",
            "attention.py", 
            "layers.py",
            "transformer.py",
            "config.py",
            "requirements.txt"
        ]
        self.optional_files = [
            "performance.py",
            "checkpoint.py",
            "visualization.py",
            "optimization.py",
            "validation.py",
            "test_framework.py"
        ]
    
    def validate_environment(self) -> bool:
        """Validate that the deployment environment is ready."""
        logger.info("Validating deployment environment...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major != 3 or python_version.minor < 7:
            logger.error(f"Python 3.7+ required, found {python_version.major}.{python_version.minor}")
            return False
        
        logger.info(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check required files
        missing_files = []
        for file_name in self.required_files:
            file_path = self.target_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            logger.error(f"❌ Missing required files: {missing_files}")
            return False
        
        logger.info("✅ All required files present")
        
        # Check optional files
        present_optional = []
        for file_name in self.optional_files:
            file_path = self.target_dir / file_name
            if file_path.exists():
                present_optional.append(file_name)
        
        logger.info(f"✅ Optional files present: {present_optional}")
        
        return True
    
    def install_dependencies(self, upgrade: bool = False) -> bool:
        """Install required dependencies."""
        logger.info("Installing dependencies...")
        
        requirements_file = self.target_dir / "requirements.txt"
        if not requirements_file.exists():
            logger.warning("No requirements.txt found, creating minimal requirements")
            self._create_minimal_requirements()
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            if upgrade:
                cmd.append("--upgrade")
            
            result = subprocess.run(
                cmd,
                cwd=self.target_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to install dependencies: {result.stderr}")
                return False
            
            logger.info("✅ Dependencies installed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Dependency installation timed out")
            return False
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run the test suite to validate the implementation."""
        logger.info("Running test suite...")
        
        test_file = self.target_dir / "test_framework.py"
        if not test_file.exists():
            logger.warning("No test framework found, running basic validation")
            return self._run_basic_validation()
        
        try:
            result = subprocess.run(
                [sys.executable, str(test_file)],
                cwd=self.target_dir,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode != 0:
                logger.error(f"Tests failed: {result.stderr}")
                logger.info(f"Test output: {result.stdout}")
                return False
            
            logger.info("✅ All tests passed")
            logger.info(f"Test summary: {result.stdout.split('Tests run:')[-1].split('\\n')[0] if 'Tests run:' in result.stdout else 'Completed'}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Test suite timed out")
            return False
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return False
    
    def run_benchmarks(self) -> Optional[Dict]:
        """Run performance benchmarks."""
        logger.info("Running performance benchmarks...")
        
        try:
            # Import and run benchmarks
            sys.path.insert(0, str(self.target_dir))
            
            from test_framework import BenchmarkSuite
            
            benchmark_suite = BenchmarkSuite()
            results = benchmark_suite.run_full_benchmark()
            
            logger.info("✅ Benchmarks completed")
            
            # Log summary
            for component, component_results in results.items():
                avg_throughput = sum(r['throughput'] for r in component_results) / len(component_results)
                logger.info(f"  {component}: Average throughput {avg_throughput:.2f} tokens/s")
            
            return results
            
        except ImportError as e:
            logger.warning(f"Could not import benchmark suite: {e}")
            return None
        except Exception as e:
            logger.error(f"Error running benchmarks: {e}")
            return None
    
    def create_production_config(self, config_type: str = "default") -> bool:
        """Create production configuration files."""
        logger.info(f"Creating {config_type} production configuration...")
        
        try:
            sys.path.insert(0, str(self.target_dir))
            from config import get_config
            
            config = get_config(config_type)
            
            # Update for production
            config.experiment.experiment_name = f"transformer_production_{config_type}"
            config.experiment.description = f"Production deployment with {config_type} configuration"
            config.experiment.log_to_file = True
            config.experiment.log_to_console = True
            
            # Save configuration
            config_file = self.target_dir / f"production_config_{config_type}.json"
            config.save_to_file(str(config_file))
            
            logger.info(f"✅ Production configuration saved to {config_file}")
            return True
            
        except ImportError as e:
            logger.warning(f"Could not import config module: {e}")
            return False
        except Exception as e:
            logger.error(f"Error creating production config: {e}")
            return False
    
    def validate_model_functionality(self) -> bool:
        """Validate core model functionality."""
        logger.info("Validating model functionality...")
        
        try:
            sys.path.insert(0, str(self.target_dir))
            
            # Test basic imports
            from embed import positional_encoding, add_positional_encoding
            from attention import SelfAttention, MultiHeadAttention
            from layers import LayerNormalization, TransformerBlock
            
            # Test basic functionality
            import numpy as np
            
            # Test positional encoding
            pos_enc = positional_encoding(100, 512)
            assert pos_enc.shape == (100, 512)
            
            # Test attention
            attention = SelfAttention(512)
            sample_input = np.random.randn(2, 10, 512).astype(np.float32)
            output = attention.forward(sample_input)
            assert output.shape == sample_input.shape
            
            # Test transformer block
            transformer = TransformerBlock(d_model=512, num_heads=8, d_ff=2048)
            output = transformer.forward(sample_input)
            assert output.shape == sample_input.shape
            
            logger.info("✅ Model functionality validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Model functionality validation failed: {e}")
            return False
    
    def create_deployment_report(self, benchmark_results: Optional[Dict] = None) -> str:
        """Create a deployment report."""
        logger.info("Creating deployment report...")
        
        report_content = [
            "# Transformer Deployment Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Environment Information",
            f"- Python Version: {sys.version}",
            f"- Platform: {sys.platform}",
            f"- Deployment Directory: {self.target_dir}",
            "",
            "## File Status",
        ]
        
        # Required files status
        report_content.append("### Required Files")
        for file_name in self.required_files:
            file_path = self.target_dir / file_name
            status = "✅ Present" if file_path.exists() else "❌ Missing"
            size = f" ({file_path.stat().st_size} bytes)" if file_path.exists() else ""
            report_content.append(f"- {file_name}: {status}{size}")
        
        # Optional files status  
        report_content.append("")
        report_content.append("### Optional Files")
        for file_name in self.optional_files:
            file_path = self.target_dir / file_name
            if file_path.exists():
                size = f" ({file_path.stat().st_size} bytes)"
                report_content.append(f"- {file_name}: ✅ Present{size}")
        
        # Benchmark results
        if benchmark_results:
            report_content.extend([
                "",
                "## Performance Benchmarks",
            ])
            
            for component, results in benchmark_results.items():
                report_content.append(f"### {component.upper()}")
                for result in results:
                    report_content.append(
                        f"- Batch: {result['batch_size']}, Seq: {result['seq_len']}, "
                        f"D_model: {result['d_model']} → {result['throughput']:.2f} tokens/s"
                    )
        
        report_content.extend([
            "",
            "## Deployment Status",
            "✅ Deployment completed successfully",
            "",
            "## Next Steps",
            "1. Review configuration files",
            "2. Test with your specific data",
            "3. Monitor performance in production",
            "4. Set up logging and monitoring",
        ])
        
        report_file = self.target_dir / "deployment_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_content))
        
        logger.info(f"✅ Deployment report saved to {report_file}")
        return str(report_file)
    
    def _create_minimal_requirements(self):
        """Create a minimal requirements.txt file."""
        minimal_requirements = [
            "numpy>=1.19.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "psutil>=5.7.0",
        ]
        
        requirements_file = self.target_dir / "requirements.txt"
        with open(requirements_file, 'w') as f:
            f.write('\n'.join(minimal_requirements))
        
        logger.info(f"Created minimal requirements.txt with: {minimal_requirements}")
    
    def _run_basic_validation(self) -> bool:
        """Run basic validation without full test framework."""
        try:
            return self.validate_model_functionality()
        except Exception as e:
            logger.error(f"Basic validation failed: {e}")
            return False
    
    def deploy(self, 
               config_type: str = "default",
               run_tests: bool = True,
               run_benchmarks: bool = True,
               upgrade_deps: bool = False) -> bool:
        """Run complete deployment process."""
        logger.info("🚀 Starting transformer deployment...")
        
        steps = [
            ("Environment Validation", self.validate_environment),
            ("Dependency Installation", lambda: self.install_dependencies(upgrade_deps)),
            ("Model Functionality", self.validate_model_functionality),
        ]
        
        if run_tests:
            steps.append(("Test Suite", self.run_tests))
        
        steps.append(("Production Config", lambda: self.create_production_config(config_type)))
        
        # Execute steps
        for step_name, step_func in steps:
            logger.info(f"Executing: {step_name}")
            if not step_func():
                logger.error(f"❌ Deployment failed at: {step_name}")
                return False
        
        # Run benchmarks (optional)
        benchmark_results = None
        if run_benchmarks:
            benchmark_results = self.run_benchmarks()
        
        # Create deployment report
        report_file = self.create_deployment_report(benchmark_results)
        
        logger.info("🎉 Deployment completed successfully!")
        logger.info(f"📄 Deployment report: {report_file}")
        
        return True


def main():
    """Main deployment script entry point."""
    parser = argparse.ArgumentParser(description="Deploy transformer implementation")
    
    parser.add_argument(
        "--target-dir", 
        default=".",
        help="Target deployment directory"
    )
    parser.add_argument(
        "--config",
        choices=["default", "small", "large"],
        default="default",
        help="Configuration preset to use"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running test suite"
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true", 
        help="Skip running benchmarks"
    )
    parser.add_argument(
        "--upgrade-deps",
        action="store_true",
        help="Upgrade dependencies to latest versions"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create deployment manager
    deployment_manager = DeploymentManager(args.target_dir)
    
    # Run deployment
    success = deployment_manager.deploy(
        config_type=args.config,
        run_tests=not args.skip_tests,
        run_benchmarks=not args.skip_benchmarks,
        upgrade_deps=args.upgrade_deps
    )
    
    if success:
        print("\n🎉 Deployment successful!")
        print(f"📁 Deployed to: {Path(args.target_dir).resolve()}")
        print("📄 Check deployment_report.md for details")
        sys.exit(0)
    else:
        print("\n❌ Deployment failed!")
        print("Check the logs above for error details")
        sys.exit(1)


if __name__ == "__main__":
    main()