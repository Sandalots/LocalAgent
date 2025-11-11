"""
Experiment Runner Module

Executes experiments from codebases based on paper methodology.
"""

import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import time


logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for running an experiment."""
    script_path: Path
    args: List[str]
    env_vars: Dict[str, str]
    working_dir: Path
    timeout: int = 3600  # 1 hour default


@dataclass
class ExperimentResult:
    """Results from running an experiment."""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    duration: float
    outputs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.outputs is None:
            self.outputs = {}


class ExperimentRunner:
    """Execute experiments and capture results."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def setup_environment(self, codebase_path: Path, dependencies: List[str]) -> bool:
        """
        Set up the environment for running experiments (install dependencies, etc.).
        
        Args:
            codebase_path: Path to the codebase
            dependencies: List of dependencies to install
            
        Returns:
            True if setup successful, False otherwise
        """
        logger.info(f"Setting up environment for {codebase_path}")
        
        # Check if virtual environment exists, create if not
        venv_path = codebase_path / 'venv'
        if not venv_path.exists():
            logger.info("Creating virtual environment...")
            try:
                subprocess.run(
                    ['python3', '-m', 'venv', str(venv_path)],
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to create virtual environment: {e}")
                return False
        
        # Install dependencies
        if dependencies:
            pip_executable = venv_path / 'bin' / 'pip'
            logger.info(f"Installing {len(dependencies)} dependencies...")
            
            try:
                subprocess.run(
                    [str(pip_executable), 'install'] + dependencies,
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info("Dependencies installed successfully")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install dependencies: {e.stderr}")
                return False
        
        return True
    
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a single experiment with the given configuration.
        
        Args:
            config: Experiment configuration
            
        Returns:
            ExperimentResult with outputs and status
        """
        logger.info(f"Running experiment: {config.script_path}")
        
        start_time = time.time()
        
        # Prepare command
        cmd = ['python3', str(config.script_path)] + config.args
        
        # Prepare environment variables
        env = os.environ.copy()
        env.update(config.env_vars)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(config.working_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=config.timeout
            )
            
            duration = time.time() - start_time
            
            # Try to extract outputs (look for JSON output, saved files, etc.)
            outputs = self._extract_outputs(config.working_dir, result.stdout)
            
            return ExperimentResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                duration=duration,
                outputs=outputs
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.error(f"Experiment timed out after {duration:.2f} seconds")
            
            return ExperimentResult(
                success=False,
                stdout="",
                stderr=f"Experiment timed out after {config.timeout} seconds",
                return_code=-1,
                duration=duration
            )
        
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error running experiment: {e}")
            
            return ExperimentResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                duration=duration
            )
    
    def _extract_outputs(self, working_dir: Path, stdout: str) -> Dict[str, Any]:
        """
        Extract experiment outputs from files or stdout.
        
        Args:
            working_dir: Directory where experiment ran
            stdout: Standard output from the experiment
            
        Returns:
            Dictionary of extracted outputs (metrics, artifacts, etc.)
        """
        outputs = {}
        
        # Look for common output files in working directory and subdirectories
        output_patterns = [
            'results.json',
            'metrics.json',
            'evaluation.json',
            'output.json',
            'complete_results.json'
        ]
        
        # Check root directory
        for pattern in output_patterns:
            output_file = working_dir / pattern
            if output_file.exists():
                try:
                    with open(output_file, 'r') as f:
                        outputs[pattern] = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not parse {pattern}: {e}")
        
        # Check common output subdirectories
        for output_subdir in ['outputs', 'results', 'outputs_all_methods', 'output']:
            subdir = working_dir / output_subdir
            if subdir.exists() and subdir.is_dir():
                for pattern in output_patterns:
                    output_file = subdir / pattern
                    if output_file.exists():
                        try:
                            with open(output_file, 'r') as f:
                                outputs[f"{output_subdir}/{pattern}"] = json.load(f)
                                logger.info(f"✓ Found results in {output_subdir}/{pattern}")
                        except Exception as e:
                            logger.warning(f"Could not parse {output_subdir}/{pattern}: {e}")
        
        # Try to extract metrics from stdout
        # Look for common patterns like "accuracy: 0.95", "f1-score: 0.88", etc.
        metrics = self._extract_metrics_from_text(stdout)
        if metrics:
            outputs['stdout_metrics'] = metrics
        
        return outputs
    
    def _extract_metrics_from_text(self, text: str) -> Dict[str, float]:
        """Extract numeric metrics from text output."""
        import re
        
        metrics = {}
        
        # Common metric patterns
        patterns = [
            r'(\w+(?:\s+\w+)?)\s*[:=]\s*([0-9.]+)',
            r'(\w+)\s+score\s*[:=]\s*([0-9.]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for metric_name, value in matches:
                try:
                    metrics[metric_name.strip()] = float(value)
                except ValueError:
                    pass
        
        return metrics
    
    def prepare_experiment_config_with_llm(self, paper_context: str, 
                                           codebase_info: Any) -> List[ExperimentConfig]:
        """
        Use LLM to determine how to run experiments based on paper context.
        
        Args:
            paper_context: Extracted methodology and experiments from paper
            codebase_info: Analyzed codebase information
            
        Returns:
            List of ExperimentConfig objects to run
        """
        # This will be implemented with LLM integration
        pass


import os
