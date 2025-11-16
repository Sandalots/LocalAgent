"""
Experiment Executor Module - Stage 3

Combined module that:
1. Analyzes codebase structure and dependencies
2. Sets up execution environment  
3. Runs experiments and captures results
"""

import os
import sys
import platform
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import time

logger = logging.getLogger(__name__)


def _get_python_executable():
    """Get the appropriate Python executable for the current platform."""
    if platform.system() == 'Windows':
        # Try python first, then py launcher
        for cmd in ['python', 'py']:
            try:
                result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    return cmd
            except FileNotFoundError:
                continue
        return 'python'  # fallback
    else:
        # Unix/macOS - prefer python3
        return 'python3'


def _get_venv_paths(venv_path: Path):
    """Get platform-specific paths for venv executables.
    
    Returns:
        Tuple of (python_executable, pip_executable, scripts_dir)
    """
    if platform.system() == 'Windows':
        scripts_dir = venv_path / 'Scripts'
        python_exe = scripts_dir / 'python.exe'
        pip_exe = scripts_dir / 'pip.exe'
    else:
        scripts_dir = venv_path / 'bin'
        python_exe = scripts_dir / 'python'
        pip_exe = scripts_dir / 'pip'
    
    return python_exe, pip_exe, scripts_dir


@dataclass
class CodebaseInfo:
    """Information about a codebase."""
    path: Path
    language: str
    entry_points: List[Path]
    dependencies: List[str]
    readme_content: Optional[str] = None


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


class ExperimentExecutor:
    """Analyze codebase and execute experiments - Stage 3 of the agent."""
    
    def __init__(self, config=None):
        """
        Initialize experiment executor.
        
        Args:
            config: Configuration dict from config.yaml
        """
        self.config = config or {}
    
    # ============================================================================
    # PART 1: CODEBASE ANALYSIS
    # ============================================================================
    
    def analyze_codebase(self, codebase_path: Path) -> CodebaseInfo:
        """
        Analyze a codebase to extract structure and metadata.
        
        Args:
            codebase_path: Path to the codebase directory
            
        Returns:
            CodebaseInfo object with analysis results
        """
        if not codebase_path.exists():
            raise ValueError(f"Codebase path does not exist: {codebase_path}")
        
        logger.info(f"Analyzing codebase at: {codebase_path}")
        
        # Detect primary language
        language = self._detect_language(codebase_path)
        logger.info(f"  Detected language: {language}")
        
        # Find entry points (main scripts, train scripts, etc.)
        entry_points = self._find_entry_points(codebase_path, language)
        logger.info(f"  Found {len(entry_points)} entry point(s)")
        
        # Extract dependencies
        dependencies = self._extract_dependencies(codebase_path, language)
        logger.info(f"  Found {len(dependencies)} dependencies")
        
        # Read README if available
        readme_content = self._read_readme(codebase_path)
        
        return CodebaseInfo(
            path=codebase_path,
            language=language,
            entry_points=entry_points,
            dependencies=dependencies,
            readme_content=readme_content
        )
    
    def _detect_language(self, path: Path) -> str:
        """Detect the primary programming language of the codebase."""
        extensions = {}
        
        for file_path in path.rglob('*'):
            if file_path.is_file() and not any(p.startswith('.') for p in file_path.parts):
                ext = file_path.suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1
        
        # Map extensions to languages
        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.r': 'r',
            '.jl': 'julia'
        }
        
        for ext, lang in lang_map.items():
            if ext in extensions:
                return lang
        
        return 'unknown'
    
    def _find_entry_points(self, path: Path, language: str) -> List[Path]:
        """Find potential entry point scripts (main, train, experiment, etc.)."""
        entry_patterns = [
            'main*.py', 'train*.py', 'run*.py', 'experiment*.py',
            'evaluate*.py', '*_local_all*.py'  # For our specific paper
        ]
        
        entry_points = []
        
        # Search in root and immediate subdirectories
        for pattern in entry_patterns:
            # Root level
            matches = list(path.glob(pattern))
            # Filter out library files and venv
            for match in matches:
                if not any(part.startswith('.') or part in ['venv', 'env', 'site-packages', '__pycache__'] 
                          for part in match.parts):
                    if match.is_file() and match not in entry_points:
                        entry_points.append(match)
        
        # Sort by likelihood (main > run > train > experiment > evaluate)
        priority_order = ['main', 'run', 'train', 'experiment', 'evaluate']
        def sort_key(p: Path):
            name = p.stem.lower()
            for i, prefix in enumerate(priority_order):
                if name.startswith(prefix):
                    return i
            return len(priority_order)
        
        entry_points.sort(key=sort_key)
        return entry_points
    
    def _extract_dependencies(self, path: Path, language: str) -> List[str]:
        """Extract project dependencies from requirements files."""
        dependencies = []
        
        if language == 'python':
            # Check requirements.txt
            req_file = path / 'requirements.txt'
            if req_file.exists():
                try:
                    with open(req_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                # Remove comments and whitespace
                                dep = line.split('#')[0].strip()
                                if dep:
                                    dependencies.append(dep)
                except Exception as e:
                    logger.warning(f"Error reading requirements.txt: {e}")
            
            # Check setup.py for install_requires
            setup_file = path / 'setup.py'
            if setup_file.exists():
                try:
                    with open(setup_file, 'r') as f:
                        content = f.read()
                        # Simple regex-free extraction
                        if 'install_requires' in content:
                            logger.debug("Found install_requires in setup.py")
                except Exception as e:
                    logger.warning(f"Error reading setup.py: {e}")
        
        return dependencies
    
    def _read_readme(self, path: Path) -> Optional[str]:
        """Read README file if available."""
        readme_names = ['README.md', 'README.txt', 'README', 'readme.md']
        
        for name in readme_names:
            readme_path = path / name
            if readme_path.exists():
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    logger.warning(f"Error reading {name}: {e}")
        
        return None
    
    # ============================================================================
    # PART 2: ENVIRONMENT SETUP & VALIDATION
    # ============================================================================
    
    def validate_data_integrity(self, codebase_path: Path) -> Dict[str, Any]:
        """
        Validate dataset files before running experiments.
        
        Args:
            codebase_path: Path to codebase
            
        Returns:
            Dict with validation results
        """
        data_dir = codebase_path / "data"
        results = {
            'valid': True,
            'warnings': [],
            'file_stats': {}
        }
        
        if not data_dir.exists():
            results['valid'] = False
            results['warnings'].append(f"Data directory not found: {data_dir}")
            return results
        
        # Expected data files with minimum requirements
        expected_files = {
            'qa.jsonl': {'min_lines': 500, 'description': 'Q&A dataset'},
            'papers.jsonl': {'min_lines': 50, 'description': 'Papers corpus'},
            'qa-unlabeled.jsonl': {'min_lines': 1000, 'description': 'Unlabeled Q&A'},
            'qa-augmented-answers.jsonl': {'min_lines': 500, 'description': 'Augmented answers'}
        }
        
        for filename, requirements in expected_files.items():
            filepath = data_dir / filename
            
            if not filepath.exists():
                results['warnings'].append(f"Missing {requirements['description']}: {filename}")
                continue
            
            try:
                line_count = sum(1 for _ in open(filepath, 'r', encoding='utf-8'))
                file_size = filepath.stat().st_size
                
                results['file_stats'][filename] = {
                    'lines': line_count,
                    'size_mb': file_size / (1024 * 1024),
                    'exists': True
                }
                
                if line_count < requirements['min_lines']:
                    results['warnings'].append(
                        f"⚠️  {filename} has only {line_count} lines "
                        f"(expected >{requirements['min_lines']})"
                    )
                else:
                    logger.info(f"✓ {filename}: {line_count} lines, {file_size/(1024*1024):.1f}MB")
                    
            except Exception as e:
                results['warnings'].append(f"Error reading {filename}: {e}")
                results['file_stats'][filename] = {'error': str(e)}
        
        if results['warnings']:
            logger.warning(f"Data validation found {len(results['warnings'])} issues")
        else:
            logger.info("✓ Data validation passed")
        
        return results
    
    def setup_environment(self, codebase_path: Path, dependencies: List[str]) -> bool:
        """
        Set up the environment for running experiments.
        
        Args:
            codebase_path: Path to the codebase
            dependencies: List of dependencies to install
            
        Returns:
            True if setup successful
        """
        logger.info(f"Setting up environment for {codebase_path}")
        
        # Check if virtual environment exists
        venv_path = codebase_path / 'venv'
        if not venv_path.exists():
            logger.info("Creating virtual environment...")
            try:
                python_cmd = _get_python_executable()
                subprocess.run(
                    [python_cmd, '-m', 'venv', str(venv_path)],
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to create virtual environment: {e}")
                return False
        
        # Install dependencies
        if dependencies:
            _, pip_executable, _ = _get_venv_paths(venv_path)
            logger.info(f"Installing {len(dependencies)} dependencies...")
            logger.info("⏳ This may take a few minutes depending on package sizes...")
            logger.info(f"   (Timeout: 10 minutes)")
            
            try:
                subprocess.run(
                    [str(pip_executable), 'install'] + dependencies,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )
                logger.info("✓ Dependencies installed successfully")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install dependencies: {e.stderr}")
                return False
            except subprocess.TimeoutExpired:
                logger.error("Dependency installation timed out after 10 minutes")
                return False
        
        return True
    
    # ============================================================================
    # PART 3: EXPERIMENT EXECUTION
    # ============================================================================
    
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
        
        # Prepare command with platform-specific Python
        python_cmd = _get_python_executable()
        cmd = [python_cmd, str(config.script_path)] + config.args
        
        # Prepare environment variables
        env = os.environ.copy()
        env.update(config.env_vars)
        
        # Set random seeds for reproducibility
        if 'random_seed' in self.config.get('experiments', {}):
            seed = self.config['experiments']['random_seed']
            if self.config['experiments'].get('set_environment_seeds', True):
                env['PYTHONHASHSEED'] = str(seed)
                env['RANDOM_SEED'] = str(seed)
                logger.debug(f"Set PYTHONHASHSEED and RANDOM_SEED to {seed}")
        
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
            
            # Try to parse any JSON output files
            outputs = self._collect_outputs(config.working_dir)
            
            return ExperimentResult(
                success=(result.returncode == 0),
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                duration=duration,
                outputs=outputs
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.error(f"Experiment timed out after {config.timeout} seconds")
            return ExperimentResult(
                success=False,
                stdout="",
                stderr=f"Timeout after {config.timeout} seconds",
                return_code=-1,
                duration=duration
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Experiment failed with error: {e}")
            return ExperimentResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                duration=duration
            )
    
    def _collect_outputs(self, working_dir: Path) -> Dict[str, Any]:
        """Collect output files from experiment execution."""
        outputs = {}
        
        # Look for common output patterns
        output_patterns = ['output*.json', 'results*.json', '*.json']
        
        for pattern in output_patterns:
            for output_file in working_dir.glob(pattern):
                try:
                    with open(output_file, 'r') as f:
                        outputs[output_file.name] = json.load(f)
                except Exception as e:
                    logger.debug(f"Could not parse {output_file}: {e}")
        
        return outputs
