"""===============================================================================
EVALLAB STAGE 3/4: EXPERIMENT EXECUTION

Analyzes codebase, sets up isolated environment, runs experiments.
Outputs reproduced metrics (JSON) for Stage 4 (Result Evaluation).
===============================================================================
"""

import os
import platform
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
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
                result = subprocess.run(
                    [cmd, '--version'], capture_output=True, text=True)
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

    def _collect_outputs(self, working_dir: Path) -> dict:
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
                    self.logger.debug(f"Could not parse {output_file}: {e}")
        return outputs

    def __init__(self, config=None, paper_name=None):
        """
        Initialize experiment executor.

        Args:
            config: Configuration dict from config.yaml
            paper_name: Name of the paper for per-paper logging
        """
        self.config = config or {}
        self.logger = logger
        if paper_name:
            log_filename = f"agent_execution_{paper_name}.log"
            file_handler = logging.FileHandler(log_filename)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s'))
            self.logger.addHandler(file_handler)

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
            'evaluate*.py', 'test*.py'
        ]

        entry_points = []

        # Recursively search for entry points, excluding unwanted dirs
        exclude_dirs = {'venv', 'env', 'site-packages', '__pycache__'}
        for pattern in entry_patterns:
            for match in path.rglob(pattern):
                # Exclude files in unwanted directories
                if not any(part.startswith('.') or part in exclude_dirs for part in match.parts):
                    if match.is_file() and match not in entry_points:
                        entry_points.append(match)

        # Sort by likelihood (evaluate > test > main > run > train > experiment)
        priority_order = ['evaluate', 'test',
                          'main', 'run', 'train', 'experiment']

        def sort_key(p: Path):
            name = p.stem.lower()
            for i, prefix in enumerate(priority_order):
                if name.startswith(prefix):
                    return i
            return len(priority_order)

        entry_points.sort(key=sort_key)
        return entry_points

    def _extract_dependencies(self, path: Path, language: str) -> List[str]:
        """Extract project dependencies from requirements files and entry script imports."""
        dependencies = []

        # Auto-detect additional dependencies from entry script imports
        entry_points = self._find_entry_points(path, language)
        if entry_points:
            main_script = entry_points[0]
            try:
                with open(main_script, 'r', encoding='utf-8') as f:
                    code = f.read()
                    import_lines = [line for line in code.splitlines() if line.strip().startswith('import') or line.strip().startswith('from')]
                    for line in import_lines:
                        if 'numpy' in line and 'numpy' not in dependencies:
                            dependencies.append('numpy')
                        if 'matplotlib' in line and 'matplotlib' not in dependencies:
                            dependencies.append('matplotlib')
                        if 'torch' in line and 'torch' not in dependencies:
                            dependencies.append('torch')
                        if 'torchvision' in line and 'torchvision' not in dependencies:
                            dependencies.append('torchvision')
            except Exception as e:
                self.logger.warning(f"Error auto-detecting dependencies from entry script: {e}")

        if language == 'python':
            req_files = [
                'requirements.txt',
                'floyd_requirements.txt',
                'requirements-dev.txt',
                'requirements-prod.txt',
                'requirements_test.txt',
            ]
            found_req = False
            for req_name in req_files:
                req_file = path / req_name
                if req_file.exists():
                    found_req = True
                    self.logger.info(f"✓ Found requirements file: {req_file.name}")
                    try:
                        with open(req_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    dep = line.split('#')[0].strip()
                                    if dep and dep not in dependencies:
                                        dependencies.append(dep)
                    except Exception as e:
                        self.logger.warning(f"Error reading {req_file.name}: {e}")

            setup_file = path / 'setup.py'
            if setup_file.exists():
                try:
                    with open(setup_file, 'r') as f:
                        content = f.read()
                        if 'install_requires' in content:
                            self.logger.debug("Found install_requires in setup.py")
                except Exception as e:
                    self.logger.warning(f"Error reading setup.py: {e}")

            # If no requirements.txt, parse README for PyTorch
            if not found_req:
                readme_path = path / 'README.md'
                if readme_path.exists():
                    try:
                        with open(readme_path, 'r', encoding='utf-8') as f:
                            readme = f.read().lower()
                            if 'pytorch' in readme or 'torchvision' in readme:
                                self.logger.info("✓ README mentions PyTorch, adding torch and torchvision to dependencies")
                                if 'torch' not in dependencies:
                                    dependencies.append('torch')
                                if 'torchvision' not in dependencies:
                                    dependencies.append('torchvision')
                    except Exception as e:
                        self.logger.warning(f"Error reading README.md: {e}")

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
                results['warnings'].append(
                    f"Missing {requirements['description']}: {filename}")
                continue

            try:
                line_count = sum(1 for _ in open(
                    filepath, 'r', encoding='utf-8'))
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
                    logger.info(
                        f"✓ {filename}: {line_count} lines, {file_size/(1024*1024):.1f}MB")

            except Exception as e:
                results['warnings'].append(f"Error reading {filename}: {e}")
                results['file_stats'][filename] = {'error': str(e)}

        if results['warnings']:
            logger.warning(
                f"Data validation found {len(results['warnings'])} issues")
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
        venv_exists = venv_path.exists()
        requirements_path = codebase_path / 'requirements.txt'

        # Check if venv is already set up and dependencies installed
        venv_ready = False
        if venv_exists:
            python_executable, _, _ = _get_venv_paths(venv_path)
            # Check if all dependencies are installed
            if requirements_path.exists():
                with open(requirements_path) as f:
                    reqs = [line.strip() for line in f if line.strip()
                            and not line.startswith('#')]
                missing = []
                for dep in reqs:
                    result = subprocess.run(
                        [str(python_executable), '-m', 'pip', 'show', dep],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode != 0:
                        missing.append(dep)
                if not missing:
                    logger.info(
                        "✓ Reusing cached venv and dependencies (no install needed)")
                    venv_ready = True
                else:
                    logger.info(
                        f"Some dependencies missing in venv: {missing}")

        if not venv_exists:
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

        # Install dependencies if not already installed
        if dependencies and not venv_ready:
            python_executable, _, _ = _get_venv_paths(venv_path)
            logger.info(f"Installing {len(dependencies)} dependencies...")
            logger.info(
                "⏳ This may take a few minutes depending on package sizes...")
            logger.info(f"   (Timeout: 30 minutes)")

            for dep in dependencies:
                logger.info(f"→ Installing: {dep}")
                try:
                    result = subprocess.run(
                        [str(python_executable), '-m', 'pip', 'install', dep],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=1800  # 30 minute timeout
                    )
                    logger.info(f"✓ Installed: {dep}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"✗ Failed to install {dep}: {e.stderr}")
                    return False
                except subprocess.TimeoutExpired:
                    logger.error(
                        f"✗ Timeout installing {dep} after 30 minutes")
                    return False
            logger.info("✓ All dependencies installed successfully")
            return True

        logger.info(
            "✓ Environment setup complete (venv cached if previously installed)")
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

        self.logger.info(f"[run_experiment] Starting experiment: {config.script_path}")
        self.logger.info(f"[run_experiment] Command: {config.script_path}")
        self.logger.info(f"[run_experiment] Args: {config.args}")
        self.logger.info(f"[run_experiment] Working directory: {config.working_dir if config.working_dir else config.script_path.parent}")
        self.logger.info(f"[run_experiment] Environment variables: {config.env_vars}")

        start_time = time.time()
        from datetime import datetime
        start_time_str = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        self.logger.info(f"[run_experiment] Start time: {start_time_str}")

        # Ensure script_path is absolute
        script_path = config.script_path.resolve()

        # Use venv python if available
        venv_path = config.working_dir / 'venv'
        if venv_path.exists():
            python_executable, _, _ = _get_venv_paths(venv_path)
            python_cmd = str(python_executable)
        else:
            python_cmd = _get_python_executable()
        cmd = [python_cmd, str(script_path)] + config.args

        # Prepare environment variables
        env = os.environ.copy()
        env.update(config.env_vars)

        # Set working directory to script's parent
        working_dir = config.working_dir if config.working_dir else script_path.parent

        try:
            self.logger.info(f"[run_experiment] Running subprocess: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=str(working_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=config.timeout
            )

            duration = time.time() - start_time
            end_time_str = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            self.logger.info(f"[run_experiment] End time: {end_time_str}")
            self.logger.info(f"[run_experiment] Duration: {duration:.2f} seconds")

            # Try to parse any JSON output files
            outputs = self._collect_outputs(working_dir)

            # Log full stdout and stderr for debugging
            if result.returncode != 0:
                self.logger.error(f"Experiment failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

                # Auto-retry if dataset error detected
                if 'Dataset not found or corrupted' in result.stderr and 'download=False' in result.stderr:
                    self.logger.warning("[run_experiment] Detected missing dataset error. Retrying with download=True...")
                    # Try to patch args if possible
                    # If script supports --download, add it
                    if '--download' not in config.args:
                        patched_args = config.args + ['--download']
                        cmd2 = [python_cmd, str(script_path)] + patched_args
                        self.logger.info(f"[run_experiment] Retrying subprocess: {' '.join(cmd2)}")
                        result2 = subprocess.run(
                            cmd2,
                            cwd=str(working_dir),
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=config.timeout
                        )
                        duration2 = time.time() - start_time
                        self.logger.info(f"[run_experiment] Retry duration: {duration2:.2f} seconds")
                        outputs2 = self._collect_outputs(working_dir)
                        if result2.returncode != 0:
                            self.logger.error(f"Experiment retry failed.\nSTDOUT:\n{result2.stdout}\nSTDERR:\n{result2.stderr}")
                        else:
                            self.logger.info(f"[run_experiment] Experiment retry completed successfully.\nSTDOUT:\n{result2.stdout}")
                        return ExperimentResult(
                            success=(result2.returncode == 0),
                            stdout=result2.stdout,
                            stderr=result2.stderr,
                            return_code=result2.returncode,
                            duration=duration2,
                            outputs=outputs2
                        )
            else:
                self.logger.info(f"[run_experiment] Experiment completed successfully.\nSTDOUT:\n{result.stdout}")

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
            self.logger.error(f"Experiment timed out after {config.timeout} seconds")
            return ExperimentResult(
                success=False,
                stdout="",
                stderr=f"Timeout after {config.timeout} seconds",
                return_code=-1,
                duration=duration
            )
        except Exception as e:
            import traceback
            duration = time.time() - start_time
            tb = traceback.format_exc()
            self.logger.error(f"Experiment failed with error: {e}\n{tb}")
            return ExperimentResult(
                success=False,
                stdout="",
                stderr=tb,
                return_code=-1,
                duration=duration
            )
