"""
Code Analyzer Module

Scans and analyzes codebases (GitHub repos or local uploads) to understand
experiment structure and identify runnable scripts.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import tempfile
import shutil
import logging


logger = logging.getLogger(__name__)


@dataclass
class CodebaseInfo:
    """Information about a codebase."""
    path: Path
    language: str
    entry_points: List[Path]
    dependencies: List[str]
    structure: Dict[str, any]
    readme_content: Optional[str] = None


class CodeAnalyzer:
    """Analyze codebases to understand structure and experiment setup."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.temp_dir = Path(tempfile.gettempdir()) / "paper_reproduction"
        self.temp_dir.mkdir(exist_ok=True)
    
    def clone_github_repo(self, github_url: str) -> Path:
        """
        Clone a GitHub repository to a temporary location.
        
        Args:
            github_url: URL of the GitHub repository
            
        Returns:
            Path to cloned repository
        """
        repo_name = github_url.rstrip('/').split('/')[-1]
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        
        clone_path = self.temp_dir / repo_name
        
        # Remove existing directory if present
        if clone_path.exists():
            shutil.rmtree(clone_path)
        
        try:
            logger.info(f"Cloning {github_url} to {clone_path}")
            subprocess.run(
                ['git', 'clone', github_url, str(clone_path)],
                check=True,
                capture_output=True,
                text=True
            )
            return clone_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e.stderr}")
            raise
    
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
        
        # Detect primary language
        language = self._detect_language(codebase_path)
        
        # Find entry points (main scripts, train scripts, etc.)
        entry_points = self._find_entry_points(codebase_path, language)
        
        # Extract dependencies
        dependencies = self._extract_dependencies(codebase_path, language)
        
        # Read README if available
        readme_content = self._read_readme(codebase_path)
        
        # Build directory structure
        structure = self._build_structure(codebase_path)
        
        return CodebaseInfo(
            path=codebase_path,
            language=language,
            entry_points=entry_points,
            dependencies=dependencies,
            structure=structure,
            readme_content=readme_content
        )
    
    def _detect_language(self, path: Path) -> str:
        """Detect the primary programming language of the codebase."""
        extensions = {}
        
        for file_path in path.rglob('*'):
            if file_path.is_file():
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
            'evaluate*.py'
        ]
        
        entry_points = []
        
        # Only search in the root directory, not in subdirs or venv
        for pattern in entry_patterns:
            matches = list(path.glob(pattern))
            # Filter out library files and venv
            for match in matches:
                # Skip if in venv, site-packages, or hidden directories
                if not any(part.startswith('.') or part in ['venv', 'env', 'site-packages', '__pycache__'] 
                          for part in match.parts):
                    entry_points.append(match)
        
        # Also look for scripts in common directories (only 1 level deep)
        for script_dir in ['scripts', 'experiments']:
            script_path = path / script_dir
            if script_path.exists() and script_path.is_dir():
                py_files = list(script_path.glob('*.py'))
                entry_points.extend([f for f in py_files if f.stem != '__init__'])
        
        return list(set(entry_points))  # Remove duplicates
    
    def _extract_dependencies(self, path: Path, language: str) -> List[str]:
        """Extract dependencies from requirement files."""
        dependencies = []
        
        if language == 'python':
            # Check requirements.txt
            req_file = path / 'requirements.txt'
            if req_file.exists():
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # Skip empty lines and comments
                        if not line or line.startswith('#'):
                            continue
                        # Remove inline comments
                        if '#' in line:
                            line = line.split('#')[0].strip()
                        if line:
                            dependencies.append(line)
            
            # Check setup.py or pyproject.toml
            # (Could be extended)
        
        return dependencies
    
    def _read_readme(self, path: Path) -> Optional[str]:
        """Read README file if it exists."""
        for readme_name in ['README.md', 'README.rst', 'README.txt', 'README']:
            readme_path = path / readme_name
            if readme_path.exists():
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    logger.warning(f"Could not read README: {e}")
        
        return None
    
    def _build_structure(self, path: Path, max_depth: int = 3) -> Dict:
        """Build a dictionary representation of the directory structure."""
        def _recurse(current_path: Path, current_depth: int = 0) -> Dict:
            if current_depth > max_depth:
                return {"_truncated": True}
            
            structure = {}
            
            try:
                for item in current_path.iterdir():
                    # Skip hidden files and common ignore patterns
                    if item.name.startswith('.') or item.name in ['__pycache__', 'node_modules', 'venv']:
                        continue
                    
                    if item.is_dir():
                        structure[item.name] = _recurse(item, current_depth + 1)
                    else:
                        structure[item.name] = "file"
            except PermissionError:
                structure["_error"] = "Permission denied"
            
            return structure
        
        return _recurse(path)
    
    def identify_experiment_scripts_with_llm(self, codebase_info: CodebaseInfo) -> List[Dict]:
        """
        Use LLM to identify which scripts run experiments based on paper context.
        
        Args:
            codebase_info: Analyzed codebase information
            
        Returns:
            List of experiment script information
        """
        # This will be implemented with LLM integration
        pass
