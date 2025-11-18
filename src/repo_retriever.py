"""===============================================================================
EVALLAB STAGE 2/4: CODE RETRIEVAL

Locates experiment codebase via: user path → GitHub (paper-specific) → local directory (fallback).
Outputs local codebase path for Stage 3 (Experiment Execution).
===============================================================================
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)

class RepoRetriever:
    """Retrieve code repositories from various sources."""
    def __init__(self, workspace_root: Path = None):
        """
        Initialize repository retriever.

        Args:
            workspace_root: Root directory of the workspace (defaults to current dir)
        """
        self.workspace_root = workspace_root or Path.cwd()
        self.paper_source_dir = self.workspace_root / "paper_source_code"

    def retrieve_code(self, github_urls: List[str] = None,
                     local_path: Path = None) -> Optional[Path]:
        """
        Retrieve code from available sources with the following priority:
        1. User-provided local path (--code argument)
        2. GitHub URLs from the paper (paper-specific)
        3. Local paper_source_code directory (fallback/default)

        Args:
            github_urls: List of GitHub repository URLs found in paper
            local_path: Optional user-provided local codebase path

        Returns:
            Path to the retrieved codebase, or None if not found
        """
        # Priority 1: User-provided local path (explicit override)
        if local_path and local_path.exists():
            logger.info(f"✓ Using user-provided codebase: {local_path}")
            return local_path

        # Priority 2: Clone from GitHub (paper-specific code)
        if github_urls:
            logger.info(f"Found {len(github_urls)} GitHub URL(s) in paper")
            for url in github_urls:
                logger.info(f"  - {url}")

            cloned_path = self._clone_github_repo(github_urls[0])
            if cloned_path:
                logger.info(f"✓ Using paper-specific GitHub repository")
                return cloned_path

        # Priority 3: Check paper_source_code directory (fallback)
        local_code = self._find_local_code()
        if local_code:
            logger.info(f"✓ Using local codebase (fallback): {local_code}")
            return local_code

        logger.error("❌ No codebase found! Checked:")
        logger.error(f"  - User-provided path: {local_path}")
        logger.error(f"  - GitHub URLs: {github_urls}")
        logger.error(f"  - Local directory: {self.paper_source_dir}")
        return None

    def _find_local_code(self) -> Optional[Path]:
        """
        Find code in the local paper_source_code directory.
        Looks for supplementary_material/code or similar structures.

        Returns:
            Path to code directory if found
        """
        if not self.paper_source_dir.exists():
            return None

        # Check common structures
        candidates = [
            self.paper_source_dir / "supplementary_material" / "code",
            self.paper_source_dir / "supplementary_material",
            self.paper_source_dir / "code",
            self.paper_source_dir
        ]

        for candidate in candidates:
            if candidate.exists() and self._looks_like_code_dir(candidate):
                return candidate

        return None

    def _looks_like_code_dir(self, path: Path) -> bool:
        """
        Check if a directory looks like it contains code.

        Args:
            path: Directory to check

        Returns:
            True if directory appears to contain code
        """
        if not path.is_dir():
            return False

        # Check for common code indicators
        code_indicators = [
            '*.py',      # Python
            '*.js',      # JavaScript
            '*.java',    # Java
            '*.cpp',     # C++
            '*.c',       # C
            '*.go',      # Go
            '*.rs',      # Rust
            'requirements.txt',
            'package.json',
            'setup.py',
            'Makefile',
            'README.md'
        ]

        for indicator in code_indicators:
            if list(path.glob(indicator)) or list(path.rglob(indicator)):
                return True

        return False

    def _clone_github_repo(self, github_url: str) -> Optional[Path]:
        """
        Clone a GitHub repository to local workspace.

        Args:
            github_url: GitHub repository URL

        Returns:
            Path to cloned repository, or None if clone failed
        """
        try:
            # Extract repo name from URL
            repo_name = github_url.rstrip('/').split('/')[-1]
            if repo_name.endswith('.git'):
                repo_name = repo_name[:-4]

            clone_dir = self.workspace_root / "cloned_repos" / repo_name

            # Skip if already cloned
            if clone_dir.exists():
                logger.info(f"✓ Repository already cloned: {clone_dir}")
                return clone_dir

            # Create parent directory
            clone_dir.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Cloning repository from {github_url}...")
            result = subprocess.run(
                ['git', 'clone', github_url, str(clone_dir)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                logger.info(f"✓ Successfully cloned to: {clone_dir}")
                return clone_dir
            else:
                logger.error(f"Failed to clone repository: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Repository clone timed out after 5 minutes")
            return None
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            return None

    def get_codebase_info(self, code_path: Path) -> dict:
        """
        Get basic information about the retrieved codebase.

        Args:
            code_path: Path to codebase

        Returns:
            Dictionary with codebase metadata
        """
        info = {
            'path': code_path,
            'exists': code_path.exists(),
            'is_git_repo': (code_path / '.git').exists(),
            'has_python': bool(list(code_path.glob('*.py'))),
            'has_requirements': (code_path / 'requirements.txt').exists(),
            'has_setup': (code_path / 'setup.py').exists(),
        }

        # Count files by extension
        extensions = {}
        for file in code_path.rglob('*'):
            if file.is_file() and not any(p.startswith('.') for p in file.parts):
                ext = file.suffix
                extensions[ext] = extensions.get(ext, 0) + 1

        info['file_counts'] = extensions

        return info
