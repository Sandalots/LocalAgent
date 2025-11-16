"""EVALLab: Research Paper Reproduction Agent

A 4-stage autonomous AI agent for reproducing computational experiments from research papers.

STAGE ARCHITECTURE:
    Stage 1: Paper Parsing       - Extract text, figures, and GitHub URLs from PDF
    Stage 2: Code Retrieval      - Locate and retrieve experiment codebase
    Stage 3: Experiment Executor - Setup environment and run experiments
    Stage 4: Result Evaluator    - Compare results and generate reports
"""

__version__ = "1.0.0"
__author__ = "EVALLab Team"

# Import main components for easy access
from .pipeline import ReproductionAgent
from .paper_parser import PaperParser
from .repo_retriever import RepoRetriever
from .experiment_executor import ExperimentExecutor
from .result_evaluator import ResultEvaluator

__all__ = [
    'ReproductionAgent',
    'PaperParser',
    'RepoRetriever',
    'ExperimentExecutor',
    'ResultEvaluator',
]
