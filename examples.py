"""
Example script showing how to use the agent programmatically.
"""

from pathlib import Path
from src.main import ReproductionAgent


def example_basic_usage():
    """Basic example: analyze a paper with GitHub URL."""
    
    # Initialize agent
    agent = ReproductionAgent()
    
    # Run reproduction workflow
    paper_path = Path("path/to/your/paper.pdf")
    
    results = agent.run(
        paper_path=paper_path,
        codebase_source=None  # Will use GitHub URL from paper
    )
    
    # Access results
    if 'error' not in results:
        print(f"Found {len(results['comparisons'])} metrics to compare")
        print(results['report'])


def example_local_codebase():
    """Example: use a local codebase."""
    
    agent = ReproductionAgent()
    
    results = agent.run(
        paper_path=Path("paper.pdf"),
        codebase_source="/path/to/local/codebase"
    )


def example_custom_config():
    """Example: use custom configuration."""
    
    agent = ReproductionAgent(
        config_path=Path("custom_config.yaml")
    )
    
    results = agent.run(
        paper_path=Path("paper.pdf"),
        codebase_source="https://github.com/user/repo"
    )


if __name__ == '__main__':
    # Run examples
    print("See example functions above for usage patterns")
