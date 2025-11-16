"""
Main Agent Orchestrator

Coordinates the entire workflow: paper parsing → code analysis → 
experiment execution → result evaluation.
Includes integrated Ollama LLM client functionality.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any
import yaml
from dotenv import load_dotenv
import os
import json
import re
import requests
from dataclasses import dataclass

from paper_parser import PaperParser, PaperContent
from repo_retriever import RepoRetriever
from experiment_executor import ExperimentExecutor, CodebaseInfo, ExperimentConfig
from result_evaluator import ResultEvaluator, BaselineMetrics


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReproductionAgent:
    """Main agent that coordinates paper reproduction workflow with integrated LLM."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the reproduction agent.
        
        Args:
            config_path: Optional path to config.yaml file
        """
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize LLM settings (integrated from llm_client)
        self.ollama_base_url = self.config['ollama']['base_url']
        self.ollama_model = self.config['ollama']['model']
        self.ollama_temperature = self.config['ollama']['temperature']
        self.ollama_timeout = self.config['ollama']['timeout']
        
        # Initialize components (using new 4-stage architecture)
        self.paper_parser = PaperParser()
        self.repo_retriever = RepoRetriever()
        self.experiment_executor = ExperimentExecutor(config=self.config)
        self.result_evaluator = ResultEvaluator(
            llm_client=self,  # Pass self as we have integrated LLM methods
            threshold=self.config['evaluation']['threshold']
        )
        
        # Create output directory
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)

    
    def _load_config(self, config_path: Optional[Path]) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config.yaml'
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._default_config()
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            'ollama': {
                'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                'model': os.getenv('OLLAMA_MODEL', 'llama3'),
                'temperature': float(os.getenv('OLLAMA_TEMPERATURE', '0.7')),
                'timeout': int(os.getenv('OLLAMA_TIMEOUT', '120'))
            },
            'logging': {'level': 'INFO'},
            'experiment': {'timeout': 3600, 'max_retries': 2, 'use_venv': True},
            'evaluation': {'threshold': 0.05},
            'paths': {
                'temp_dir': '/tmp/paper_reproduction',
                'output_dir': './outputs',
                'cache_dir': './.cache'
            }
        }
    
    # ============================================================================
    # INTEGRATED OLLAMA LLM CLIENT METHODS
    # ============================================================================
    
    def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 temperature: Optional[float] = None) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt for context
            temperature: Optional temperature override
            
        Returns:
            Generated text response
        """
        url = f"{self.ollama_base_url}/api/generate"
        
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.ollama_temperature
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(
                url, 
                json=payload, 
                timeout=self.ollama_timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], 
             temperature: Optional[float] = None) -> str:
        """
        Chat completion with conversation history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Optional temperature override
            
        Returns:
            Generated response
        """
        url = f"{self.ollama_base_url}/api/chat"
        
        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.ollama_temperature
            }
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.ollama_timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('message', {}).get('content', '').strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error in chat completion: {e}")
            raise
    
    def extract_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response and parse it as JSON.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Parsed JSON dictionary
        """
        response = self.generate(prompt, system_prompt, temperature=0.1)
        
        # Try to extract JSON from markdown code blocks first
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
        
        # Try to find JSON object in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            response = json_match.group(0)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Response was: {response[:500]}")
            return {}
    
    # ============================================================================
    # MAIN WORKFLOW METHODS
    # ============================================================================
    
    def run(self, paper_path: Optional[Path] = None, codebase_source: Optional[str] = None) -> dict:
        """
        Run the complete reproduction workflow.
        
        Args:
            paper_path: Path to the research paper PDF (if None, searches ./paper/ directory)
            codebase_source: Either GitHub URL or path to local codebase (if None, searches ./paper_source_code/)
            
        Returns:
            Dictionary with results and evaluation
        """
        logger.info("="*70)
        logger.info("Starting Research Paper Reproduction Agent")
        logger.info("="*70)
        
        # Auto-detect paper if not provided
        if paper_path is None:
            workspace_root = Path(__file__).parent.parent
            paper_dir = workspace_root / "paper"
            
            if paper_dir.exists():
                pdf_files = list(paper_dir.glob("*.pdf"))
                if pdf_files:
                    paper_path = pdf_files[0]
                    logger.info(f"✓ Auto-detected paper: {paper_path.name}")
                else:
                    logger.error("No PDF files found in ./paper/ directory")
                    return {'error': 'No paper PDF found'}
            else:
                logger.error("./paper/ directory not found")
                return {'error': 'No paper directory found'}
        
        # Step 1: Check Ollama availability
        if not self.is_available():
            logger.error("Ollama is not running or not accessible!")
            logger.error(f"Please start Ollama and ensure it's running at {self.ollama_base_url}")
            return {'error': 'Ollama not available'}
        
        logger.info(f"✓ Connected to Ollama (model: {self.ollama_model})")
        
        # Step 2: Parse the paper (Stage 1)
        print("\n" + "="*80)
        print("\033[94m┌" + "─"*78 + "┐\033[0m")
        print("\033[94m│\033[0m" + "\033[1;96m STAGE 1/4: PAPER PARSING".center(78) + "\033[94m│\033[0m")
        print("\033[94m├" + "─"*78 + "┤\033[0m")
        print("\033[94m│\033[0m" + " 📄 Extracting text, figures, and GitHub URLs from PDF".ljust(78) + "\033[94m│\033[0m")
        print("\033[94m│\033[0m" + " 🔍 Parsing paper structure and metadata".ljust(78) + "\033[94m│\033[0m")
        print("\033[94m└" + "─"*78 + "┘\033[0m")
        print("="*80 + "\n")
        
        paper_content = self.paper_parser.parse_pdf(paper_path)
        logger.info(f"✓ Extracted {len(paper_content.raw_text)} characters from paper")
        
        if paper_content.github_urls:
            logger.info(f"✓ Found {len(paper_content.github_urls)} GitHub URLs in paper")
            for url in paper_content.github_urls:
                logger.info(f"  - {url}")
        
        # Step 3: Extract key sections using LLM
        print("\n" + "="*80)
        print("\033[93m┌" + "─"*78 + "┐\033[0m")
        print("\033[93m│\033[0m" + "\033[1;93m STAGE 1.5/4: LLM SECTION EXTRACTION".center(78) + "\033[93m│\033[0m")
        print("\033[93m├" + "─"*78 + "┤\033[0m")
        print("\033[93m│\033[0m" + " 🤖 Using Ollama LLM to extract key sections".ljust(78) + "\033[93m│\033[0m")
        print("\033[93m│\033[0m" + " 📝 Identifying: Abstract, Methodology, Experiments, Results".ljust(78) + "\033[93m│\033[0m")
        print("\033[93m└" + "─"*78 + "┘\033[0m")
        print("="*80 + "\n")
        
        sections = self._extract_paper_sections(paper_content.raw_text)
        paper_content.abstract = sections.get('abstract', '')
        paper_content.methodology = sections.get('methodology', '')
        paper_content.experiments = sections.get('experiments', '')
        paper_content.results = sections.get('results', '')
        
        logger.info(f"✓ Extracted abstract ({len(paper_content.abstract)} chars)")
        logger.info(f"✓ Extracted methodology ({len(paper_content.methodology)} chars)")
        logger.info(f"✓ Extracted experiments ({len(paper_content.experiments)} chars)")
        
        # Step 4: Retrieve codebase (Stage 2 - NEW UNIFIED MODULE)
        print("\n" + "="*80)
        print("\033[92m┌" + "─"*78 + "┐\033[0m")
        print("\033[92m│\033[0m" + "\033[1;92m STAGE 2/4: CODE RETRIEVAL".center(78) + "\033[92m│\033[0m")
        print("\033[92m├" + "─"*78 + "┤\033[0m")
        print("\033[92m│\033[0m" + " 🔎 Priority: User path → Local directory → GitHub URLs".ljust(78) + "\033[92m│\033[0m")
        print("\033[92m│\033[0m" + " 📦 Searching for experiment code and dependencies".ljust(78) + "\033[92m│\033[0m")
        print("\033[92m└" + "─"*78 + "┘\033[0m")
        print("="*80 + "\n")
        
        # Convert codebase_source to Path if it's a string
        local_path = Path(codebase_source) if codebase_source else None
        
        codebase_path = self.repo_retriever.retrieve_code(
            github_urls=paper_content.github_urls,
            local_path=local_path
        )
        
        if not codebase_path:
            logger.error("No codebase available!")
            return {'error': 'No codebase available'}
        
        logger.info(f"✓ Codebase retrieved at: {codebase_path}")
        
        # Step 5: Analyze codebase structure (Stage 3 - UNIFIED ANALYSIS & EXECUTION)
        print("\n" + "="*80)
        print("\033[95m┌" + "─"*78 + "┐\033[0m")
        print("\033[95m│\033[0m" + "\033[1;95m STAGE 3/4: EXPERIMENT EXECUTION".center(78) + "\033[95m│\033[0m")
        print("\033[95m├" + "─"*78 + "┤\033[0m")
        print("\033[95m│\033[0m" + " 🔬 Analyzing codebase structure and dependencies".ljust(78) + "\033[95m│\033[0m")
        print("\033[95m│\033[0m" + " ⚙️  Setting up environment and validating data".ljust(78) + "\033[95m│\033[0m")
        print("\033[95m│\033[0m" + " 🚀 Running experiments with reproducible seeds".ljust(78) + "\033[95m│\033[0m")
        print("\033[95m└" + "─"*78 + "┘\033[0m")
        print("="*80 + "\n")
        
        codebase_info = self.experiment_executor.analyze_codebase(codebase_path)
        logger.info(f"✓ Analyzed codebase (language: {codebase_info.language})")
        logger.info(f"✓ Found {len(codebase_info.entry_points)} potential entry points")
        logger.info(f"✓ Found {len(codebase_info.dependencies)} dependencies")
        
        # Validate data integrity before running experiments
        logger.info("\n[Stage 3.5/4] Validating data integrity...")
        validation_results = self.experiment_executor.validate_data_integrity(codebase_info.path)
        
        if not validation_results['valid']:
            logger.warning("⚠️  Data validation failed - experiments may not reproduce correctly")
        
        if validation_results['file_stats']:
            total_size = sum(s.get('size_mb', 0) for s in validation_results['file_stats'].values())
            logger.info(f"✓ Data validation complete - {len(validation_results['file_stats'])} files, {total_size:.1f}MB total")
        
        # Run experiments
        logger.info("\n[Stage 3.6/4] Running experiments...")
        experiment_results = self._run_experiments_unified(paper_content, codebase_info)
        
        if not experiment_results:
            logger.error("No experiments were run successfully")
            return {'error': 'Experiment execution failed'}
        
        logger.info(f"✓ Completed {len(experiment_results)} experiments")
        
        # Step 6: Evaluate results (Stage 4)
        print("\n" + "="*80)
        print("\033[96m┌" + "─"*78 + "┐\033[0m")
        print("\033[96m│\033[0m" + "\033[1;96m STAGE 4/4: RESULT EVALUATION".center(78) + "\033[96m│\033[0m")
        print("\033[96m├" + "─"*78 + "┤\033[0m")
        print("\033[96m│\033[0m" + " 📊 Comparing reproduced metrics to baseline results".ljust(78) + "\033[96m│\033[0m")
        print("\033[96m│\033[0m" + " 📈 Generating visualizations and performance reports".ljust(78) + "\033[96m│\033[0m")
        print("\033[96m│\033[0m" + " ✅ Calculating success rate and deviation metrics".ljust(78) + "\033[96m│\033[0m")
        print("\033[96m└" + "─"*78 + "┘\033[0m")
        print("="*80 + "\n")
        
        # Load ALL experiment results from all output directories
        experiment_sets = self.result_evaluator.load_all_experiment_results(codebase_info.path)
        
        if not experiment_sets:
            logger.error("No experiment result sets found")
            return {'error': 'No experiment results found'}
        
        logger.info(f"✓ Loaded {len(experiment_sets)} experiment sets")
        
        # Extract all metrics from all experiments
        all_reproduced_metrics = self.result_evaluator.extract_all_metrics_from_experiments(experiment_sets)
        logger.info(f"✓ Extracted {len(all_reproduced_metrics)} total metrics from all experiments")
        
        # Try to extract baseline from paper AND report.md files
        baseline = self.result_evaluator.extract_baseline_from_paper(
            paper_content.results or paper_content.raw_text,
            codebase_path=codebase_info.path  # Pass codebase path for report.md parsing
        )
        logger.info(f"✓ Extracted {len(baseline.metrics)} baseline metrics")
        logger.info(f"  Source: {baseline.source}")
        
        # Compare all reproduced metrics to baseline
        comparisons = self.result_evaluator.compare_results(baseline, all_reproduced_metrics)
        logger.info(f"✓ Generated {len(comparisons)} metric comparisons")
        
        # Generate comprehensive report
        report = self.result_evaluator.generate_report(comparisons)
        logger.info("\n" + report)
        
        # Generate summary statistics
        summary_stats = self.result_evaluator.generate_summary_statistics(comparisons)
        logger.info(summary_stats)
        
        # Get LLM analysis of differences
        analysis = ""
        if comparisons:
            logger.info("\n[LLM Analysis] Analyzing result differences...")
            analysis = self.result_evaluator.analyze_differences_with_llm(
                comparisons,
                paper_content.methodology + "\n" + paper_content.experiments
            )
            logger.info("\nLLM ANALYSIS:")
            logger.info(analysis)
        
        # Generate comprehensive conclusions and recommendations
        logger.info("\n[Generating Conclusions] Creating comprehensive analysis...")
        conclusions = self.result_evaluator.generate_comprehensive_conclusions(
            comparisons,
            experiment_sets,
            baseline,
            paper_content.methodology + "\n" + paper_content.experiments
        )
        logger.info(conclusions)
        
        # Generate visualizations
        logger.info("\n[Generating Visualizations] Creating plots and charts...")
        try:
            viz_dir = Path('outputs') / 'visualizations'
            paper_name = paper_path.stem  # Get filename without extension
            viz_files = self.result_evaluator.generate_visualizations(
                comparisons,
                output_dir=viz_dir,
                paper_name=paper_name
            )
            logger.info(f"✓ Generated {len(viz_files)} visualization files")
            logger.info(f"📊 View visualizations: {viz_dir / 'visualizations.html'}")
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
            import traceback
            traceback.print_exc()

        
        # Save results
        self._save_results(paper_path, comparisons, report, summary_stats, 
                          analysis, conclusions, experiment_sets)
        
        logger.info("\n" + "="*70)
        logger.info("Reproduction workflow completed!")
        logger.info("="*70)
        
        return {
            'paper_content': paper_content,
            'codebase_info': codebase_info,
            'experiment_results': experiment_results,
            'baseline_metrics': baseline,
            'comparisons': comparisons,
            'report': report,
            'conclusions': conclusions
        }
    
    def _extract_paper_sections(self, raw_text: str) -> dict:
        """Use LLM to extract key sections from paper."""
        system_prompt = """You are an expert at reading research papers.
Extract the following sections: abstract, methodology, experiments, and results.
Return ONLY valid JSON with these exact keys. Each value should be a detailed text string."""
        
        # Truncate text to fit in context window
        truncated_text = raw_text[:8000]
        
        user_prompt = f"""Extract the following sections from this research paper and return as JSON:
1. abstract - The paper's abstract
2. methodology - The approach/methodology section
3. experiments - The experimental setup and evaluation
4. results - The results and findings

Paper text:
{truncated_text}

Return ONLY a JSON object (no markdown, no explanation):
{{"abstract": "text here", "methodology": "text here", "experiments": "text here", "results": "text here"}}"""
        
        try:
            sections = self.extract_json(user_prompt, system_prompt)
            # Ensure all keys exist
            default_sections = {"abstract": "", "methodology": "", "experiments": "", "results": ""}
            default_sections.update(sections)
            return default_sections
        except Exception as e:
            logger.error(f"Failed to extract sections with LLM: {e}")
            logger.debug(f"Attempting simple text extraction as fallback...")
            # Fallback: try to find sections by headers
            return self._simple_section_extraction(raw_text)
    
    def _simple_section_extraction(self, text: str) -> dict:
        """Simple regex-based section extraction as fallback."""
        import re
        sections = {"abstract": "", "methodology": "", "experiments": "", "results": ""}
        
        # Try to find abstract
        abstract_match = re.search(r'(?:Abstract|ABSTRACT)[:\s]+(.*?)(?=\n\n|\n[A-Z])', text, re.DOTALL)
        if abstract_match:
            sections['abstract'] = abstract_match.group(1).strip()[:1000]
        
        return sections
    
    def _extract_metrics_from_nested_dict(self, data: dict, prefix: str = "") -> dict:
        """
        Recursively extract numeric metrics from nested dictionaries.
        
        Args:
            data: Nested dictionary (e.g., from complete_results.json)
            prefix: Prefix for metric names
            
        Returns:
            Flat dictionary of metric_name: value
        """
        metrics = {}
        
        for key, value in data.items():
            current_key = f"{prefix}/{key}" if prefix else key
            
            if isinstance(value, dict):
                # Check if this dict contains metrics
                if 'metrics' in value:
                    # Extract metrics from this level
                    metric_dict = value['metrics']
                    for metric_name, metric_value in metric_dict.items():
                        if isinstance(metric_value, dict):
                            # Handle metrics with multiple thresholds (e.g., recall@1, recall@5)
                            for threshold, val in metric_value.items():
                                if isinstance(val, (int, float)):
                                    metrics[f"{metric_name}@{threshold}"] = float(val)
                        elif isinstance(metric_value, (int, float)):
                            metrics[metric_name] = float(metric_value)
                else:
                    # Recurse into nested dicts
                    nested = self._extract_metrics_from_nested_dict(value, current_key)
                    metrics.update(nested)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                # Direct numeric value
                metrics[current_key] = float(value)
        
        return metrics
    
    
    def _run_experiments_unified(self, paper_content: PaperContent, 
                        codebase_info: CodebaseInfo) -> list:
        """Run experiments using the unified experiment executor (Stage 3)."""
        # Set up environment
        logger.info("Setting up experiment environment...")
        setup_success = self.experiment_executor.setup_environment(
            codebase_info.path,
            codebase_info.dependencies
        )
        
        if not setup_success:
            logger.warning("Environment setup had issues, proceeding anyway...")
        
        # Check for README instructions
        priority_scripts = []
        if codebase_info.readme_content:
            # Look for python commands in README
            python_cmds = re.findall(r'python[3]?\s+([\w_/\.]+\.py)(?:\s+(.*))?', 
                                    codebase_info.readme_content, re.IGNORECASE)
            for script_name, args in python_cmds:
                script_path = codebase_info.path / script_name
                if script_path.exists():
                    logger.info(f"Found priority script from README: {script_name}")
                    priority_scripts.append((script_path, args.strip().split() if args else []))
        
        results = []
        
        # Run priority scripts from README first
        for script_path, args in priority_scripts[:2]:  # Limit to 2
            logger.info(f"Running priority script: {script_path.name} {' '.join(args)}")
            
            config = ExperimentConfig(
                script_path=script_path,
                args=args,
                env_vars={},
                working_dir=codebase_info.path,
                timeout=self.config['experiment']['timeout']
            )
            
            result = self.experiment_executor.run_experiment(config)
            results.append(result)
            
            if result.success:
                logger.info(f"  ✓ Success (duration: {result.duration:.2f}s)")
            else:
                logger.warning(f"  ✗ Failed: {result.stderr[:200]}")
        
        # If no priority scripts, use entry points
        if not results:
            for entry_point in codebase_info.entry_points[:3]:  # Limit to 3
                logger.info(f"Running: {entry_point.name}")
                
                config = ExperimentConfig(
                    script_path=entry_point,
                    args=[],
                    env_vars={},
                    working_dir=codebase_info.path,
                    timeout=self.config['experiment']['timeout']
                )
                
                result = self.experiment_executor.run_experiment(config)
                results.append(result)
                
                if result.success:
                    logger.info(f"  ✓ Success (duration: {result.duration:.2f}s)")
                else:
                    logger.warning(f"  ✗ Failed: {result.stderr[:200]}")
        
        return [r for r in results if r.success]
    
    def _save_results(self, paper_path: Path, comparisons: list, 
                     report: str, summary_stats: str, analysis: str,
                     conclusions: str, experiment_sets: list):
        """Save results to output directory."""
        output_file = self.output_dir / f"{paper_path.stem}_results.txt"
        
        with open(output_file, 'w') as f:
            f.write(f"Results for: {paper_path.name}\n")
            f.write(f"Experiment sets analyzed: {len(experiment_sets)}\n")
            f.write(f"Total comparisons: {len(comparisons)}\n")
            f.write("\n" + report + "\n\n")
            f.write(summary_stats + "\n\n")
            if analysis:
                f.write("LLM ANALYSIS:\n")
                f.write(analysis + "\n\n")
            if conclusions:
                f.write(conclusions + "\n")
        
        logger.info(f"✓ Results saved to: {output_file}")


def main():
    """Main entry point for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Local Research Paper Reproduction Agent - Auto-detects paper and code from workspace'
    )
    parser.add_argument(
        'paper_path',
        nargs='?',  # Make optional
        type=Path,
        help='Path to research paper PDF (default: auto-detect from ./paper/)'
    )
    parser.add_argument(
        '--codebase',
        type=str,
        help='GitHub URL or local path to codebase (default: auto-detect from ./paper_source_code/)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to config.yaml file'
    )
    
    args = parser.parse_args()
    
    # Validate paper path if provided
    if args.paper_path and not args.paper_path.exists():
        print(f"Error: Paper file not found: {args.paper_path}")
        sys.exit(1)
    
    # Create and run agent
    agent = ReproductionAgent(config_path=args.config)
    results = agent.run(args.paper_path, args.codebase)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        sys.exit(1)


if __name__ == '__main__':
    main()
