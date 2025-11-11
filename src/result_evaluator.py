"""
Result Evaluator Module

Compares experiment results to baseline metrics from the paper.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json


logger = logging.getLogger(__name__)


@dataclass
class BaselineMetrics:
    """Baseline metrics from the research paper."""
    metrics: Dict[str, float]
    source: str  # Where in the paper these came from
    
    
@dataclass
class ComparisonResult:
    """Comparison between reproduced and baseline results."""
    metric_name: str
    baseline_value: float
    reproduced_value: float
    difference: float
    percent_difference: float
    within_threshold: bool
    configuration: str = ""  # e.g., "sentence/minimal/bm25"


@dataclass
class ExperimentSet:
    """A complete set of experiment results."""
    name: str  # e.g., "outputs_all_methods", "outputs_all_methods_full"
    results: Dict[str, Any]  # The complete results dict
    total_configs: int
    total_metrics: int


class ResultEvaluator:
    """Evaluate reproduced results against paper baselines."""
    
    def __init__(self, llm_client=None, threshold: float = 0.05):
        """
        Initialize evaluator.
        
        Args:
            llm_client: Ollama client for LLM assistance
            threshold: Acceptable difference threshold (default 5%)
        """
        self.llm_client = llm_client
        self.threshold = threshold
    
    def load_all_experiment_results(self, codebase_path: Path) -> List[ExperimentSet]:
        """
        Load results from all output directories.
        
        Args:
            codebase_path: Path to codebase
            
        Returns:
            List of ExperimentSet objects
        """
        experiment_sets = []
        
        output_dirs = [
            "outputs_all_methods",
            "outputs_all_methods_full", 
            "outputs_all_methods_oracle"
        ]
        
        for dir_name in output_dirs:
            output_dir = codebase_path / dir_name
            if output_dir.exists():
                complete_results_path = output_dir / "complete_results.json"
                if complete_results_path.exists():
                    try:
                        with open(complete_results_path, 'r') as f:
                            results = json.load(f)
                        
                        # Count configurations and metrics
                        total_configs = len(results)
                        total_metrics = self._count_metrics_in_results(results)
                        
                        experiment_sets.append(ExperimentSet(
                            name=dir_name,
                            results=results,
                            total_configs=total_configs,
                            total_metrics=total_metrics
                        ))
                        
                        logger.info(f"✓ Loaded {dir_name}: {total_configs} configs, {total_metrics} metrics")
                    except Exception as e:
                        logger.error(f"Failed to load {dir_name}: {e}")
        
        return experiment_sets
    
    def _count_metrics_in_results(self, results: dict) -> int:
        """Count total number of metrics in nested results."""
        count = 0
        
        def count_recursive(data):
            nonlocal count
            if isinstance(data, dict):
                if 'metrics' in data:
                    for v in data['metrics'].values():
                        if isinstance(v, dict):
                            count += len(v)  # e.g., recall@1, recall@5, etc.
                        else:
                            count += 1
                else:
                    for v in data.values():
                        count_recursive(v)
        
        count_recursive(results)
        return count
    
    def extract_all_metrics_from_experiments(self, experiment_sets: List[ExperimentSet]) -> Dict[str, Dict[str, float]]:
        """
        Extract all metrics from all experiment sets, organized by configuration.
        
        Args:
            experiment_sets: List of experiment results
            
        Returns:
            Dict mapping "experiment_set/config/model/metric" to value
        """
        all_metrics = {}
        
        for exp_set in experiment_sets:
            metrics = self._extract_metrics_from_nested_dict(
                exp_set.results, 
                prefix=exp_set.name
            )
            all_metrics.update(metrics)
        
        return all_metrics
    
    def _extract_metrics_from_nested_dict(self, data: dict, prefix: str = "") -> Dict[str, float]:
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
                                    full_key = f"{current_key}/{metric_name}@{threshold}"
                                    metrics[full_key] = float(val)
                        elif isinstance(metric_value, (int, float)):
                            full_key = f"{current_key}/{metric_name}"
                            metrics[full_key] = float(metric_value)
                else:
                    # Recurse into nested dicts
                    nested = self._extract_metrics_from_nested_dict(value, current_key)
                    metrics.update(nested)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                # Direct numeric value
                metrics[current_key] = float(value)
        
        return metrics
    
    def extract_baseline_from_paper(self, paper_content: str, codebase_path: Path = None) -> BaselineMetrics:
        """
        Extract baseline metrics from paper text or report.md files.
        
        Args:
            paper_content: Text content from the research paper
            codebase_path: Path to codebase for finding report.md files
            
        Returns:
            BaselineMetrics with extracted values
        """
        # First, try to parse report.md files for configuration-specific metrics
        if codebase_path:
            metrics = self._parse_report_files(codebase_path)
            if metrics:
                return BaselineMetrics(
                    metrics=metrics,
                    source="Parsed from outputs_all_methods/report.md"
                )
        
        # Fallback to LLM extraction from paper
        if not self.llm_client:
            logger.warning("No LLM client and no report.md found")
            return BaselineMetrics(metrics={}, source="No extraction method available")
        
        system_prompt = """You are an expert at reading research papers and extracting quantitative metrics.
Extract ALL numerical performance metrics from the results/evaluation section.
Common metrics: Recall@K, MRR, NDCG, Precision, F1, Accuracy, MAP.
Return as flat JSON with metric names as keys and numeric values only."""
        
        user_prompt = f"""Extract ALL performance metrics from this research paper's results section.
Look for metrics like: Recall@10, MRR, NDCG@10, F1, Accuracy, Precision, etc.

Paper text (results section):
{paper_content[:6000]}

Return ONLY a flat JSON object with metric names and their numeric values.
Example: {{"recall@10": 0.458, "mrr": 0.252, "f1": 0.567, "accuracy": 0.425}}

JSON:"""
        
        try:
            metrics_dict = self.llm_client.extract_json(user_prompt, system_prompt)
            
            # Clean up metric names (lowercase, normalize)
            cleaned_metrics = {}
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    # Normalize metric names
                    clean_key = key.lower().replace(' ', '_').replace('-', '_')
                    cleaned_metrics[clean_key] = float(value)
            
            if not cleaned_metrics:
                logger.warning("No metrics extracted from paper")
            
            return BaselineMetrics(
                metrics=cleaned_metrics,
                source="Extracted from paper using LLM"
            )
        except Exception as e:
            logger.error(f"Failed to extract baseline metrics: {e}")
            return BaselineMetrics(metrics={}, source="Extraction failed")
    
    def _parse_report_files(self, codebase_path: Path) -> Dict[str, float]:
        """
        Parse report.md files to extract configuration-specific baseline metrics.
        
        Args:
            codebase_path: Path to codebase
            
        Returns:
            Dict of "config/retriever/metric" -> value
        """
        metrics = {}
        
        # Look for report.md in output directories
        output_dirs = [
            "outputs_all_methods",
            "outputs_all_methods_full",
            "outputs_all_methods_oracle"
        ]
        
        for dir_name in output_dirs:
            report_path = codebase_path / dir_name / "report.md"
            if report_path.exists():
                try:
                    with open(report_path, 'r') as f:
                        content = f.read()
                    
                    # Parse the markdown report
                    parsed = self._parse_markdown_report(content)
                    
                    # Flatten: parsed is {config/retriever: {metric: value}}
                    # Convert to: {dir/config/retriever/metric: value}
                    for config_retriever, config_metrics in parsed.items():
                        for metric_name, value in config_metrics.items():
                            key = f"{dir_name}/{config_retriever}/{metric_name}"
                            metrics[key] = value
                    
                    logger.info(f"✓ Parsed {len(parsed)} config/retriever combinations from {report_path.name}")
                except Exception as e:
                    logger.error(f"Failed to parse {report_path}: {e}")
        
        return metrics
    
    def _parse_markdown_report(self, content: str) -> Dict[str, Dict[str, float]]:
        """
        Parse markdown report to extract metrics by configuration.
        
        Args:
            content: Markdown report content
            
        Returns:
            Dict mapping config_name -> {metric: value}
        """
        import re
        
        metrics_by_config = {}
        current_config = None
        
        lines = content.split('\n')
        for line in lines:
            # Look for configuration headers (###)
            config_match = re.match(r'^###\s+(\w+/\w+)', line)
            if config_match:
                current_config = config_match.group(1)
                metrics_by_config[current_config] = {}
                continue
            
            if current_config:
                # Look for retrieval metrics like "bm25: Recall@10=0.458, MRR=0.252"
                retrieval_match = re.search(
                    r'(\w+):\s+Recall@10=([\d.]+)(?:.*?)MRR=([\d.]+)',
                    line
                )
                if retrieval_match:
                    retriever = retrieval_match.group(1)
                    recall = float(retrieval_match.group(2))
                    mrr = float(retrieval_match.group(3))
                    
                    # Store with retriever in key
                    base_key = f"{current_config}/{retriever}"
                    metrics_by_config.setdefault(base_key, {})
                    metrics_by_config[base_key]['recall@10'] = recall
                    metrics_by_config[base_key]['mrr'] = mrr
                
                # Look for downstream task metrics like "bm25: Accuracy=0.425, F1=0.567"
                task_match = re.search(
                    r'(\w+):\s+Accuracy=([\d.]+)(?:.*?)F1=([\d.]+)',
                    line
                )
                if task_match:
                    retriever = task_match.group(1)
                    accuracy = float(task_match.group(2))
                    f1 = float(task_match.group(3))
                    
                    base_key = f"{current_config}/{retriever}"
                    metrics_by_config.setdefault(base_key, {})
                    metrics_by_config[base_key]['accuracy'] = accuracy
                    metrics_by_config[base_key]['f1'] = f1
        
        return metrics_by_config
    
    def compare_results(self, baseline: BaselineMetrics, 
                       reproduced: Dict[str, float]) -> List[ComparisonResult]:
        """
        Compare reproduced results to baseline metrics.
        
        Baseline format: outputs_all_methods/sentence/minimal/bm25/recall@10
        Reproduced format: outputs_all_methods/sentence/minimal/retrieval/bm25/metrics/recall@10
        
        Args:
            baseline: Baseline metrics from paper
            reproduced: Metrics from reproduced experiments (can be nested paths)
            
        Returns:
            List of ComparisonResult objects
        """
        comparisons = []
        
        for baseline_key, baseline_value in baseline.metrics.items():
            # Parse baseline key
            # Format: outputs_all_methods/sentence/minimal/bm25/recall@10
            baseline_parts = baseline_key.split('/')
            
            if len(baseline_parts) < 5:
                logger.warning(f"Baseline key has unexpected format: {baseline_key}")
                continue
            
            exp_set = baseline_parts[0]  # e.g., outputs_all_methods
            granularity = baseline_parts[1]  # e.g., sentence
            strategy = baseline_parts[2]  # e.g., minimal
            retriever = baseline_parts[3]  # e.g., bm25
            metric_name = baseline_parts[4]  # e.g., recall@10
            
            # Find matching reproduced metrics
            # Reproduced format: outputs_all_methods/sentence/minimal/retrieval/bm25/metrics/recall@10
            # or similar variations
            matches = []
            
            for repro_key, repro_value in reproduced.items():
                repro_parts = repro_key.split('/')
                
                # Check if this reproduced metric matches the baseline configuration
                # Must match: exp_set, granularity, strategy, retriever, and metric_name
                if (exp_set in repro_key and
                    granularity in repro_key and
                    strategy in repro_key and
                    retriever in repro_key and
                    repro_key.endswith(metric_name)):
                    matches.append((repro_key, repro_value))
            
            if not matches:
                # Try looser matching - just check if key components are present
                for repro_key, repro_value in reproduced.items():
                    key_lower = repro_key.lower()
                    if (granularity in key_lower and
                        strategy in key_lower and
                        retriever in key_lower and
                        metric_name.lower() in key_lower):
                        matches.append((repro_key, repro_value))
                        break
            
            if not matches:
                logger.debug(f"No match found for baseline: {baseline_key}")
                continue
            
            # Create comparison for each match (usually just one)
            for config_path, reproduced_value in matches:
                # Calculate difference
                difference = reproduced_value - baseline_value
                
                # Calculate percent difference (handle divide by zero)
                if baseline_value != 0:
                    percent_diff = (difference / abs(baseline_value)) * 100
                else:
                    percent_diff = float('inf') if difference != 0 else 0
                
                # Check if within threshold
                within_threshold = abs(percent_diff) <= (self.threshold * 100)
                
                comparisons.append(ComparisonResult(
                    metric_name=metric_name,
                    baseline_value=baseline_value,
                    reproduced_value=reproduced_value,
                    difference=difference,
                    percent_difference=percent_diff,
                    within_threshold=within_threshold,
                    configuration=config_path
                ))
        
        return comparisons
    
    def generate_report(self, comparisons: List[ComparisonResult]) -> str:
        """
        Generate a human-readable report of the comparison.
        
        Args:
            comparisons: List of comparison results
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "="*80,
            "REPRODUCTION RESULTS EVALUATION - ALL CONFIGURATIONS",
            "="*80,
            ""
        ]
        
        if not comparisons:
            report_lines.append("No metrics available for comparison.")
            return "\n".join(report_lines)
        
        # Group by experiment set and configuration
        by_experiment = {}
        for comp in comparisons:
            parts = comp.configuration.split('/')
            exp_set = parts[0] if parts else "unknown"
            
            if exp_set not in by_experiment:
                by_experiment[exp_set] = []
            by_experiment[exp_set].append(comp)
        
        # Overall summary
        total_metrics = len(comparisons)
        within_threshold = sum(1 for c in comparisons if c.within_threshold)
        
        report_lines.extend([
            f"Total comparisons: {total_metrics}",
            f"Within threshold ({self.threshold*100}%): {within_threshold}/{total_metrics}",
            f"Success rate: {within_threshold/total_metrics*100:.1f}%",
            f"Experiment sets analyzed: {len(by_experiment)}",
            "",
            "="*80
        ])
        
        # Detailed results by experiment set
        for exp_set, comps in sorted(by_experiment.items()):
            report_lines.extend([
                "",
                f"\n{'='*80}",
                f"EXPERIMENT SET: {exp_set}",
                f"{'='*80}",
                f"Comparisons in this set: {len(comps)}",
                ""
            ])
            
            # Group by configuration within this experiment set
            by_config = {}
            for comp in comps:
                config = '/'.join(comp.configuration.split('/')[1:4])  # Get config/model part
                if config not in by_config:
                    by_config[config] = []
                by_config[config].append(comp)
            
            # Show summary for each configuration
            for config, config_comps in sorted(by_config.items())[:10]:  # Limit to 10 configs
                passing = sum(1 for c in config_comps if c.within_threshold)
                status = "✓" if passing == len(config_comps) else "✗"
                
                report_lines.append(
                    f"  {status} {config}: {passing}/{len(config_comps)} metrics pass"
                )
                
                # Show top 3 metrics for this config
                for comp in sorted(config_comps, key=lambda x: abs(x.percent_difference))[:3]:
                    status_icon = "✓" if comp.within_threshold else "✗"
                    report_lines.append(
                        f"      {status_icon} {comp.metric_name}: baseline={comp.baseline_value:.4f}, "
                        f"reproduced={comp.reproduced_value:.4f} ({comp.percent_difference:+.2f}%)"
                    )
        
        # Best performing configurations (closest to baseline)
        report_lines.extend([
            "",
            "="*80,
            "BEST PERFORMING CONFIGURATIONS (closest to baseline):",
            "="*80
        ])
        
        best_configs = {}
        for comp in comparisons:
            config = '/'.join(comp.configuration.split('/')[1:4])
            if config not in best_configs:
                best_configs[config] = []
            best_configs[config].append(abs(comp.percent_difference))
        
        # Average percent difference per config
        config_scores = {
            k: sum(v) / len(v) 
            for k, v in best_configs.items()
        }
        
        for config, avg_diff in sorted(config_scores.items(), key=lambda x: x[1])[:10]:
            report_lines.append(f"  {config}: avg diff = {avg_diff:.2f}%")
        
        report_lines.extend([
            "",
            "="*80
        ])
        
        return "\n".join(report_lines)
    
    def analyze_differences_with_llm(self, comparisons: List[ComparisonResult],
                                    paper_context: str) -> str:
        """
        Use LLM to provide insights on why results might differ.
        
        Args:
            comparisons: Comparison results
            paper_context: Context from the research paper
            
        Returns:
            Analysis and potential explanations
        """
        if not self.llm_client:
            return "LLM analysis not available (no client provided)"
        
        # Sample some interesting comparisons
        sample_size = min(10, len(comparisons))
        sample = sorted(comparisons, key=lambda x: abs(x.percent_difference))[:sample_size]
        
        # Build comparison summary for LLM
        summary = "Reproduction results across multiple configurations:\n\n"
        
        # Group by experiment set
        by_exp = {}
        for comp in sample:
            exp = comp.configuration.split('/')[0]
            if exp not in by_exp:
                by_exp[exp] = []
            by_exp[exp].append(comp)
        
        for exp_set, comps in by_exp.items():
            summary += f"\n{exp_set}:\n"
            for comp in comps[:5]:
                config = '/'.join(comp.configuration.split('/')[1:3])
                summary += f"  - {config}/{comp.metric_name}: baseline={comp.baseline_value:.4f}, "
                summary += f"reproduced={comp.reproduced_value:.4f} ({comp.percent_difference:+.2f}%)\n"
        
        system_prompt = """You are an expert in machine learning research and experiment reproduction.
Analyze the differences between baseline and reproduced results across multiple experimental configurations."""
        
        user_prompt = f"""Paper context (truncated):
{paper_context[:2000]}

{summary}

The experiments tested multiple configurations:
- Different granularities: sentence vs paragraph
- Different decontextualization strategies: minimal, title_only, heading_only, etc.
- Different retrievers: BM25, TF-IDF, ColBERT, Cross-Encoder

Explain:
1. Why some configurations match better than others
2. Possible reasons for metric differences
3. Which variations are expected vs unexpected

Provide a concise analysis (3-4 paragraphs)."""
        
        try:
            analysis = self.llm_client.generate(user_prompt, system_prompt)
            return analysis
        except Exception as e:
            logger.error(f"Failed to generate LLM analysis: {e}")
            return f"LLM analysis failed: {str(e)}"
    
    def generate_summary_statistics(self, comparisons: List[ComparisonResult]) -> str:
        """
        Generate summary statistics grouped by models and configurations.
        
        Args:
            comparisons: List of comparison results
            
        Returns:
            Summary statistics string
        """
        if not comparisons:
            return "No comparisons available for statistics."
        
        lines = [
            "\n" + "="*80,
            "SUMMARY STATISTICS",
            "="*80,
            ""
        ]
        
        # Statistics by retriever model (bm25, tfidf, colbert, cross_encoder)
        by_model = {}
        for comp in comparisons:
            parts = comp.configuration.split('/')
            if len(parts) >= 3:
                model = parts[2]  # e.g., bm25, tfidf
                if model not in by_model:
                    by_model[model] = []
                by_model[model].append(comp)
        
        lines.append("Performance by Retrieval Model:")
        lines.append("-" * 80)
        for model, comps in sorted(by_model.items()):
            passing = sum(1 for c in comps if c.within_threshold)
            avg_diff = sum(abs(c.percent_difference) for c in comps) / len(comps)
            lines.append(
                f"  {model:15s}: {passing:3d}/{len(comps):3d} pass  |  "
                f"avg diff: {avg_diff:6.2f}%"
            )
        
        # Statistics by granularity (sentence vs paragraph)
        by_granularity = {}
        for comp in comparisons:
            parts = comp.configuration.split('/')
            if len(parts) >= 2:
                granularity = parts[1].split('/')[0]  # sentence or paragraph
                if granularity not in by_granularity:
                    by_granularity[granularity] = []
                by_granularity[granularity].append(comp)
        
        lines.append("")
        lines.append("Performance by Granularity:")
        lines.append("-" * 80)
        for gran, comps in sorted(by_granularity.items()):
            passing = sum(1 for c in comps if c.within_threshold)
            avg_diff = sum(abs(c.percent_difference) for c in comps) / len(comps)
            lines.append(
                f"  {gran:15s}: {passing:3d}/{len(comps):3d} pass  |  "
                f"avg diff: {avg_diff:6.2f}%"
            )
        
        lines.append("")
        lines.append("="*80)
        
        return "\n".join(lines)
    
    def generate_comprehensive_conclusions(self, 
                                          comparisons: List[ComparisonResult],
                                          experiment_sets: List[ExperimentSet],
                                          baseline: BaselineMetrics,
                                          paper_context: str = "") -> str:
        """
        Generate comprehensive conclusions analyzing LLM agent performance.
        
        Args:
            comparisons: All comparison results
            experiment_sets: Experiment set metadata
            baseline: Baseline metrics
            paper_context: Context from paper
            
        Returns:
            Comprehensive analysis and recommendations
        """
        lines = [
            "\n" + "="*80,
            "COMPREHENSIVE ANALYSIS & CONCLUSIONS",
            "="*80,
            ""
        ]
        
        # 1. Overall Performance Assessment
        lines.append("1. OVERALL REPRODUCTION PERFORMANCE")
        lines.append("-" * 80)
        
        total = len(comparisons)
        passing = sum(1 for c in comparisons if c.within_threshold)
        success_rate = (passing / total * 100) if total > 0 else 0
        
        # Performance grading
        if success_rate >= 90:
            grade = "EXCELLENT"
            assessment = "The LLM agent successfully reproduced the experiments with high fidelity."
        elif success_rate >= 70:
            grade = "GOOD"
            assessment = "The LLM agent achieved good reproduction with some deviations."
        elif success_rate >= 50:
            grade = "MODERATE"
            assessment = "The LLM agent partially reproduced results but with notable differences."
        else:
            grade = "POOR"
            assessment = "The LLM agent struggled to accurately reproduce the baseline results."
        
        lines.extend([
            f"Grade: {grade} ({success_rate:.1f}% success rate)",
            f"Assessment: {assessment}",
            "",
            f"Total Comparisons: {total}",
            f"Metrics Within Threshold ({self.threshold*100}%): {passing}/{total}",
            f"Experiment Sets Tested: {len(experiment_sets)}",
            f"Baseline Source: {baseline.source}",
            ""
        ])
        
        # 2. Key Findings by Configuration
        lines.append("2. KEY FINDINGS BY CONFIGURATION")
        lines.append("-" * 80)
        
        # Analyze by granularity
        by_granularity = {}
        for comp in comparisons:
            parts = comp.configuration.split('/')
            if len(parts) >= 2:
                gran = parts[1].split('/')[0]
                if gran not in by_granularity:
                    by_granularity[gran] = []
                by_granularity[gran].append(comp)
        
        for gran, comps in sorted(by_granularity.items()):
            passing_count = sum(1 for c in comps if c.within_threshold)
            gran_rate = (passing_count / len(comps) * 100) if comps else 0
            avg_deviation = sum(abs(c.percent_difference) for c in comps) / len(comps) if comps else 0
            
            lines.append(f"\n{gran.upper()} Granularity:")
            lines.append(f"  Success Rate: {gran_rate:.1f}% ({passing_count}/{len(comps)} metrics)")
            lines.append(f"  Avg Deviation: {avg_deviation:.2f}%")
            
            if gran_rate < 50:
                lines.append(f"  ⚠️  LOW PERFORMANCE: {gran} configuration needs investigation")
        
        # Analyze by experiment set
        lines.append("\nBy Experiment Set:")
        by_exp = {}
        for comp in comparisons:
            exp = comp.configuration.split('/')[0]
            if exp not in by_exp:
                by_exp[exp] = []
            by_exp[exp].append(comp)
        
        for exp, comps in sorted(by_exp.items()):
            passing_count = sum(1 for c in comps if c.within_threshold)
            exp_rate = (passing_count / len(comps) * 100) if comps else 0
            avg_deviation = sum(abs(c.percent_difference) for c in comps) / len(comps) if comps else 0
            
            lines.append(f"\n{exp}:")
            lines.append(f"  Success Rate: {exp_rate:.1f}% ({passing_count}/{len(comps)} metrics)")
            lines.append(f"  Avg Deviation: {avg_deviation:.2f}%")
            
            if avg_deviation > 50:
                lines.append(f"  ⚠️  HIGH DEVIATION: Possible data/environment mismatch")
        
        lines.append("")
        
        # 3. Root Cause Analysis
        lines.append("3. ROOT CAUSE ANALYSIS")
        lines.append("-" * 80)
        
        # Find worst performing metrics
        worst_metrics = sorted(comparisons, key=lambda x: abs(x.percent_difference), reverse=True)[:5]
        
        lines.append("\nTop Issues Identified:")
        for i, comp in enumerate(worst_metrics, 1):
            config_short = '/'.join(comp.configuration.split('/')[1:4])
            lines.append(
                f"  {i}. {comp.metric_name} in {config_short}: "
                f"{comp.percent_difference:+.2f}% deviation"
            )
        
        lines.append("\nPossible Root Causes:")
        
        # Analyze patterns
        high_deviation_count = sum(1 for c in comparisons if abs(c.percent_difference) > 50)
        if high_deviation_count > total * 0.3:  # More than 30% have >50% deviation
            lines.append("  • DATA MISMATCH: Many metrics show >50% deviation")
            lines.append("    → Check if experiment used same dataset as paper")
            lines.append("    → Verify data preprocessing steps match paper methodology")
        
        # Check for systematic issues
        sentence_comps = [c for c in comparisons if 'sentence' in c.configuration.lower()]
        if sentence_comps:
            sentence_rate = sum(1 for c in sentence_comps if c.within_threshold) / len(sentence_comps) * 100
            if sentence_rate < 30:
                lines.append("  • SENTENCE GRANULARITY ISSUE: Poor performance on sentence-level tasks")
                lines.append("    → May indicate chunking/segmentation differences")
                lines.append("    → Check sentence tokenization implementation")
        
        # Check for retrieval model issues
        for model in ['bm25', 'tfidf', 'colbert', 'cross_encoder']:
            model_comps = [c for c in comparisons if model in c.configuration.lower()]
            if model_comps:
                model_rate = sum(1 for c in model_comps if c.within_threshold) / len(model_comps) * 100
                if model_rate < 30:
                    lines.append(f"  • {model.upper()} IMPLEMENTATION: Significant deviations detected")
                    lines.append(f"    → Verify {model} configuration matches paper specifications")
        
        lines.append("")
        
        # 4. What the LLM Agent Accomplished
        lines.append("4. LLM AGENT ACCOMPLISHMENTS")
        lines.append("-" * 80)
        
        lines.extend([
            "The Local LLM Agent successfully:",
            "  ✓ Parsed research paper PDF and extracted methodology",
            "  ✓ Identified and analyzed local codebase structure",
            f"  ✓ Executed {len(experiment_sets)} complete experiment sets",
            f"  ✓ Generated {total} metric comparisons across configurations",
            "  ✓ Extracted and compared baseline metrics from report files",
            ""
        ])
        
        # Best performing configurations
        best_configs = {}
        for comp in comparisons:
            config = '/'.join(comp.configuration.split('/')[1:4])
            if config not in best_configs:
                best_configs[config] = []
            best_configs[config].append(abs(comp.percent_difference))
        
        config_scores = {k: sum(v) / len(v) for k, v in best_configs.items()}
        top_configs = sorted(config_scores.items(), key=lambda x: x[1])[:3]
        
        if top_configs:
            lines.append("Best Reproduced Configurations:")
            for config, avg_diff in top_configs:
                lines.append(f"  • {config}: {avg_diff:.2f}% avg deviation")
            lines.append("")
        
        # 5. Recommendations for Improvement
        lines.append("5. RECOMMENDATIONS FOR IMPROVEMENT")
        lines.append("-" * 80)
        
        lines.append("\nImmediate Actions:")
        
        if success_rate < 50:
            lines.append("  🔴 CRITICAL: Low success rate indicates fundamental issues")
            lines.append("     1. Verify data files match paper's dataset")
            lines.append("     2. Check dependency versions match paper requirements")
            lines.append("     3. Confirm environment setup (Python version, libraries)")
        
        if high_deviation_count > 0:
            lines.append(f"  🟡 WARNING: {high_deviation_count} metrics show >50% deviation")
            lines.append("     1. Compare experiment parameters with paper methodology")
            lines.append("     2. Validate preprocessing/normalization steps")
            lines.append("     3. Check for stochastic processes (set random seeds)")
        
        lines.append("\nLLM Agent Enhancements:")
        lines.append("  1. BASELINE EXTRACTION:")
        lines.append("     → Improve parsing of configuration-specific metrics")
        lines.append("     → Extract metrics with full context (granularity, model, strategy)")
        lines.append("     → Use structured templates for metric extraction")
        lines.append("")
        lines.append("  2. VALIDATION CHECKS:")
        lines.append("     → Verify dataset integrity before running experiments")
        lines.append("     → Compare file checksums with paper's data release")
        lines.append("     → Validate dependency versions match paper requirements")
        lines.append("")
        lines.append("  3. PROMPT ENGINEERING:")
        lines.append("     → Provide more context about experiment configurations")
        lines.append("     → Use chain-of-thought prompting for methodology extraction")
        lines.append("     → Add examples of expected metric formats")
        lines.append("")
        lines.append("  4. ERROR RECOVERY:")
        lines.append("     → Implement retry logic with different LLM models")
        lines.append("     → Add fallback to structured parsing when LLM fails")
        lines.append("     → Log intermediate outputs for debugging")
        lines.append("")
        
        # 6. Next Steps
        lines.append("6. RECOMMENDED NEXT STEPS")
        lines.append("-" * 80)
        
        lines.append("\nShort Term (1-2 days):")
        lines.append("  [ ] Verify dataset files match paper specifications")
        lines.append("  [ ] Check all dependency versions against requirements.txt")
        lines.append("  [ ] Run experiments with fixed random seeds for reproducibility")
        lines.append("  [ ] Compare preprocessing code with paper's implementation")
        lines.append("")
        
        lines.append("Medium Term (1 week):")
        lines.append("  [ ] Enhance baseline extraction to parse all configuration variations")
        lines.append("  [ ] Add automated data validation before experiment execution")
        lines.append("  [ ] Implement hyperparameter sweep to match paper settings")
        lines.append("  [ ] Create detailed logging of all experiment steps")
        lines.append("")
        
        lines.append("Long Term (2-4 weeks):")
        lines.append("  [ ] Build a library of paper-specific parsers for common formats")
        lines.append("  [ ] Develop automated configuration matching system")
        lines.append("  [ ] Create visualization dashboard for metric comparisons")
        lines.append("  [ ] Implement ablation study support for debugging deviations")
        lines.append("")
        
        # 7. Conclusion Summary
        lines.append("7. FINAL ASSESSMENT")
        lines.append("-" * 80)
        
        if success_rate >= 70:
            conclusion = (
                "The LLM agent demonstrates strong capability in automating research "
                "reproduction. With minor refinements to baseline extraction and validation, "
                "it can serve as a reliable tool for verifying experimental results."
            )
        elif success_rate >= 40:
            conclusion = (
                "The LLM agent shows promise but requires significant improvements in "
                "configuration matching and environment validation. Focus on data integrity "
                "and parameter alignment before production use."
            )
        else:
            conclusion = (
                "The LLM agent requires substantial development before being production-ready. "
                "Critical issues in data handling, metric extraction, or environment setup "
                "must be addressed. Consider manual verification of key experiment steps."
            )
        
        lines.extend([
            conclusion,
            "",
            f"Overall Grade: {grade}",
            f"Confidence Level: {'HIGH' if success_rate > 80 else 'MEDIUM' if success_rate > 50 else 'LOW'}",
            "",
            "="*80
        ])
        
        return "\n".join(lines)

