"""
Result Evaluator Module

Compares experiment results to baseline metrics from the paper.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np


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
    
    def extract_baseline_from_paper(self, paper_content: str) -> BaselineMetrics:
        """
        Extract baseline metrics from paper text using LLM.
        
        Args:
            paper_content: Text content from the research paper
            
        Returns:
            BaselineMetrics with extracted values
        """
        if not self.llm_client:
            raise ValueError("LLM client required for extracting baselines")
        
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
                logger.warning("No metrics extracted from paper, using report.md if available")
            
            return BaselineMetrics(
                metrics=cleaned_metrics,
                source="Extracted from paper using LLM"
            )
        except Exception as e:
            logger.error(f"Failed to extract baseline metrics: {e}")
            return BaselineMetrics(metrics={}, source="Extraction failed")
    
    def compare_results(self, baseline: BaselineMetrics, 
                       reproduced: Dict[str, float]) -> List[ComparisonResult]:
        """
        Compare reproduced results to baseline metrics.
        
        Args:
            baseline: Baseline metrics from paper
            reproduced: Metrics from reproduced experiments
            
        Returns:
            List of ComparisonResult objects
        """
        comparisons = []
        
        # Normalize all metric names for matching
        normalized_reproduced = {
            k.lower().replace(' ', '_').replace('-', '_'): v 
            for k, v in reproduced.items()
        }
        
        for metric_name, baseline_value in baseline.metrics.items():
            metric_key = metric_name.lower().replace(' ', '_').replace('-', '_')
            
            # Try exact match first
            if metric_key in normalized_reproduced:
                reproduced_value = normalized_reproduced[metric_key]
            else:
                # Try fuzzy matching (e.g., "recall_10" matches "recall@10")
                reproduced_value = None
                for key in normalized_reproduced.keys():
                    if metric_key.replace('@', '_') == key.replace('@', '_'):
                        reproduced_value = normalized_reproduced[key]
                        break
                
                if reproduced_value is None:
                    logger.warning(f"Metric {metric_name} not found in reproduced results")
                    continue
            
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
                within_threshold=within_threshold
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
            "="*70,
            "REPRODUCTION RESULTS EVALUATION",
            "="*70,
            ""
        ]
        
        if not comparisons:
            report_lines.append("No metrics available for comparison.")
            return "\n".join(report_lines)
        
        # Summary statistics
        total_metrics = len(comparisons)
        within_threshold = sum(1 for c in comparisons if c.within_threshold)
        
        report_lines.extend([
            f"Total metrics compared: {total_metrics}",
            f"Within threshold ({self.threshold*100}%): {within_threshold}/{total_metrics}",
            f"Success rate: {within_threshold/total_metrics*100:.1f}%",
            "",
            "DETAILED COMPARISON:",
            "-"*70
        ])
        
        # Detailed results for each metric
        for comp in comparisons:
            status = "✓ PASS" if comp.within_threshold else "✗ FAIL"
            
            report_lines.extend([
                f"\nMetric: {comp.metric_name}",
                f"  Baseline:    {comp.baseline_value:.4f}",
                f"  Reproduced:  {comp.reproduced_value:.4f}",
                f"  Difference:  {comp.difference:+.4f} ({comp.percent_difference:+.2f}%)",
                f"  Status:      {status}"
            ])
        
        report_lines.extend([
            "",
            "="*70
        ])
        
        return "\n".join(report_lines)
    
    def analyze_differences_with_llm(self, comparisons: List[ComparisonResult],
                                    paper_context: str) -> str:
        """
        Use LLM to provide insights on why results might differ.
        
        Args:
            comparisons: Comparison results
            comparisons: List of comparison results
            paper_context: Context from the research paper
            
        Returns:
            Analysis and potential explanations
        """
        if not self.llm_client:
            return "LLM analysis not available (no client provided)"
        
        # Build comparison summary for LLM
        summary = "Reproduction results:\n"
        for comp in comparisons:
            summary += f"- {comp.metric_name}: baseline={comp.baseline_value:.4f}, "
            summary += f"reproduced={comp.reproduced_value:.4f} ({comp.percent_difference:+.2f}%)\n"
        
        system_prompt = """You are an expert in machine learning research and experiment reproduction.
Analyze the differences between baseline and reproduced results and suggest possible reasons."""
        
        user_prompt = f"""Paper context (truncated):
{paper_context[:2000]}

{summary}

Explain possible reasons for these differences. Consider:
1. Random seed variations
2. Hardware differences
3. Software version differences
4. Implementation details not mentioned in the paper
5. Dataset preprocessing differences

Provide a concise analysis."""
        
        try:
            analysis = self.llm_client.generate(user_prompt, system_prompt)
            return analysis
        except Exception as e:
            logger.error(f"Failed to generate LLM analysis: {e}")
            return f"LLM analysis failed: {str(e)}"
