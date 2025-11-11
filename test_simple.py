#!/usr/bin/env python3
"""Simple test without LLM."""

from pathlib import Path
from src.result_evaluator import ResultEvaluator, BaselineMetrics
import json

def test():
    evaluator = ResultEvaluator()
    codebase_path = Path("/Users/sandymacdonald/Projects/LocalAgent/paper_source_code/supplementary_material/code")
    
    print("="*70)
    print("TEST 1: Report.md Parsing")
    print("="*70)
    baseline_metrics = evaluator._parse_report_files(codebase_path)
    
    print(f"\nFound {len(baseline_metrics)} baseline metrics")
    print("\nSample baseline metrics:")
    for key, value in list(baseline_metrics.items())[:10]:
        print(f"  {key}: {value}")
    
    if len(baseline_metrics) == 0:
        print("\n❌ NO BASELINE METRICS FOUND!")
        return
    
    print(f"\n✓ Successfully parsed {len(baseline_metrics)} baseline metrics")
    
    print("\n" + "="*70)
    print("TEST 2: Reproduced Metrics Extraction")
    print("="*70)
    
    # Load experiment results
    exp_sets = evaluator.load_all_experiment_results(codebase_path)
    print(f"\nLoaded {len(exp_sets)} experiment sets")
    
    # Extract all metrics
    reproduced_metrics = evaluator.extract_all_metrics_from_experiments(exp_sets)
    print(f"Extracted {len(reproduced_metrics)} reproduced metrics")
    print("\nSample reproduced metrics:")
    for key, value in list(reproduced_metrics.items())[:10]:
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("TEST 3: Metric Matching")
    print("="*70)
    
    # Create baseline object
    baseline = BaselineMetrics(
        metrics=baseline_metrics,
        source="Test"
    )
    
    # Compare
    comparisons = evaluator.compare_results(baseline, reproduced_metrics)
    print(f"\nFound {len(comparisons)} comparisons")
    
    if len(comparisons) == 0:
        print("\n❌ NO COMPARISONS FOUND!")
        print("\nBaseline sample:")
        for k in list(baseline_metrics.keys())[:3]:
            print(f"  {k}")
        print("\nReproduced sample:")
        for k in list(reproduced_metrics.keys())[:3]:
            print(f"  {k}")
    else:
        print("\nSample comparisons:")
        for comp in comparisons[:5]:
            print(f"  {comp.metric_name}: baseline={comp.baseline_value:.4f}, reproduced={comp.reproduced_value:.4f} ({comp.percent_difference:+.2f}%)")
        
        passing = sum(1 for c in comparisons if c.within_threshold)
        print(f"\n✓ {passing}/{len(comparisons)} comparisons within threshold")

if __name__ == '__main__':
    test()
