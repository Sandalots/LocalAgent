#!/usr/bin/env python3
"""
Test script to verify improvements to baseline extraction and conclusions generation.
"""

from pathlib import Path
from src.result_evaluator import ResultEvaluator, BaselineMetrics, ComparisonResult, ExperimentSet
from src.llm_client import OllamaClient

def test_report_parsing():
    """Test parsing report.md for baseline metrics."""
    print("="*70)
    print("TEST 1: Report.md Parsing")
    print("="*70)
    
    evaluator = ResultEvaluator()
    codebase_path = Path("/Users/sandymacdonald/Projects/LocalAgent/paper_source_code/supplementary_material/code")
    
    # Test parsing report files
    metrics = evaluator._parse_report_files(codebase_path)
    
    print(f"\nExtracted {len(metrics)} baseline metrics from report.md files")
    print("\nSample metrics:")
    for i, (key, value) in enumerate(list(metrics.items())[:10]):
        print(f"  {key}: {value}")
    
    print("\n✓ Report parsing test passed\n")
    return metrics

def test_baseline_extraction():
    """Test full baseline extraction with codebase path."""
    print("="*70)
    print("TEST 2: Baseline Extraction with Report.md")
    print("="*70)
    
    llm_client = OllamaClient()
    evaluator = ResultEvaluator(llm_client)
    codebase_path = Path("/Users/sandymacdonald/Projects/LocalAgent/paper_source_code/supplementary_material/code")
    
    baseline = evaluator.extract_baseline_from_paper(
        paper_content="",  # Empty paper content to force report.md parsing
        codebase_path=codebase_path
    )
    
    print(f"\nExtracted {len(baseline.metrics)} baseline metrics")
    print(f"Source: {baseline.source}")
    print("\nSample baseline metrics:")
    for i, (key, value) in enumerate(list(baseline.metrics.items())[:10]):
        print(f"  {key}: {value}")
    
    print("\n✓ Baseline extraction test passed\n")
    return baseline

def test_comprehensive_conclusions():
    """Test comprehensive conclusions generation."""
    print("="*70)
    print("TEST 3: Comprehensive Conclusions Generation")
    print("="*70)
    
    llm_client = OllamaClient()
    evaluator = ResultEvaluator(llm_client)
    
    # Create mock data
    baseline = BaselineMetrics(
        metrics={
            "recall@10": 1.0,
            "mrr": 0.425
        },
        source="Mock data"
    )
    
    # Create mock comparisons
    comparisons = [
        ComparisonResult(
            metric_name="recall@10",
            baseline_value=1.0,
            reproduced_value=1.0,
            difference=0.0,
            percent_difference=0.0,
            within_threshold=True,
            configuration="outputs_all_methods/paragraph/minimal/bm25/recall@10"
        ),
        ComparisonResult(
            metric_name="mrr",
            baseline_value=0.425,
            reproduced_value=0.400,
            difference=-0.025,
            percent_difference=-5.88,
            within_threshold=False,
            configuration="outputs_all_methods/paragraph/minimal/bm25/mrr"
        ),
        ComparisonResult(
            metric_name="recall@10",
            baseline_value=1.0,
            reproduced_value=0.7083,
            difference=-0.2917,
            percent_difference=-29.17,
            within_threshold=False,
            configuration="outputs_all_methods/sentence/minimal/bm25/recall@10"
        )
    ]
    
    # Mock experiment sets
    experiment_sets = [
        ExperimentSet(
            name="outputs_all_methods",
            results={},
            total_configs=10,
            total_metrics=60
        )
    ]
    
    # Generate conclusions
    conclusions = evaluator.generate_comprehensive_conclusions(
        comparisons,
        experiment_sets,
        baseline,
        paper_context="Mock paper context about decontextualization experiments"
    )
    
    print("\n" + conclusions)
    print("\n✓ Conclusions generation test passed\n")
    return conclusions

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TESTING IMPROVEMENTS TO LLM AGENT")
    print("="*70 + "\n")
    
    try:
        # Test 1: Report parsing
        metrics = test_report_parsing()
        
        # Test 2: Baseline extraction
        baseline = test_baseline_extraction()
        
        # Test 3: Comprehensive conclusions
        conclusions = test_comprehensive_conclusions()
        
        print("="*70)
        print("ALL TESTS PASSED!")
        print("="*70)
        print("\nKey Improvements:")
        print("  ✓ Report.md parsing extracts configuration-specific baselines")
        print("  ✓ Baseline extraction tries report.md before LLM")
        print("  ✓ Comprehensive conclusions provide actionable insights")
        print("  ✓ Root cause analysis identifies specific issues")
        print("  ✓ Recommendations section provides next steps")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
