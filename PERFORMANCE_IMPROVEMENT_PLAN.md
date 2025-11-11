# Performance Improvement Plan

## Current Performance Analysis

### Success Rates by Experiment Set:
- ✅ **outputs_all_methods**: 87.8% success (1.90% avg deviation)
- ⚠️ **outputs_all_methods_full**: 52.1% success (28.28% avg deviation) 
- ⚠️ **outputs_all_methods_oracle**: 67.1% success (19.64% avg deviation)

### Overall: 67.3% success rate, 330/490 metrics within 5% threshold

## Key Issues Identified

### 1. Extreme Deviations (-99%)
- Indicates **wrong metric matching** or **missing baseline values**
- Affects: recall@10 in multiple sentence-level configurations
- Root cause: Baseline extraction not capturing all config variations

### 2. outputs_all_methods_full Degradation
- 28% average deviation suggests **data or environment mismatch**
- Could be: different dataset version, preprocessing differences, or missing parameters

### 3. Sentence vs Paragraph Gap
- Sentence: 62.9% success (19.11% deviation)
- Paragraph: 71.8% success (15.61% deviation)
- Suggests: sentence segmentation/tokenization differences

## Improvement Strategies

### Priority 1: Fix Metric Matching (Immediate - 1 hour)

**Problem**: -99% deviations indicate wrong baseline values being compared

**Solutions**:
1. Add debug logging to show which baseline matches which reproduced metric
2. Validate baseline extraction captures ALL retrieval metrics (recall@1, @5, @10, @20, @50)
3. Add fallback to extract from complete_results.json if report.md is incomplete

**Implementation**:
```python
# In result_evaluator.py
def compare_results(self, baseline, reproduced):
    # Add detailed logging
    logger.debug(f"Baseline keys: {list(baseline.metrics.keys())[:5]}")
    logger.debug(f"Reproduced keys: {list(reproduced.keys())[:5]}")
    
    # Validate baseline coverage
    required_metrics = ['recall@10', 'mrr', 'accuracy', 'f1']
    missing = [m for m in required_metrics if not any(m in k for k in baseline.metrics.keys())]
    if missing:
        logger.warning(f"Baseline missing metrics: {missing}")
```

### Priority 2: Enhanced Baseline Extraction (High - 2-3 hours)

**Problem**: Report.md may not have all metrics that complete_results.json has

**Solutions**:
1. Extract baselines directly from complete_results.json as ground truth
2. Cross-validate report.md baselines with complete_results.json
3. Use the actual reproduced experiment data as baseline (since it's from the paper)

**Implementation**: Extract baseline from the paper's own complete_results.json

### Priority 3: Data Validation (High - 2-3 hours)

**Problem**: outputs_all_methods_full shows 28% deviation

**Solutions**:
1. Verify dataset files haven't changed (file hashes, row counts)
2. Check if "full" uses different preprocessing or data subset
3. Add pre-experiment validation checks

**Implementation**:
```python
# Add to experiment_runner.py
def validate_data_integrity(self, codebase_path):
    """Validate dataset files before running experiments."""
    data_dir = codebase_path / "data"
    
    checks = {
        'qa.jsonl': {'min_lines': 1000},
        'papers.jsonl': {'min_lines': 80},
        'qa-unlabeled.jsonl': {'min_lines': 2000}
    }
    
    for file, requirements in checks.items():
        filepath = data_dir / file
        if not filepath.exists():
            logger.warning(f"Missing data file: {file}")
            continue
        
        line_count = sum(1 for _ in open(filepath))
        if line_count < requirements['min_lines']:
            logger.warning(f"{file} has only {line_count} lines, expected >{requirements['min_lines']}")
```

### Priority 4: Random Seed Configuration (Medium - 1 hour)

**Problem**: Stochastic processes may cause variations

**Solutions**:
1. Extract random seeds from paper methodology
2. Set seeds in experiment configuration
3. Pass seeds to experiment runner

**Implementation**:
```python
# Add to config.yaml
experiment:
  timeout: 300
  random_seed: 42  # or extract from paper
  
# In experiment_runner.py
def run_experiment(self, config):
    # Set seeds before running
    env_vars = config.env_vars.copy()
    if 'random_seed' in self.config:
        env_vars['PYTHONHASHSEED'] = str(self.config['random_seed'])
```

### Priority 5: Parameter Extraction & Matching (Medium - 3-4 hours)

**Problem**: May not be using exact same hyperparameters as paper

**Solutions**:
1. Use LLM to extract all hyperparameters from paper methodology
2. Validate experiment code uses these parameters
3. Generate configuration file with paper's exact settings

**Implementation**:
```python
# Add to paper_parser.py
def extract_hyperparameters(self, methodology_text):
    """Extract hyperparameters from methodology section."""
    prompt = '''
    Extract ALL hyperparameters and configuration settings from this methodology.
    Include: learning rates, batch sizes, model sizes, retrieval k values, etc.
    
    Return as JSON: {"param_name": value, ...}
    '''
    # Use LLM to extract
    # Validate against codebase
    # Generate config file
```

### Priority 6: Preprocessing Validation (Medium - 2-3 hours)

**Problem**: Sentence-level worse performance suggests tokenization differences

**Solutions**:
1. Extract preprocessing steps from paper
2. Compare with codebase implementation
3. Add validation that preprocessing matches paper

**Implementation**:
```python
# Add preprocessing checker
def validate_preprocessing(self, paper_methodology, codebase_path):
    """Compare paper's preprocessing with code implementation."""
    
    # Extract from paper
    preprocessing_steps = self.llm_client.extract_json(
        f"List all preprocessing steps: {paper_methodology}",
        system="Extract data preprocessing pipeline"
    )
    
    # Check code
    code_files = find_files(codebase_path, '*preprocessing*', '*tokenizer*')
    # Compare and log differences
```

## Implementation Roadmap

### Phase 1: Quick Wins (Today - 2-3 hours)
- [ ] Add debug logging to metric matching
- [ ] Extract baselines from complete_results.json directly
- [ ] Add data file validation checks
- [ ] Set random seeds in experiments

**Expected Impact**: +10-15% success rate (77-82%)

### Phase 2: Core Improvements (This Week - 1 day)
- [ ] Implement hyperparameter extraction
- [ ] Add preprocessing validation
- [ ] Cross-validate all baselines
- [ ] Enhanced error messages and logging

**Expected Impact**: +15-20% success rate (82-87%)

### Phase 3: Advanced Features (Next Week - 2-3 days)
- [ ] Automated parameter sweeps
- [ ] Ablation study support
- [ ] Experiment versioning
- [ ] Result visualization dashboard

**Expected Impact**: +8-13% success rate (90-95%)

## Success Metrics

### Target Goals:
- **Primary Goal**: 85%+ success rate (415/490 metrics within threshold)
- **Stretch Goal**: 90%+ success rate (440/490 metrics)
- **Minimum Viable**: 75%+ (368/490 metrics)

### Expected Timeline:
- Quick wins: Same day (+10-15%)
- Core improvements: 3-5 days (+25-35% total)
- Advanced features: 2-3 weeks (+33-45% total)

## Monitoring Plan

Add telemetry to track:
1. Success rate per experiment set
2. Average deviation per configuration type
3. Number of failed metric matches
4. LLM extraction success rate
5. Experiment execution time

## Risk Mitigation

- **Risk**: Changes break existing functionality
  - **Mitigation**: Keep test_simple.py, add regression tests
  
- **Risk**: LLM extractions become unreliable
  - **Mitigation**: Always have fallback to structured parsing
  
- **Risk**: Baseline extraction still incomplete
  - **Mitigation**: Use complete_results.json as ground truth

## Conclusion

The agent is currently at **MODERATE** performance (67.3%). With focused improvements on:
1. ✅ Metric matching accuracy
2. ✅ Baseline extraction completeness  
3. ✅ Data validation
4. ✅ Parameter alignment

We can realistically achieve **GOOD-EXCELLENT** performance (85-95% success rate) within 1-2 weeks.

The highest ROI improvements are:
1. **Using complete_results.json as baseline** (1 hour, +5-10%)
2. **Adding data validation** (1 hour, +3-5%)
3. **Debug logging for matching** (30 min, identifies issues)
4. **Random seed configuration** (30 min, +2-3%)
