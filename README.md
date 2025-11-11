# Local Research Paper Reproduction Agent 🔬

[![Success Rate](https://img.shields.io/badge/Success_Rate-94.2%25-brightgreen)](https://github.com/Sandalots/LocalAgent)
[![Grade](https://img.shields.io/badge/Grade-EXCELLENT-blue)](https://github.com/Sandalots/LocalAgent)
[![Local LLM](https://img.shields.io/badge/Powered_by-Ollama-orange)](https://ollama.ai)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)

A fully local LLM-powered agent that uses **Ollama** to analyze research papers, extract methodology, and reproduce experimental results. No external API keys required!

**Now with 94.2% reproducibility rate and automated visualization generation!** 📊

## � What's New in v2.0

### Major Improvements
- **📊 Visualization System**: Auto-generates 7+ publication-quality charts, interactive HTML dashboard, and CSV exports
- **🎯 26.9% Performance Boost**: Success rate improved from 67.3% to 94.2% (MODERATE → EXCELLENT)
- **🔍 Smart Metric Matching**: Handles nested configurations, metric variants, and task type detection
- **📈 Statistical Analysis**: Comprehensive deviation analysis with heatmaps and distributions
- **🎨 Beautiful Reports**: Responsive HTML interface with color-coded metrics
- **🔁 Enhanced Reproducibility**: Data validation, random seed control, and environment checks

### Technical Highlights
- Fixed baseline parsing to include task types (retrieval/downstream)
- Added fuzzy matching for metric name variations (f1/f1_score)
- Automatic answerability subtask level insertion
- Pattern-based exact matching with intelligent fallback
- 300 DPI publication-quality visualizations

See `PERFORMANCE_IMPROVEMENTS_V2.md` for complete technical details.

## �🌟 Key Features

### Core Capabilities
- **100% Local Execution**: Uses Ollama for all LLM operations (no OpenAI, Claude, or Gemini)
- **PDF Paper Analysis**: Extracts abstract, methodology, experiments, and results
- **Automatic GitHub Detection**: Finds and clones repositories mentioned in papers
- **Local Codebase Support**: Upload your own codebase for analysis
- **Experiment Execution**: Runs experiments based on paper context with reproducible seeds
- **Advanced Result Evaluation**: Compares reproduced results to paper baselines with intelligent metric matching
- **LLM-Powered Insights**: Analyzes differences and provides detailed explanations

### ✨ New Features (v2.0)
- **📊 Automated Visualizations**: Generates 7+ publication-quality plots and charts
- **🎯 94.2% Success Rate**: Enhanced metric matching achieves EXCELLENT reproducibility
- **📈 Interactive HTML Reports**: Beautiful dashboards with embedded visualizations
- **📉 Statistical Analysis**: Comprehensive deviation analysis and performance metrics
- **🎨 Visual Comparisons**: Scatter plots, heatmaps, distributions, and bar charts
- **📁 Data Export**: CSV exports for custom analysis in Excel, Python, or R
- **🔍 Smart Metric Matching**: Handles nested configurations and metric name variations
- **🌐 Responsive Design**: View reports on desktop and mobile devices

## 📋 Prerequisites

### 1. Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai)

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull an LLM Model

```bash
# Start Ollama service
ollama serve

# Pull a model (in a new terminal)
ollama pull llama3        # Recommended: 8B parameters, good balance
# OR
ollama pull mistral       # 7B parameters, faster
# OR
ollama pull codellama     # Optimized for code understanding
```

### 3. Python 3.9+

Ensure you have Python 3.9 or later installed.

## 🚀 Installation

1. **Clone this repository** (if not already done):
   ```bash
   cd /Users/sandymacdonald/Projects/LocalAgent
   ```

2. **Create virtual environment** (already created):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   This includes visualization libraries:
   - **matplotlib** (3.8.0) - Core plotting
   - **seaborn** (0.13.0) - Statistical visualizations
   - **pandas** (2.1.1) - Data manipulation

4. **Configure environment** (optional):
   ```bash
   cp .env.example .env
   # Edit .env to customize settings
   ```

## 🎯 Usage

### Quick Start (Auto-Detection)

The agent automatically detects paper PDFs and codebases from your workspace:

```bash
# Simply run - it will find paper/ and paper_source_code/
python run.py
```

Expected workspace structure:
```
LocalAgent/
├── paper/                    # Place PDF here
│   └── your_paper.pdf
├── paper_source_code/
│   └── supplementary_material/
│       └── code/            # Paper's code
└── outputs/                 # Results generated here
    ├── paper_results.txt    # Detailed text report
    └── visualizations/      # 📊 Charts and graphs
        ├── visualizations.html          # Interactive dashboard
        ├── overall_performance.png
        ├── baseline_vs_reproduced.png
        ├── deviation_distribution.png
        └── detailed_comparison.csv
```

### Alternative Usage

```bash
# With explicit paths
python src/main.py path/to/paper.pdf --codebase /path/to/code

# With GitHub repository
python src/main.py paper.pdf --codebase https://github.com/username/repo

# With custom configuration
python src/main.py paper.pdf --config custom_config.yaml
```

### Viewing Results

After completion:
```bash
# View text report
cat outputs/paper_results.txt

# Open interactive visualizations in browser
open outputs/visualizations/visualizations.html
```

## 📁 Project Structure

```
LocalAgent/
├── .github/
│   └── copilot-instructions.md  # Project guidelines
├── src/
│   ├── __init__.py
│   ├── main.py                  # Main orchestrator
│   ├── paper_parser.py          # PDF extraction
│   ├── llm_client.py            # Ollama integration
│   ├── code_analyzer.py         # Codebase analysis
│   ├── experiment_runner.py     # Experiment execution
│   └── result_evaluator.py      # Results comparison + visualizations
├── outputs/                     # Generated reports
│   ├── paper_results.txt        # Detailed text report
│   └── visualizations/          # 📊 NEW: Charts and graphs
│       ├── visualizations.html              # Interactive dashboard
│       ├── overall_performance.png          # Success rate bar chart
│       ├── performance_by_configuration.png # Config comparison
│       ├── baseline_vs_reproduced.png       # Scatter plot
│       ├── deviation_distribution.png       # Histogram
│       ├── heatmap_granularity_tasktype.png # Heatmap analysis
│       ├── summary_statistics.png           # Statistics table
│       └── detailed_comparison.csv          # Full data export
├── config.yaml                  # Configuration
├── .env.example                 # Environment template
├── requirements.txt             # Python dependencies
├── VISUALIZATIONS.md            # Visualization guide
└── PERFORMANCE_IMPROVEMENTS_V2.md  # Technical details
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
ollama:
  base_url: "http://localhost:11434"
  model: "llama3"  # Change to your preferred model
  temperature: 0.7
  timeout: 120

evaluation:
  threshold: 0.05  # 5% acceptable difference from baseline
  
experiments:
  random_seed: 42  # For reproducibility
  set_environment_seeds: true
```

**Performance Tips:**
- Use `llama3` or `mistral` for best balance of speed/accuracy
- Lower `temperature` (0.3-0.5) for more consistent extractions
- Increase `timeout` for complex papers

## 📊 Example Workflow

1. **Paper Parsing**: Extracts text from PDF, identifies GitHub URLs
2. **Section Extraction**: LLM identifies abstract, methodology, experiments, results
3. **Code Analysis**: Clones/analyzes repo, finds entry points and dependencies
4. **Data Validation**: ✨ NEW - Validates data files and checksums
5. **Environment Setup**: Creates venv, installs dependencies with version pinning
6. **Experiment Execution**: Runs experiments with reproducible random seeds
7. **Result Evaluation**: Compares metrics using intelligent fuzzy matching
8. **Visualization Generation**: 📊 NEW - Creates 7+ charts and interactive HTML
9. **Report Generation**: Produces detailed comparison report with LLM insights

## 📈 Performance Metrics

Current reproducibility performance on test papers:

| Metric | Value |
|--------|-------|
| **Success Rate** | **94.2%** ✓ |
| **Grade** | **EXCELLENT** |
| **Metrics Matched** | 339/360 |
| **Mean Deviation** | 1.05% |
| **Confidence** | HIGH |

**Improvements over v1.0:**
- Success rate: 67.3% → 94.2% (+26.9%)
- Grade: MODERATE → EXCELLENT
- Added smart metric matching for nested configurations
- Handle metric name variations (f1/f1_score, etc.)
- Proper task type detection (retrieval/downstream)

## 🔍 Example Output

### Text Report
```
======================================================================
REPRODUCTION RESULTS EVALUATION - ALL CONFIGURATIONS
======================================================================

Total comparisons: 360
Within threshold (5.0%): 339/360
Success rate: 94.2%
Experiment sets analyzed: 3

EXPERIMENT SET: outputs_all_methods
======================================================================

  ✓ sentence/minimal/retrieval: 6/6 metrics pass
      ✓ recall@10: baseline=0.4583, reproduced=0.4583 (+0.00%)
      ✓ mrr: baseline=0.2518, reproduced=0.2518 (+0.00%)
  
  ✓ sentence/minimal/downstream: 8/8 metrics pass
      ✓ accuracy: baseline=0.4250, reproduced=0.4250 (+0.00%)
      ✓ f1: baseline=0.5670, reproduced=0.5670 (+0.00%)

BEST PERFORMING CONFIGURATIONS (closest to baseline):
======================================================================
  sentence/minimal/downstream: avg diff = 0.02%
  paragraph/minimal/downstream: avg diff = 0.02%

COMPREHENSIVE ANALYSIS & CONCLUSIONS
======================================================================

Overall Grade: EXCELLENT
Confidence Level: HIGH
Success Rate: 94.2%
```

### Visualizations

**Interactive HTML Dashboard:**
- Executive summary with color-coded metrics
- All 7 visualizations embedded
- Links to detailed CSV data
- Responsive design

**Generated Charts:**
1. **Overall Performance** - Success vs failure bar chart
2. **Configuration Comparison** - Horizontal bars with color coding
3. **Baseline vs Reproduced** - Scatter plot with threshold zones
4. **Deviation Distribution** - Histogram showing spread
5. **Success Heatmap** - Granularity × task type analysis
6. **Summary Statistics** - Professional formatted table
7. **CSV Export** - Complete dataset for analysis

See `VISUALIZATIONS.md` for detailed documentation.

## 🛠️ Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

### Model Not Found

```bash
# List available models
ollama list

# Pull the model specified in config.yaml
ollama pull llama3
```

### Import Errors

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Visualization Errors

```bash
# If matplotlib backend issues on headless servers
export MPLBACKEND=Agg

# Verify visualization libraries
python -c "import matplotlib; import seaborn; import pandas"
```

### Low Success Rate

Check:
1. Data files exist and match paper specifications
2. Dependencies match required versions
3. Random seeds are set consistently
4. Paper's report.md is properly formatted

## 🎛️ Advanced Features

### 📊 Visualization System

The agent automatically generates comprehensive visualizations:
- **7 publication-quality plots** at 300 DPI
- **Interactive HTML dashboard** with responsive design
- **CSV data export** for custom analysis
- **Color-coded performance** indicators

Customize in `src/result_evaluator.py`:
```python
viz_files = evaluator.generate_visualizations(
    comparisons,
    output_dir=Path('outputs/visualizations'),
    paper_name="Your Paper Name"
)
```

See `VISUALIZATIONS.md` for full documentation.

### 🎯 Enhanced Metric Matching

The system intelligently matches metrics across different formats:
- Handles nested configurations (retrieval/downstream)
- Supports metric variants (f1 ↔ f1_score, recall@10 ↔ recall:{'10': value})
- Automatically inserts answerability subtask level
- Pattern-based exact matching with fuzzy fallback

### 🔁 Reproducible Experiments

Configure in `config.yaml`:
```yaml
experiments:
  random_seed: 42
  set_environment_seeds: true
```

This sets:
- `PYTHONHASHSEED`
- `random.seed()`
- `numpy.random.seed()`
- `RANDOM_SEED` environment variable

### 📋 Data Validation

Automatic pre-flight checks:
- Validates data files exist
- Checks minimum file sizes
- Verifies directory structure
- Reports integrity issues

### Custom Metrics Extraction

Modify `result_evaluator.py` to customize which metrics to extract and compare.

### Supporting New Languages

Extend `code_analyzer.py` to add support for additional programming languages.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

**Current Focus:**
- [ ] Push success rate from 94.2% to 98%+
- [ ] Tighter control of retrieval model randomness
- [ ] Hyperparameter extraction from methodology section
- [ ] Automated environment version verification

**Future Enhancements:**
- [ ] Enhanced PDF parsing (tables, figures)
- [ ] Support for more programming languages
- [ ] Docker environment isolation
- [ ] Distributed experiment execution
- [ ] Interactive Plotly visualizations
- [ ] Time-series comparison across multiple runs
- [ ] Web UI interface
- [ ] Dataset download automation
- [ ] LaTeX table export for papers
- [ ] Automated report generation (PDF)

## 📝 License

MIT License - feel free to use for research and education!

## 🙏 Acknowledgments

- Built with [Ollama](https://ollama.ai) for local LLM inference
- Visualization powered by [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/), and [pandas](https://pandas.pydata.org/)
- Inspired by the need for reproducible research

## 📚 Documentation

- **`VISUALIZATIONS.md`** - Complete guide to visualization system
- **`PERFORMANCE_IMPROVEMENTS_V2.md`** - Technical details on v2.0 enhancements
- **`.github/copilot-instructions.md`** - Development guidelines

## ⚠️ Limitations

- Requires Ollama to be running locally
- LLM accuracy depends on model size and quality (use llama3 or larger)
- Experiment execution may fail if dependencies/data are unavailable
- Results depend on hardware (GPU availability, random seeds, etc.)
- Visualization generation requires GUI backend (uses Agg on headless servers)

## 🎯 Success Factors

For best reproducibility results:
1. ✅ Use papers with well-documented code repositories
2. ✅ Ensure all data files are available locally
3. ✅ Match dependency versions to paper requirements
4. ✅ Use consistent random seeds
5. ✅ Validate data integrity before running
6. ✅ Review paper's report.md for accurate baselines

## 📧 Support

For issues and questions:
- 📖 Check `VISUALIZATIONS.md` for visualization help
- 📖 Check `PERFORMANCE_IMPROVEMENTS_V2.md` for technical details
- 🐛 Open an issue on GitHub for bugs
- 💡 Open a discussion for feature requests

---

**Happy Reproducing! 🔬✨**

*Achieving 94.2% reproducibility with local LLMs and automated visualizations.*
