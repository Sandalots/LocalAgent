# Local Research Paper Reproduction Agent 🔬

A fully local LLM-powered agent that uses **Ollama** to analyze research papers, extract methodology, and reproduce experimental results. No external API keys required!

## 🌟 Features

- **100% Local Execution**: Uses Ollama for all LLM operations (no OpenAI, Claude, or Gemini)
- **PDF Paper Analysis**: Extracts abstract, methodology, experiments, and results
- **Automatic GitHub Detection**: Finds and clones repositories mentioned in papers
- **Local Codebase Support**: Upload your own codebase for analysis
- **Experiment Execution**: Runs experiments based on paper context
- **Result Evaluation**: Compares reproduced results to paper baselines
- **LLM-Powered Insights**: Analyzes differences and provides explanations

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

4. **Configure environment** (optional):
   ```bash
   cp .env.example .env
   # Edit .env to customize settings
   ```

## 🎯 Usage

### Basic Usage

```bash
# Start Ollama (if not already running)
ollama serve

# In a new terminal, activate venv and run
source .venv/bin/activate
python src/main.py path/to/paper.pdf
```

### With Explicit Codebase

```bash
# Use a specific GitHub repository
python src/main.py paper.pdf --codebase https://github.com/username/repo

# Or use a local codebase
python src/main.py paper.pdf --codebase /path/to/local/code
```

### With Custom Configuration

```bash
python src/main.py paper.pdf --config custom_config.yaml
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
│   └── result_evaluator.py      # Results comparison
├── outputs/                     # Generated reports
├── config.yaml                  # Configuration
├── .env.example                 # Environment template
└── requirements.txt             # Python dependencies
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
```

## 📊 Example Workflow

1. **Paper Parsing**: Extracts text from PDF, identifies GitHub URLs
2. **Section Extraction**: LLM identifies abstract, methodology, experiments, results
3. **Code Analysis**: Clones/analyzes repo, finds entry points and dependencies
4. **Environment Setup**: Creates venv, installs dependencies
5. **Experiment Execution**: Runs experiment scripts, captures outputs
6. **Result Evaluation**: Compares metrics to paper baselines
7. **Report Generation**: Produces detailed comparison report with LLM insights

## 🔍 Example Output

```
======================================================================
REPRODUCTION RESULTS EVALUATION
======================================================================

Total metrics compared: 3
Within threshold (5%): 2/3
Success rate: 66.7%

DETAILED COMPARISON:
----------------------------------------------------------------------

Metric: accuracy
  Baseline:    0.9450
  Reproduced:  0.9380
  Difference:  -0.0070 (-0.74%)
  Status:      ✓ PASS

Metric: f1_score
  Baseline:    0.8820
  Reproduced:  0.8650
  Difference:  -0.0170 (-1.93%)
  Status:      ✓ PASS

Metric: precision
  Baseline:    0.9100
  Reproduced:  0.8450
  Difference:  -0.0650 (-7.14%)
  Status:      ✗ FAIL
```

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

## 🎛️ Advanced Features

### Custom Metrics Extraction

Modify `result_evaluator.py` to customize which metrics to extract and compare.

### Supporting New Languages

Extend `code_analyzer.py` to add support for additional programming languages.

### Custom Experiment Configurations

Implement LLM-powered experiment configuration in `experiment_runner.py` for smarter script selection.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Enhanced PDF parsing (tables, figures)
- [ ] Support for more programming languages
- [ ] Docker environment isolation
- [ ] Distributed experiment execution
- [ ] Web UI interface
- [ ] Dataset download automation

## 📝 License

MIT License - feel free to use for research and education!

## 🙏 Acknowledgments

- Built with [Ollama](https://ollama.ai) for local LLM inference
- Inspired by the need for reproducible research

## ⚠️ Limitations

- Requires Ollama to be running locally
- LLM accuracy depends on model size and quality
- Experiment execution may fail if dependencies/data are unavailable
- Results depend on hardware (GPU availability, etc.)

## 📧 Support

For issues and questions, please open an issue on GitHub.

---

**Happy Reproducing! 🔬✨**
