# 🚀 Quick Start Guide

## Your Workspace Structure

```
LocalAgent/
├── paper/                              # 📄 Put your research paper PDF here
│   └── 300_Decontextualization_Everyw.pdf
├── paper_source_code/                  # 💻 Source code for the paper
│   └── supplementary_material/
│       ├── code/                       # Main experiment code
│       ├── idea_generation/            # Research ideas
│       └── references/                 # Reference management
└── src/                                # 🤖 Agent code (don't modify)
```

## ✅ Setup Complete!

Ollama is running with **llama3** model! ✨

## 🎯 How to Run

### Option 1: Auto-Detect Everything (Simplest!)

Just run:
```bash
source .venv/bin/activate
python run.py
```

This will automatically:
- ✅ Find the PDF in `./paper/`
- ✅ Find the code in `./paper_source_code/supplementary_material/`
- ✅ Parse the paper with LLM
- ✅ Run experiments
- ✅ Compare results to baselines

### Option 2: Use Main CLI

```bash
source .venv/bin/activate

# Auto-detect both paper and code
python src/main.py

# Specify paper path
python src/main.py paper/your_paper.pdf

# Specify both paper and codebase
python src/main.py paper/your_paper.pdf --codebase paper_source_code/supplementary_material/code
```

### Option 3: Interactive Mode

```bash
source .venv/bin/activate
python run_interactive.py
```

You'll be prompted to:
1. Enter paper path (or it finds it automatically)
2. Choose codebase source (GitHub URL, local path, or auto-detect)

## 📝 What Happens When You Run

1. **Paper Analysis** 📄
   - Extracts text from PDF
   - Uses LLM to identify: abstract, methodology, experiments, results
   - Searches for GitHub URLs in paper

2. **Codebase Detection** 🔍
   - If GitHub URL found → clones repo
   - If no GitHub URL → uses `./paper_source_code/supplementary_material/`
   - Analyzes code structure, finds entry points, extracts dependencies

3. **Environment Setup** ⚙️
   - Creates virtual environment for experiments
   - Installs dependencies from `requirements.txt`

4. **Experiment Execution** 🧪
   - Runs main scripts (main.py, train.py, etc.)
   - Captures outputs and metrics

5. **Result Evaluation** 📊
   - Extracts baseline metrics from paper
   - Compares to reproduced results
   - Generates detailed report
   - LLM explains any differences

## 📂 Output

Results are saved to: `outputs/`

Example: `outputs/300_Decontextualization_Everyw_results.txt`

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
ollama:
  model: "llama3"  # Change to mistral, codellama, etc.
  temperature: 0.7

evaluation:
  threshold: 0.05  # 5% acceptable difference
```

## 💡 Examples

### Example 1: Your Current Setup
```bash
python run.py
# Auto-detects paper/ and paper_source_code/
```

### Example 2: Different Paper
```bash
python src/main.py /path/to/different/paper.pdf --codebase https://github.com/author/repo
```

### Example 3: Manual Code Directory
```bash
python src/main.py paper/my_paper.pdf --codebase /Users/me/code/experiment
```

## 🐛 Troubleshooting

### "Ollama is not running"
```bash
# In another terminal:
ollama serve
```

### "No models found"
```bash
ollama pull llama3
```

### "No PDF files found"
Make sure your paper is in `./paper/` directory:
```bash
ls paper/  # Should show your PDF
```

### "No codebase found"
Check the structure:
```bash
ls paper_source_code/supplementary_material/
# Should show: code/, idea_generation/, references/
```

## 🎓 Understanding Your Paper Structure

Based on your `paper_source_code/supplementary_material/ReadMe.md`:

- **Main experiments**: Run `main_local_all_new.py`
- **Oracle evaluation**: Use `--oracle` flag
- **Dependencies**: Listed in `requirements.txt`

The agent will automatically discover and run these!

## ⚡ Quick Test

Test Ollama connection:
```bash
python test_setup.py
```

Should output:
```
✅ Ollama is available
✅ Found 1 model(s):
   - llama3:latest
✅ Generation works!
```

## 🎉 You're Ready!

Try it now:
```bash
source .venv/bin/activate
python run.py
```

The agent will do the rest! 🚀
