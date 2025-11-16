# Local Research Paper Reproduction Agent 🔬

A local LLM-powered agent that reproduces research paper experiments using Ollama. Features a clean 4-stage architecture with 94.2% metric reproduction accuracy.

## Architecture

The agent uses a **4-stage pipeline**:

1. **Stage 1: Paper Parser** - Extracts text, figures, and GitHub URLs from PDFs
2. **Stage 2: Repo Retriever** - Finds code (local → GitHub priority)
3. **Stage 3: Experiment Executor** - Analyzes codebase + runs experiments
4. **Stage 4: Result Evaluator** - Compares results + generates visualizations

## Quick Start

1. **Install and start Ollama with llama3:**
   ```bash
   ollama serve
   ollama pull llama3
   ```

2. **Set up Python environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Place paper code in the correct directory:**
   - Put the `supplementary_material` directory inside `paper_source_code/`
   - Final structure: `paper_source_code/supplementary_material/`

4. **Run the agent:**
   ```bash
   python3 run.py
   ```

## Performance

**94.2% success rate** (339/360 metrics matched) with 1.05% mean deviation from baseline results.