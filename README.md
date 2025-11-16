# EVALLab
A local LLM-powered agent that reproduces research paper experiments using Ollama3. Features a clean 4-stage architecture with 94.2% metric reproduction accuracy currently on the target 'Decontextualization' Research Paper.

## Architecture
The agent uses the following **4-stage pipeline**:

1. **Stage 1: Paper Parser** - Extracts text, figures, and GitHub URLs from Research Paper PDFs
2. **Stage 2: Repo Retriever** - Finds codebase (local → GitHub priority)
3. **Stage 3: Experiment Executor** - Analyzes codebase + runs experiments
4. **Stage 4: Result Evaluator** - Compares results + generates visualizations from baseline paper authors results

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
   python3 run_EVALLab.py
   ```

## Performance
**~94.2% success rate** (339/360 metrics matched) with 1.05% mean deviation from baseline results currently when fed the 'Decontextualization' paper.
