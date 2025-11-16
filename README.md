# EVALLab:
A local LLM agent that reproduces research paper experiments currently using Llama3. Features a clean 4-stage architecture with 94.2% metric reproduction accuracy currently on the target 'Decontextualization' Research Paper.

## Agent Architecture:
The agent uses the following **4-stage pipeline**:
1. **Stage 1: Paper Parser** - Extracts text, (Abstract, methodology, experiment, figures) from the input Research Paper
2. **Stage 2: Repo Retriever** - Finds codebase (local → GitHub priority)
3. **Stage 3: Experiment Executor** - Analyses Paper codebase + runs experiments from context of first 2 stages
4. **Stage 4: Result Evaluator** - Compares results + generates visualizations from baseline paper authors results, returns plots, HTML Visualisation dashboard, csv and log data, as-well as full command line logging.

## Installation & How to Run EVALLab:
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

3. **Place paper code in the correct directory if paper is known not to have an embedded linked repo link, as in the current input target case:**
   - Put the `supplementary_material` directory inside `paper_source_code/`
   - Final structure: `paper_source_code/supplementary_material/`

4. **Run EVALLab:**
   ```bash
   python3 run_EVALLab.py
   ```
