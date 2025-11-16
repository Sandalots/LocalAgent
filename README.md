# Local Research Paper Reproduction Agent 🔬

A local LLM-powered agent that reproduces research paper experiments using Ollama, currently reproduces the 'Decontextualization' paper experiments.

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

3. **Run the agent:**
   ```bash
   python3 run.py
   ```
