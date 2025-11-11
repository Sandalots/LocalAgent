# Local Research Paper Reproduction Agent

This project uses Ollama for local LLM inference to analyze research papers and reproduce experiments.

## Project Guidelines

- Use Ollama models (llama3, mistral, etc.) for all LLM operations
- No external API keys (OpenAI, Claude, Gemini, etc.)
- Focus on reproducibility and local execution
- Parse PDFs to extract methodology and experiments
- Support GitHub repo scanning and local codebase upload
- Execute experiments based on paper context
- Evaluate results against baseline metrics

## Development Rules

- Keep all LLM interactions through Ollama API
- Use async operations for long-running tasks
- Implement proper error handling for experiment execution
- Log all steps for debugging and transparency
