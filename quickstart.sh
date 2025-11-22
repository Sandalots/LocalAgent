#!/bin/bash
# Quick start script for EVALLab
set -e

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# check if papers/codebases/supplementary_materials directory exists
if [ ! -d "papers/codebases/supplementary_material" ]; then
# tell user to manually download the decontextualization codebase and place it here.
  echo "Please download the Decontextualization codebase from https://openreview.net/forum?id=cK8YYMc65B#discussion and place it in the papers/codebases/ directory."

# else tell user that the codebase is found
else
  echo "Decontextualization codebase found."
fi

# tell user to run python3 run_EVALLab.py papers/decontextualisation.pdf --code ./papers/codebases/supplementary_material
echo "To run EVALLab on the Decontextualization paper, use the following command:"
echo "python3 run_EVALLab.py papers/decontextualisation.pdf --code ./papers/codebases/supplementary_material/"

# check if papers/codebases/suplementary_materials exists
if [ ! -d "papers/codebases/supplementary_material" ]; then
    echo "Note: The manually placed Decontextualization codebase is required to run the example."

# else run python3 run_EVALLab.py papers/decontextualisation.pdf --code ./papers/codebases/supplementary_material
else
    python3 run_EVALLab.py papers/decontextualisation.pdf --code ./papers/codebases/supplementary_material/
fi
