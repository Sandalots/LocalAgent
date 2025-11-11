#!/bin/bash

# Quick Start Script for Local Research Paper Reproduction Agent

set -e

echo "🚀 Local Research Paper Reproduction Agent - Quick Start"
echo "=========================================================="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed!"
    echo "   Please install from: https://ollama.ai"
    echo "   macOS: brew install ollama"
    exit 1
fi

echo "✅ Ollama is installed"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama is not running"
    echo "   Starting Ollama in the background..."
    ollama serve > /dev/null 2>&1 &
    sleep 2
    echo "✅ Ollama started"
else
    echo "✅ Ollama is running"
fi

# Check for models
echo ""
echo "Checking for available models..."
MODELS=$(ollama list 2>/dev/null | tail -n +2)

if [ -z "$MODELS" ]; then
    echo "⚠️  No models found!"
    echo "   Pulling llama3 (this may take a few minutes)..."
    ollama pull llama3
    echo "✅ llama3 downloaded"
else
    echo "✅ Available models:"
    ollama list
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo ""
    echo "Activating virtual environment..."
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo ""
    echo "⚠️  Virtual environment not found"
    echo "   Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "✅ Virtual environment created and activated"
    
    echo ""
    echo "Installing dependencies..."
    pip install -r requirements.txt > /dev/null 2>&1
    echo "✅ Dependencies installed"
fi

echo "Usage examples:"
echo "  python src/main.py paper.pdf"
echo "  python src/main.py paper.pdf --codebase https://github.com/user/repo"
echo ""
echo "For more info, see README.md"
echo "=========================================================="
