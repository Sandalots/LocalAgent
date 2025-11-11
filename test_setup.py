#!/usr/bin/env python3
"""
Quick test script to verify Ollama connection and basic functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from llm_client import OllamaClient, LLMConfig


def test_ollama_connection():
    """Test connection to Ollama."""
    print("Testing Ollama connection...")
    
    client = OllamaClient()
    
    if not client.is_available():
        print("❌ Ollama is not available!")
        print(f"   Please ensure Ollama is running at {client.base_url}")
        print("   Run: ollama serve")
        return False
    
    print("✅ Ollama is available")
    
    # List models
    models = client.list_models()
    if not models:
        print("⚠️  No models found!")
        print("   Please pull a model first:")
        print("   ollama pull llama3")
        return False
    
    print(f"✅ Found {len(models)} model(s):")
    for model in models:
        print(f"   - {model}")
    
    # Test generation
    print("\nTesting text generation...")
    try:
        response = client.generate(
            "Say 'Hello from Ollama!' in one sentence.",
            temperature=0.1
        )
        print(f"✅ Generation works!")
        print(f"   Response: {response[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False


if __name__ == '__main__':
    success = test_ollama_connection()
    sys.exit(0 if success else 1)
