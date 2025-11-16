import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main import ReproductionAgent


def main():
    """Run agent with auto-detection."""
    print("="*70)
    print("🔬 LOCAL RESEARCH PAPER REPRODUCTION AGENT")
    print("="*70)
    print("\nAuto-detecting paper and codebase from workspace...")
    print("\nExpected structure:")
    print("  📂 paper/                    # Contains PDF paper")
    print("  📂 paper_source_code/")
    print("      └── supplementary_material/")
    print("          ├── code/            # Main code")
    print("          ├── idea_generation/")
    print("          └── references/")
    print("="*70)
    
    # Check workspace structure
    workspace_root = Path(__file__).parent
    paper_dir = workspace_root / "paper"
    code_dir = workspace_root / "paper_source_code" / "supplementary_material"
    
    print("\n📍 Checking workspace structure...")
    
    if not paper_dir.exists():
        print(f"❌ ./paper/ directory not found!")
        print("   Please create it and add your research paper PDF")
        return 1
    
    pdf_files = list(paper_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in ./paper/")
        print("   Please add your research paper PDF to ./paper/")
        return 1
    
    print(f"✓ Found paper: {pdf_files[0].name}")
    
    if not code_dir.exists():
        print(f"⚠️  ./paper_source_code/supplementary_material/ not found!")
        print("   Will search for GitHub URLs in paper...")
    else:
        print(f"✓ Found code directory: {code_dir}")
    
    # Check Ollama
    print("\n🔍 Checking Ollama...")
    from llm_client import OllamaClient
    client = OllamaClient()
    
    if not client.is_available():
        print("❌ Ollama is not running!")
        print("\n   Please start Ollama in another terminal:")
        print("   $ ollama serve")
        print("\n   And ensure you have a model installed:")
        print("   $ ollama pull llama3")
        return 1
    
    models = client.list_models()
    if not models:
        print("❌ No Ollama models found!")
        print("\n   Please pull a model:")
        print("   $ ollama pull llama3")
        return 1
    
    print(f"✓ Ollama is running (models: {', '.join(models[:3])})")
    
    # Run agent
    print("\n" + "="*70)
    print("🚀 Starting reproduction workflow...")
    print("="*70 + "\n")
    
    try:
        agent = ReproductionAgent()
        results = agent.run()  # No arguments - auto-detect!
        
        if 'error' in results:
            print(f"\n❌ Error: {results['error']}")
            return 1
        
        print("\n" + "="*70)
        print("✅ WORKFLOW COMPLETED!")
        print("="*70)
        print(f"\n📊 Check outputs/ directory for results")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
