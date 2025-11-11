#!/usr/bin/env python3
"""
Interactive script to run the reproduction agent with manual inputs.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main import ReproductionAgent


def get_paper_path():
    """Prompt user for paper PDF path."""
    while True:
        paper_input = input("\n📄 Enter path to research paper PDF: ").strip()
        
        if not paper_input:
            print("❌ Path cannot be empty!")
            continue
        
        paper_path = Path(paper_input).expanduser()
        
        if not paper_path.exists():
            print(f"❌ File not found: {paper_path}")
            retry = input("Try again? (y/n): ").lower()
            if retry != 'y':
                return None
            continue
        
        if paper_path.suffix.lower() != '.pdf':
            print(f"⚠️  Warning: File doesn't have .pdf extension")
            proceed = input("Continue anyway? (y/n): ").lower()
            if proceed != 'y':
                continue
        
        return paper_path


def get_codebase_source():
    """Prompt user for codebase source (GitHub URL or local path)."""
    print("\n🔗 Codebase Source Options:")
    print("  1. GitHub repository URL")
    print("  2. Local directory path")
    print("  3. Let agent search paper for GitHub links (auto)")
    
    while True:
        choice = input("\nChoose option (1/2/3): ").strip()
        
        if choice == '1':
            url = input("Enter GitHub URL: ").strip()
            if url:
                return url
            print("❌ URL cannot be empty!")
            
        elif choice == '2':
            path_input = input("Enter local codebase path: ").strip()
            if not path_input:
                print("❌ Path cannot be empty!")
                continue
            
            local_path = Path(path_input).expanduser()
            if not local_path.exists():
                print(f"❌ Directory not found: {local_path}")
                retry = input("Try again? (y/n): ").lower()
                if retry != 'y':
                    return None
                continue
            
            if not local_path.is_dir():
                print(f"❌ Not a directory: {local_path}")
                continue
            
            return str(local_path.absolute())
            
        elif choice == '3':
            print("✓ Agent will search paper for GitHub URLs")
            return None
            
        else:
            print("❌ Invalid choice! Please enter 1, 2, or 3")


def confirm_settings(paper_path, codebase_source):
    """Display settings and confirm."""
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f"📄 Paper: {paper_path}")
    
    if codebase_source:
        if codebase_source.startswith('http'):
            print(f"🔗 Codebase: {codebase_source} (GitHub)")
        else:
            print(f"📁 Codebase: {codebase_source} (Local)")
    else:
        print("🔍 Codebase: Auto-detect from paper")
    
    print("="*70)
    
    confirm = input("\nProceed with these settings? (y/n): ").lower()
    return confirm == 'y'


def main():
    """Run interactive reproduction agent."""
    print("="*70)
    print("🔬 LOCAL RESEARCH PAPER REPRODUCTION AGENT")
    print("="*70)
    print("\nThis tool will:")
    print("  1. Parse your research paper PDF")
    print("  2. Extract methodology and experiments")
    print("  3. Analyze the codebase")
    print("  4. Run experiments")
    print("  5. Compare results to baseline metrics")
    print("\n⚠️  Make sure Ollama is running: ollama serve")
    print("="*70)
    
    # Get paper path
    paper_path = get_paper_path()
    if not paper_path:
        print("\n❌ Cancelled")
        return 1
    
    # Get codebase source
    codebase_source = get_codebase_source()
    
    # Confirm settings
    if not confirm_settings(paper_path, codebase_source):
        print("\n❌ Cancelled")
        return 1
    
    # Initialize and run agent
    print("\n🚀 Starting reproduction workflow...")
    print("="*70 + "\n")
    
    try:
        agent = ReproductionAgent()
        results = agent.run(paper_path, codebase_source)
        
        if 'error' in results:
            print(f"\n❌ Error: {results['error']}")
            return 1
        
        print("\n" + "="*70)
        print("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n📊 Results saved to: outputs/{paper_path.stem}_results.txt")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
