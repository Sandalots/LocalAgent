"""===============================================================================
EVALLAB ENTRYPOINT: RUNS THE EVALLAB PIPELINE

This script is the main entrypoint for EVALLab. It parses command-line arguments,
initializes the agent's pipeline, and orchestrates the full research paper reproduction workflow, from retrieving and reading the paper, codebase, running experiments and returning evaulative results and benchmarks back to the EVALLab end-user.
===============================================================================
"""

from src.pipeline import ReproductionAgent
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def main():
    """Run agent with command-line arguments or auto-detection."""
    parser = argparse.ArgumentParser(
        description='EVALLab: Research Paper Reproduction Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python3 run_EVALLab.py                                    # Auto-detect or use Decontextualisation.pdf
    python3 run_EVALLab.py papers/MyPaper.pdf                  # Specific paper
    python3 run_EVALLab.py papers/MyPaper.pdf --code ./my_code # Specific paper + local code
  python3 run_EVALLab.py --code https://github.com/user/repo # Auto-detect paper + GitHub code
        '''
    )
    parser.add_argument('paper', nargs='?', type=str,
                        help='Path to research paper PDF (default: papers/Decontextualisation.pdf or auto-detect)')
    parser.add_argument(
        '--code', type=str, help='Path to local codebase or GitHub URL (default: auto-detect from papers_source_code/ or papers)')

    args = parser.parse_args()

    # Clear screen for fresh output (optional)
    print("\n" * 2)

    # Main EVALLab Header
    print("=" * 100)
    print("\033[1;96m" + "█" * 100 + "\033[0m")
    print("\033[1;96m█" + " " * 98 + "█\033[0m")
    print("\033[1;96m█" +
          "🔬 EVALLab: Research Paper Reproduction Agent".center(98) + "█\033[0m")
    print("\033[1;96m█" + " " * 98 + "█\033[0m")
    print("\033[1;96m" + "█" * 100 + "\033[0m")
    print("=" * 100)

    # Show command-line usage if arguments provided
    if args.paper or args.code:
        print("\n\033[1;93m📋 Command-Line Mode:\033[0m")
        if args.paper:
            print(f"  Paper: {args.paper}")
        if args.code:
            print(f"  Code: {args.code}")
        print()

    # Main EVALLab Header
    print("=" * 100)
    print("\033[1;96m" + "█" * 100 + "\033[0m")
    print("\033[1;96m█" + " " * 98 + "█\033[0m")
    print("\033[1;96m█" +
          "🔬 EVALLab: Research Paper Reproduction Agent".center(98) + "█\033[0m")
    print("\033[1;96m█" + " " * 98 + "█\033[0m")
    print("\033[1;96m" + "█" * 100 + "\033[0m")
    print("=" * 100)

    # What is EVALLab?
    print("\n\033[1;93m┌" + "─" * 98 + "┐\033[0mp")
    print("\033[1;93m│\033[0m" +
          "\033[1;97m WHAT IS EVALLAB?\033[0m".center(98) + "\033[1;93m│\033[0m")
    print("\033[1;93m├" + "─" * 98 + "┤\033[0m")
    print("\033[1;93m│\033[0m" + " An autonomous AI agent that reproduces computational experiments from research papers.".ljust(98) + "\033[1;93m│\033[0m")
    print("\033[1;93m│\033[0m" + " It parses PDFs, retrieves code, runs experiments, and validates results against baselines.".ljust(98) + "\033[1;93m│\033[0m")
    print("\033[1;93m│\033[0m" + " ".ljust(98) + "\033[1;93m│\033[0m")
    print("\033[1;93m│\033[0m" +
          " 🎯 Goal: Verify reproducibility of published research with 94.2% accuracy".ljust(98) + "\033[1;93m│\033[0m")
    print("\033[1;93m└" + "─" * 98 + "┘\033[0m")

    # 4-Stage Pipeline Overview
    print("\n\033[1;92m┌" + "─" * 98 + "┐\033[0m")
    print("\033[1;92m│\033[0m" +
          "\033[1;97m 4-STAGE REPRODUCTION PIPELINE\033[0m".center(98) + "\033[1;92m│\033[0m")
    print("\033[1;92m├" + "─" * 98 + "┤\033[0m")
    print("\033[1;92m│\033[0m" + " ".ljust(98) + "\033[1;92m│\033[0m")
    print("\033[1;92m│\033[0m" + "   \033[1;94m[1] PAPER PARSING\033[0m       → Extracts text, (Abstract, methodology, experiment, figures) from the input Research Paper".ljust(98) + "\033[1;92m│\033[0m")
    print("\033[1;92m│\033[0m" + "   \033[1;94m[2] CODE RETRIEVAL\033[0m      → Find code (local → GitHub) with smart detection".ljust(98) + "\033[1;92m│\033[0m")
    print("\033[1;92m│\033[0m" + "   \033[1;94m[3] EXPERIMENT RUN\033[0m      → Analyze codebase, setup environment, execute experiments".ljust(98) + "\033[1;92m│\033[0m")
    print("\033[1;92m│\033[0m" + "   \033[1;94m[4] RESULT EVALUATION\033[0m   → Compare metrics, generate visualizations & reports".ljust(98) + "\033[1;92m│\033[0m")
    print("\033[1;92m│\033[0m" + " ".ljust(98) + "\033[1;92m│\033[0m")
    print("\033[1;92m└" + "─" * 98 + "┘\033[0m")

    # What You Get
    print("\n\033[1;95m┌" + "─" * 98 + "┐\033[0m")
    print("\033[1;95m│\033[0m" +
          "\033[1;97m OUTPUT & DELIVERABLES\033[0m".center(98) + "\033[1;95m│\033[0m")
    print("\033[1;95m├" + "─" * 98 + "┤\033[0m")
    print("\033[1;95m│\033[0m" + " ".ljust(98) + "\033[1;95m│\033[0m")
    print("\033[1;95m│\033[0m" + "   📋 Complete Execution Log  → Full CLI output with all stages and colored metrics".ljust(98) + "\033[1;95m│\033[0m")
    print("\033[1;95m│\033[0m" + "   📊 Summary Statistics      → Success rate, deviation metrics, performance grades".ljust(98) + "\033[1;95m│\033[0m")
    print("\033[1;95m│\033[0m" +
          "   🤖 LLM Analysis            → AI-powered insights into result differences".ljust(98) + "\033[1;95m│\033[0m")
    print("\033[1;95m│\033[0m" + "   📋 Conclusions             → Comprehensive findings and improvement recommendations".ljust(98) + "\033[1;95m│\033[0m")
    print("\033[1;95m│\033[0m" + "   📈 Visualizations          → 7+ charts (bar, scatter, heatmap, histogram, tables)".ljust(98) + "\033[1;95m│\033[0m")
    print("\033[1;95m│\033[0m" +
          "   🌐 HTML Dashboard          → Interactive results browser with embedded images".ljust(98) + "\033[1;95m│\033[0m")
    print("\033[1;95m│\033[0m" +
          "   📁 CSV Exports             → Raw data for custom analysis".ljust(98) + "\033[1;95m│\033[0m")
    print("\033[1;95m│\033[0m" + " ".ljust(98) + "\033[1;95m│\033[0m")
    print("\033[1;95m└" + "─" * 98 + "┘\033[0m")

    # Next Steps Section
    print("\n\033[1;96m┌" + "─" * 98 + "┐\033[0m")
    print("\033[1;96m│\033[0m" +
          "\033[1;97m NEXT STEPS AFTER COMPLETION\033[0m".center(98) + "\033[1;96m│\033[0m")
    print("\033[1;96m├" + "─" * 98 + "┤\033[0m")
    print("\033[1;96m│\033[0m" + " ".ljust(98) + "\033[1;96m│\033[0m")
    print("\033[1;96m│\033[0m" +
          "   1. Review outputs/[paper_name]_results.txt for complete execution log".ljust(98) + "\033[1;96m│\033[0m")
    print("\033[1;96m│\033[0m" +
          "   2. Open outputs/visualizations/visualizations.html in browser".ljust(98) + "\033[1;96m│\033[0m")
    print("\033[1;96m│\033[0m" +
          "   3. Analyze outputs/visualizations/detailed_comparison.csv for raw metrics".ljust(98) + "\033[1;96m│\033[0m")
    print("\033[1;96m│\033[0m" +
          "   4. Read LLM insights and recommendations in the results file".ljust(98) + "\033[1;96m│\033[0m")
    print("\033[1;96m│\033[0m" +
          "   5. Check outputs/agent_execution.log for detailed agent activity".ljust(98) + "\033[1;96m│\033[0m")
    print("\033[1;96m│\033[0m" + " ".ljust(98) + "\033[1;96m│\033[0m")
    print("\033[1;96m└" + "─" * 98 + "┘\033[0m")

    print("\n" + "=" * 100)
    print("\033[1;92m▶ Starting Pre-Flight Checks...\033[0m")
    print("=" * 100 + "\n")

    # Resolve paper path
    workspace_root = Path(__file__).parent
    paper_dir = workspace_root / "papers"
    paper_source_dir = workspace_root / "papers_source_code"

    paper_path = None
    if args.paper:
        # User specified a paper
        paper_path = Path(args.paper)
        if not paper_path.exists():
            print(f"❌ Paper not found: {args.paper}")
            return 1
        if not paper_path.suffix == '.pdf':
            print(f"❌ Not a PDF file: {args.paper}")
            return 1
    else:
        # Auto-detect: Priority 1 - Decontextualisation.pdf (British spelling)
        default_paper = paper_dir / "Decontextualisation.pdf"
        if not default_paper.exists():
            # Also try American spelling
            default_paper = paper_dir / "Decontextualization.pdf"

        if default_paper.exists():
            paper_path = default_paper
            print(f"✓ Using default paper: {default_paper.name}")
        else:
            # Priority 2 - Any PDF in papers/
            if paper_dir.exists():
                pdf_files = list(paper_dir.glob("*.pdf"))
                if pdf_files:
                    paper_path = pdf_files[0]
                    print(f"✓ Auto-detected paper: {paper_path.name}")
                else:
                    print(f"❌ No PDF files found in ./papers/")
                    print("   Please add your research paper PDF to ./papers/")
                    return 1
            else:
                print(f"❌ ./papers/ directory not found!")
                print("   Please create it and add your research paper PDF")
                return 1

    print(f"\n📄 Paper: {paper_path.name}")

    # Resolve codebase source
    codebase_source = None
    if args.code:
        codebase_source = args.code
        if args.code.startswith('http'):
            print(f"📦 Code source: {args.code} (GitHub)")
        else:
            code_path = Path(args.code)
            if not code_path.exists():
                print(f"❌ Code path not found: {args.code}")
                return 1
            print(f"📦 Code source: {args.code} (local)")
    else:
        print(
            f"📦 Code source: auto-detect (papers_source_code/ or GitHub URLs from papers)")

    print("\n📍 Checking workspace structure...")

    if not paper_dir.exists():
        print(f"❌ ./papers/ directory not found!")
        print("   Please create it and add your research paper PDF")
        return 1

    # Check for any code in papers_source_code directory
    if not paper_source_dir.exists():
        print(f"⚠️  ./papers_source_code/ directory not found!")
        print("   📁 The agent will search for GitHub URLs in the paper.")
        print("   💡 Or create ./papers_source_code/ and place codebase there manually.")
    else:
        # Check if there are any subdirectories or Python files
        has_code = any(paper_source_dir.iterdir())
        if has_code:
            subdirs = [d.name for d in paper_source_dir.iterdir()
                       if d.is_dir()]
            if subdirs:
                print(f"✓ Found code directory: {paper_source_dir}")
                print(
                    f"   Subdirectories: {', '.join(subdirs[:3])}{' ...' if len(subdirs) > 3 else ''}")
            else:
                print(f"✓ Found files in: {paper_source_dir}")
        else:
            print(f"⚠️  ./papers_source_code/ exists but is empty!")
            print("   📁 The agent will search for GitHub URLs in the paper.")
            print("   💡 Or place your codebase in ./papers_source_code/ manually.")

    # Check Ollama
    print("\n🔍 Checking Ollama...")
    # Create a temporary agent just to check Ollama
    temp_agent = ReproductionAgent()

    if not temp_agent.is_available():
        print("❌ Ollama is not running!")
        print("\n   Please start Ollama in another terminal:")
        print("   $ ollama serve")
        print("\n   And ensure you have a model installed:")
        print("   $ ollama pull llama3")
        return 1

    models = temp_agent.list_models()
    if not models:
        print("❌ No Ollama models found!")
        print("\n   Please pull a model:")
        print("   $ ollama pull llama3")
        return 1

    print(f"✓ Ollama is running (models: {', '.join(models[:3])})")

    # Run agent
    print("\n" + "=" * 100)
    print("\033[1;92m🚀 LAUNCHING 4-STAGE REPRODUCTION WORKFLOW...\033[0m")
    print("=" * 100 + "\n")

    try:
        agent = ReproductionAgent()
        results = agent.run(paper_path=paper_path,
                            codebase_source=codebase_source)

        if 'error' in results:
            print(f"\n❌ Error: {results['error']}")
            return 1

        # Final Success Banner
        print("\n" + "=" * 100)
        print("\033[1;92m" + "█" * 100 + "\033[0m")
        print("\033[1;92m█" + " " * 98 + "█\033[0m")
        print(
            "\033[1;92m█" + "✅ REPRODUCTION WORKFLOW COMPLETED SUCCESSFULLY!".center(98) + "█\033[0m")
        print("\033[1;92m█" + " " * 98 + "█\033[0m")
        print("\033[1;92m" + "█" * 100 + "\033[0m")
        print("=" * 100)

        print("\n\033[1;96m📊 Results Summary:\033[0m")
        print("  • All outputs saved to: \033[1;93m./outputs/\033[0m")
        print(
            "  • \033[1;97mComplete execution log:\033[0m \033[1;93m./outputs/[paper_name]_results.txt\033[0m")
        print(
            "  • \033[1;97mRaw agent log:\033[0m \033[1;93m./outputs/agent_execution.log\033[0m")
        print(
            "  • Visualizations: \033[1;93m./outputs/visualizations/visualizations.html\033[0m")
        print(
            "  • CSV data: \033[1;93m./outputs/visualizations/detailed_comparison.csv\033[0m")

        print(
            "\n\033[1;92m✨ Open the HTML dashboard in your browser to explore results!\033[0m\n")

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
