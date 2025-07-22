#!/usr/bin/env python3
"""
Interactive Loss Distribution Analysis Launcher
==============================================

A simple launcher script for the interactive observed loss distribution analysis.
This script provides an easy way to run the interactive analysis with different configurations.

Usage:
    python run_interactive_analysis.py [run_name]

Examples:
    python run_interactive_analysis.py run_g1
    python run_interactive_analysis.py run_g10
    python run_interactive_analysis.py  # Uses default from config
"""

import os
import sys
import argparse
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    """Main launcher function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Interactive Observed Loss Distribution Analysis Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_interactive_analysis.py run_g1
  python run_interactive_analysis.py run_g10
  python run_interactive_analysis.py --help
        """
    )
    
    parser.add_argument(
        'run_name',
        nargs='?',
        help='Name of the run to analyze (e.g., run_g1, run_g10)'
    )
    
    parser.add_argument(
        '--port',
        help='Port name to analyze (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output directory (overrides config)'
    )
    
    parser.add_argument(
        '--list-runs',
        action='store_true',
        help='List available runs and exit'
    )
    
    args = parser.parse_args()
    
    # Import config after argument parsing
    try:
        from config import config
        print("✅ Using centralized configuration")
    except ImportError as e:
        print(f"❌ Error: Cannot import config: {e}")
        return 1
    
    # List available runs if requested
    if args.list_runs:
        print("📁 Available runs:")
        results_dir = "results/cv_results"
        if os.path.exists(results_dir):
            runs = [d for d in os.listdir(results_dir) if d.startswith('run_g')]
            runs.sort()
            for run in runs:
                print(f"   • {run}")
        else:
            print(f"   No results directory found: {results_dir}")
        return 0
    
    # Determine run name
    run_name = args.run_name or config.RUN_PATH
    port_name = args.port or config.reference_port
    
    print("🔍 INTERACTIVE OBSERVED LOSS DISTRIBUTION ANALYSIS")
    print("="*60)
    print(f"📍 Run: {run_name}")
    print(f"📍 Port: {port_name}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*60)
    
    # Update config if needed
    if args.run_name:
        config.RUN_PATH = run_name
    if args.port:
        config.reference_port = port_name
    if args.output_dir:
        config.results_output_dir = args.output_dir
    
    # Import and run the interactive analysis
    try:
        from scripts.plot_observed_loss_distribution_interactive import main as run_analysis
        run_analysis()
        return 0
    except ImportError as e:
        print(f"❌ Error importing interactive analysis: {e}")
        print("💡 Make sure you have installed the required dependencies:")
        print("   pip install plotly pandas numpy kaleido")
        return 1
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 