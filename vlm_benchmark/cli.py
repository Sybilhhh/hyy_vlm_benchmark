"""
Command-line interface for VLM Benchmark System
"""

import argparse
import asyncio
import sys
from pathlib import Path
import logging
import json

from pipeline.benchmark_pipeline import VLMBenchmark
from utils import validate_environment, create_project_structure, setup_logging


async def run_benchmark_command(args):
    """Run benchmark command"""
    # Setup logging
    setup_logging({'level': args.log_level, 'file': args.log_file})
    
    try:
        # Initialize benchmark
        benchmark = VLMBenchmark(args.config)
        
        # Parse model, dataset, and evaluator lists
        models = args.models.split(',') if args.models else []
        datasets = args.datasets.split(',') if args.datasets else []
        evaluators = args.evaluators.split(',') if args.evaluators else []
        
        if not models:
            print("Error: No models specified")
            return 1
            
        if not datasets:
            print("Error: No datasets specified") 
            return 1
            
        if not evaluators:
            print("No evaluators specified, running predictions only")

        
        # Run benchmark
        results = await benchmark.run_benchmark(
            models=models,
            datasets=datasets,
            evaluators=evaluators,
            output_dir=args.output_dir
        )
        
        print(f"Benchmark completed! Results saved to {args.output_dir}")
        return 0
        
    except Exception as e:
        logging.error(f"Benchmark failed: {e}")
        return 1


def list_components_command(args):
    """List available components"""
    try:
        benchmark = VLMBenchmark(args.config)
        
        if args.component == 'models' or args.component == 'all':
            print("Available Models:")
            for model in benchmark.list_available_models():
                print(f"  - {model}")
        
        if args.component == 'datasets' or args.component == 'all':
            print("\\nAvailable Datasets:")
            for dataset in benchmark.list_available_datasets():
                print(f"  - {dataset}")
        
        if args.component == 'evaluators' or args.component == 'all':
            print("\\nAvailable Evaluators:")
            for evaluator in benchmark.list_available_evaluators():
                print(f"  - {evaluator}")
                
        return 0
        
    except Exception as e:
        print(f"Error listing components: {e}")
        return 1


def validate_command(args):
    """Validate environment and setup"""
    env_check = validate_environment()
    
    print("Environment Validation Results:")
    print("=" * 40)
    print(f"Overall Valid: {'✓' if env_check['valid'] else '✗'}")
    print(f"CUDA Available: {'✓' if env_check['cuda_available'] else '✗'}")
    
    if env_check['missing_dependencies']:
        print("\\nMissing Required Dependencies:")
        for dep in env_check['missing_dependencies']:
            print(f"  ✗ {dep}")
        print("\\nInstall missing dependencies with:")
        print(f"  pip install {' '.join(env_check['missing_dependencies'])}")
    
    if env_check['warnings']:
        print("\\nOptional Dependencies (for full functionality):")
        for warning in env_check['warnings']:
            print(f"  ! {warning}")
    
    if env_check['valid']:
        print("\\n✓ Environment is ready for benchmarking!")
    else:
        print("\\n✗ Please install missing dependencies before running benchmarks.")
        return 1
    
    return 0


def init_command(args):
    """Initialize new benchmark project"""
    try:
        create_project_structure(args.path)
        print(f"✓ Benchmark project structure created at: {args.path}")
        print("\\nNext steps:")
        print(f"1. cd {args.path}")
        print("2. Edit configs/example_config.yaml with your settings")
        print("3. Add your datasets to the data/ directory")
        print("4. Run: vlm-benchmark validate")
        return 0
    except Exception as e:
        print(f"Error creating project structure: {e}")
        return 1

def display_table(args):
    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {args.path}")
        return
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    from evaluators import EvaluationResult

    model = next(iter(data))
    dataset = next(iter(data[model]))
    task = next(iter(data[model][dataset]))

    entry = data[model][dataset][task]

    # Turn the stored repr into a real EvaluationResult instance
    if isinstance(entry, str) and entry.startswith("EvaluationResult("):
        # Evaluate only with EvaluationResult in scope
        result: EvaluationResult = eval(entry, {"EvaluationResult": EvaluationResult})
    elif isinstance(entry, EvaluationResult):
        result = entry
    else:
        raise TypeError("Unsupported entry format; expected EvaluationResult repr or instance.")

    summary_raw = result.details["summary_table"]
    print(summary_raw.replace("\\n", "\n"))

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="VLM Benchmark System - Comprehensive evaluation of Vision-Language Models"
    )
    
    # Global arguments
    parser.add_argument(
        "--config", "-c", 
        default="configs/benchmark_config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        help="Log file path"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("run", help="Run benchmark evaluation")
    bench_parser.add_argument(
        "--models", "-m", 
        required=True,
        help="Comma-separated list of models to evaluate"
    )
    bench_parser.add_argument(
        "--datasets", "-d",
        required=True, 
        help="Comma-separated list of datasets to use"
    )
    bench_parser.add_argument(
        "--evaluators", "-e",
        required=False,
        help="Comma-separated list of evaluators to apply"  
    )
    bench_parser.add_argument(
        "--output-dir", "-o",
        default="./results",
        help="Output directory for results"
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available components")
    list_parser.add_argument(
        "component",
        choices=["models", "datasets", "evaluators", "all"],
        default="all",
        nargs="?",
        help="Component type to list"
    )
    
    # Validate command
    subparsers.add_parser("validate", help="Validate environment and dependencies")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize new benchmark project")
    init_parser.add_argument(
        "path",
        default="./vlm_benchmark_project",
        nargs="?", 
        help="Project directory path"
    )

    display_parser = subparsers.add_parser("display", help="Display a table nicely in the terminal if you missed it")
    display_parser.add_argument(
        "path",
        default='.',
        nargs="?",
        help="path to the detailed_results.json"
    )
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == "run":
        return asyncio.run(run_benchmark_command(args))
    elif args.command == "list":
        return list_components_command(args)
    elif args.command == "validate":
        return validate_command(args)
    elif args.command == "init":
        return init_command(args)
    elif args.command == "display":
        return display_table(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())