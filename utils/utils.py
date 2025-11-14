"""
Utility functions and setup for VLM benchmark system
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import yaml
import time
from datetime import datetime
import asyncio
import aiofiles
import hashlib
import os


import logging
import sys
from pathlib import Path
from typing import Any, Dict

def setup_logging(config: Dict[str, Any] = None) -> None:
    """
    Idempotent logging setup:
    - Clears existing root handlers
    - Installs exactly one console handler (+ optional file handler)
    - Uses stdout for console output
    """
    config = config or {}
    level = getattr(logging, config.get('level', 'INFO').upper(), logging.INFO)
    fmt = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    datefmt = config.get('datefmt')  # optional

    # Build handlers you want
    handlers: list[logging.Handler] = []

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    handlers.append(console)

    log_file = config.get('file')
    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        handlers.append(fh)

    # Clear existing handlers and (re)configure root exactly once
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    # Python 3.8+: force=True guarantees we overwrite any prior config
    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Optional: capture warnings from 'warnings' module into logging
    logging.captureWarnings(True)

    # If you have loggers that add their own handlers elsewhere, prevent double-print:
    # for ln in config.get("no_propagate", []):
    #     lg = logging.getLogger(ln)
    #     lg.propagate = False


class ProgressTracker:
    """Track progress of benchmark operations"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)
    
    def update(self, amount: int = 1):
        """Update progress"""
        self.current += amount
        self._log_progress()
    
    def _log_progress(self):
        """Log current progress"""
        if self.total > 0:
            percentage = (self.current / self.total) * 100
            elapsed = time.time() - self.start_time
            
            if self.current > 0:
                eta = (elapsed / self.current) * (self.total - self.current)
                eta_str = f"ETA: {eta:.1f}s"
            else:
                eta_str = "ETA: --"
            
            self.logger.info(
                f"{self.description}: {self.current}/{self.total} "
                f"({percentage:.1f}%) - {eta_str}"
            )


class ResultsManager:
    """Manage benchmark results and reports"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    async def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(results, indent=2, default=str))
        
        self.logger.info(f"Results saved to {filepath}")
        return filepath
    
    def generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report from results"""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>VLM Benchmark Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .model-section {{ margin: 20px 0; border: 1px solid #ddd; padding: 15px; }}
        .dataset-results {{ margin-left: 20px; }}
        .score {{ font-weight: bold; color: #2e8b57; }}
        .error {{ color: #d32f2f; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>VLM Benchmark Results</h1>
        <p>Generated on: {timestamp}</p>
        <p>Models evaluated: {models}</p>
        <p>Datasets used: {datasets}</p>
        <p>Evaluation methods: {evaluators}</p>
    </div>
    
    <h2>Summary Table</h2>
    {summary_table}
    
    <h2>Detailed Results</h2>
    {detailed_results}
</body>
</html>
"""
        
        # Generate summary table
        summary_table = self._generate_summary_table(results)
        
        # Generate detailed results
        detailed_results = self._generate_detailed_html(results)
        
        # Get metadata
        models = list(results.keys()) if results else []
        datasets = []
        evaluators = []
        
        if results:
            first_model = list(results.values())[0]
            datasets = list(first_model.keys()) if first_model else []
            
            if datasets and first_model:
                first_dataset = list(first_model.values())[0]
                evaluators = list(first_dataset.keys()) if first_dataset else []
        
        html = html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            models=", ".join(models),
            datasets=", ".join(datasets),
            evaluators=", ".join(evaluators),
            summary_table=summary_table,
            detailed_results=detailed_results
        )
        
        # Save HTML report
        html_file = self.output_dir / "benchmark_report.html"
        with open(html_file, 'w') as f:
            f.write(html)
        
        self.logger.info(f"HTML report saved to {html_file}")
        return str(html_file)
    
    def _generate_summary_table(self, results: Dict[str, Any]) -> str:
        """Generate HTML summary table"""
        if not results:
            return "<p>No results available</p>"
        
        # Build table
        table_html = "<table><thead><tr><th>Model</th>"
        
        # Get all unique dataset/evaluator combinations
        all_combinations = set()
        for model_results in results.values():
            for dataset, eval_results in model_results.items():
                for evaluator in eval_results.keys():
                    all_combinations.add(f"{dataset}_{evaluator}")
        
        for combo in sorted(all_combinations):
            table_html += f"<th>{combo}</th>"
        
        table_html += "</tr></thead><tbody>"
        
        # Add rows for each model
        for model_name, model_results in results.items():
            table_html += f"<tr><td>{model_name}</td>"
            
            for combo in sorted(all_combinations):
                dataset, evaluator = combo.rsplit('_', 1)
                
                if dataset in model_results and evaluator in model_results[dataset]:
                    score = model_results[dataset][evaluator]
                    if isinstance(score, dict) and 'score' in score:
                        score = score['score']
                    table_html += f'<td class="score">{score:.4f}</td>'
                else:
                    table_html += '<td class="error">N/A</td>'
            
            table_html += "</tr>"
        
        table_html += "</tbody></table>"
        return table_html
    
    def _generate_detailed_html(self, results: Dict[str, Any]) -> str:
        """Generate detailed HTML results"""
        html = ""
        
        for model_name, model_results in results.items():
            html += f'<div class="model-section"><h3>{model_name}</h3>'
            
            for dataset_name, dataset_results in model_results.items():
                html += f'<div class="dataset-results"><h4>Dataset: {dataset_name}</h4>'
                
                for evaluator_name, eval_result in dataset_results.items():
                    if isinstance(eval_result, dict):
                        score = eval_result.get('score', 'N/A')
                        details = eval_result.get('details', {})
                    else:
                        score = eval_result
                        details = {}
                    
                    html += f'<p><strong>{evaluator_name}:</strong> <span class="score">{score}</span></p>'
                    
                    if details:
                        html += '<ul>'
                        for key, value in details.items():
                            if key not in ['individual_scores']:  # Skip large arrays
                                html += f'<li>{key}: {value}</li>'
                        html += '</ul>'
                
                html += '</div>'
            
            html += '</div>'
        
        return html


class CacheManager:
    """Manage caching of model predictions and evaluation results"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def _get_cache_key(self, model_name: str, dataset_name: str, config: Dict[str, Any] = None) -> str:
        """Generate cache key for model predictions"""
        key_data = f"{model_name}_{dataset_name}"
        if config:
            # Include relevant config in cache key
            config_str = json.dumps(config, sort_keys=True)
            key_data += f"_{hashlib.md5(config_str.encode()).hexdigest()[:8]}"
        
        return key_data
    
    async def get_cached_predictions(self, model_name: str, dataset_name: str, config: Dict[str, Any] = None) -> Optional[List[Dict[str, Any]]]:
        """Get cached predictions if available"""
        cache_key = self._get_cache_key(model_name, dataset_name, config)
        cache_file = self.cache_dir / f"predictions_{cache_key}.json"
        
        if cache_file.exists():
            try:
                async with aiofiles.open(cache_file, 'r') as f:
                    content = await f.read()
                    predictions = json.loads(content)
                    self.logger.info(f"Loaded cached predictions: {cache_key}")
                    return predictions
            except Exception as e:
                self.logger.warning(f"Error loading cache {cache_key}: {e}")
        
        return None
    
    async def cache_predictions(self, model_name: str, dataset_name: str, predictions: List[Dict[str, Any]], config: Dict[str, Any] = None):
        """Cache model predictions"""
        cache_key = self._get_cache_key(model_name, dataset_name, config)
        cache_file = self.cache_dir / f"predictions_{cache_key}.json"
        
        try:
            async with aiofiles.open(cache_file, 'w') as f:
                await f.write(json.dumps(predictions, indent=2, default=str))
            self.logger.info(f"Cached predictions: {cache_key}")
        except Exception as e:
            self.logger.error(f"Error caching predictions {cache_key}: {e}")
    
    def clear_cache(self, pattern: str = None):
        """Clear cache files"""
        if pattern:
            cache_files = list(self.cache_dir.glob(f"*{pattern}*"))
        else:
            cache_files = list(self.cache_dir.glob("*.json"))
        
        for cache_file in cache_files:
            cache_file.unlink()
            self.logger.info(f"Removed cache file: {cache_file.name}")
        
        self.logger.info(f"Cleared {len(cache_files)} cache files")


def validate_environment():
    """Validate that required dependencies are available"""
    missing_deps = []
    warnings = []
    
    # Check required dependencies
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import PIL
    except ImportError:
        missing_deps.append("Pillow")
    
    try:
        import aiohttp
    except ImportError:
        missing_deps.append("aiohttp")
    
    try:
        import aiofiles
    except ImportError:
        missing_deps.append("aiofiles")
    
    # Check optional dependencies
    try:
        import transformers
    except ImportError:
        warnings.append("transformers (required for local HuggingFace models)")
    
    try:
        import scripts.openai_test as openai_test
    except ImportError:
        warnings.append("openai (required for OpenAI models and AutoDQ evaluation)")
    
    try:
        import nltk
    except ImportError:
        warnings.append("nltk (required for BLEU evaluation)")
    
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        warnings.append("rouge-score (required for ROUGE evaluation)")
    
    # Check CUDA availability
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except:
        pass
    
    return {
        'missing_dependencies': missing_deps,
        'warnings': warnings,
        'cuda_available': cuda_available,
        'valid': len(missing_deps) == 0
    }


def create_project_structure(base_dir: str = "./vlm_benchmark"):
    """Create the recommended project structure"""
    base_path = Path(base_dir)
    
    # Directory structure
    directories = [
        "configs",
        "data",
        "results",
        "logs",
        "cache",
        "models",  # For local model storage
        "scripts",  # For utility scripts
    ]
    
    for dir_name in directories:
        (base_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Create example files
    example_files = {
        "configs/example_config.yaml": '''# VLM Benchmark Configuration Example
models:
  gpt-4o:
    type: api
    api_key: "your-openai-api-key"
    temperature: 0.0
  
  qwen2.5vl-7b:
    type: local
    model_path: "./models/Qwen2.5-VL-7B-Instruct"
    device: "auto"

datasets:
  capability:
    data_path: "./data/capability/"
    annotation_file: "annotations.json"
  
evaluators:
  autodq:
    type: autodq
    llm_model: gpt-4o
    llm_api_key: "your-openai-api-key"

logging:
  level: INFO
  file: "./logs/benchmark.log"
''',
        
        "scripts/download_models.py": '''#!/usr/bin/env python3
"""
Script to download VLM models from HuggingFace
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def download_model(model_name: str, output_dir: str):
    """Download model from HuggingFace"""
    try:
        from transformers import AutoModel, AutoProcessor
        
        print(f"Downloading {model_name}...")
        model = AutoModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")

if __name__ == "__main__":
    models_to_download = [
        ("Qwen/Qwen2.5-VL-7B-Instruct", "./models/Qwen2.5-VL-7B-Instruct"),
        ("TIGER-Lab/Tarsier2-7B", "./models/Tarsier2-7B"),
    ]
    
    for model_name, output_dir in models_to_download:
        download_model(model_name, output_dir)
''',
        
        "scripts/run_benchmark.py": '''#!/usr/bin/env python3
"""
Example script to run VLM benchmark
"""
import sys
import asyncio
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from vlm_benchmark import VLMBenchmark

async def main():
    # Initialize benchmark
    benchmark = VLMBenchmark("configs/example_config.yaml")
    
    # Run benchmark
    results = await benchmark.run_benchmark(
        models=["gpt-4o", "qwen2.5vl-7b"],
        datasets=["capability", "dream-1k"],
        evaluators=["autodq", "bleu"],
        output_dir="./results"
    )
    
    print("Benchmark completed!")
    print(f"Results saved to ./results")

if __name__ == "__main__":
    asyncio.run(main())
'''
    }
    
    for file_path, content in example_files.items():
        full_path = base_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w') as f:
            f.write(content)
    
    print(f"Project structure created at: {base_path}")
    print("Directories created:", ", ".join(directories))
    print("Example files created:", ", ".join(example_files.keys()))


if __name__ == "__main__":
    # Validate environment
    env_check = validate_environment()
    
    print("Environment Validation:")
    print(f"Valid: {env_check['valid']}")
    print(f"CUDA Available: {env_check['cuda_available']}")
    
    if env_check['missing_dependencies']:
        print(f"Missing dependencies: {env_check['missing_dependencies']}")
    
    if env_check['warnings']:
        print(f"Optional dependencies missing: {env_check['warnings']}")
    
    # Create example project structure
    create_project_structure("./vlm_benchmark_example")