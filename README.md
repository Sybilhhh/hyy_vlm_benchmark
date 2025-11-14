# VLM Benchmark System

A comprehensive framework for evaluating Vision-Language Models (VLMs) across multiple datasets and evaluation metrics.

## Installation

### Basic Installation
```bash
pip install vlm-benchmark
```

### Full Installation (with all optional dependencies)
```bash
pip install vlm-benchmark[full]
```

### Development Installation
```bash
git clone https://github.com/yourusername/vlm-benchmark.git
cd vlm-benchmark
pip install -e .[dev,full]
```

## Quick Start

### 1. Initialize Project
```bash
vlm-benchmark init my_benchmark_project
cd my_benchmark_project
```

### 2. Configure Settings
Edit `configs/example_config.yaml`:

### 3. Validate Environment
```bash
vlm-benchmark validate
```

### 4. Run Benchmark
```bash
vlm-benchmark run -m gpt-4o,qwen2.5vl-7b -d capability,dream-1k -e autodq,bleu
```

## Python API Usage

```python
import asyncio
from vlm_benchmark import VLMBenchmark

async def main():
    # Initialize benchmark system
    benchmark = VLMBenchmark("configs/benchmark_config.yaml")
    
    # Run comprehensive evaluation
    results = await benchmark.run_benchmark(
        models=["gpt-4o", "qwen2.5vl-7b", "tarsier2"],
        datasets=["capability", "dream-1k", "et-bench-captioning"],
        evaluators=["autodq", "davidsonian-sg", "bleu"],
        output_dir="./results"
    )
    
    print("Benchmark completed!")

asyncio.run(main())
```

## Supported Models

### API-Based Models
- **GPT-4o**

### Local Models  
- **Qwen2.5-VL (7B/32B/72B)**
- **Tarsier2 (7B)**

## Supported Datasets

- **Capability**: General capability assessment dataset
- **Dream-1K**: Visual reasoning and dream interpretation
- **E.T. Bench-Captioning**: Image captioning evaluation
- **Custom Datasets**: Extensible JSON-based dataset loader

## Evaluation Methods

### Reference-Based (require ground truth)
- **AutoDQ**
- **ETBench**
- **Eventhallusion**
- **videohallucer**

### Reference-Free  
- **Davidsonian Scene Graph**: Scene understanding quality assessment



## CLI Reference

### Commands

```bash
# Initialize new project
vlm-benchmark init [path]

# List available components  
vlm-benchmark list [models|datasets|evaluators|all]

# Validate environment
vlm-benchmark validate

# Run benchmark
vlm-benchmark run -m MODEL1,MODEL2 -d DATASET1,DATASET2 -e EVAL1,EVAL2

# Global options
--config CONFIG_FILE    # Configuration file path
--log-level LEVEL       # Logging level (DEBUG|INFO|WARNING|ERROR)  
--log-file FILE         # Log file path
```

### Examples

```bash
# Evaluate GPT-4o and local Qwen model on multiple datasets
vlm-benchmark run \\
    -m gpt-4o,qwen2.5vl-7b \\
    -d capability,dream-1k,et-bench-captioning \\
    -e autodq,bleu,rouge \\
    -o ./results/experiment1

# List all available models
vlm-benchmark list models

# Check environment setup
vlm-benchmark validate
```

## Extending the Framework

### Adding New Models

```python
from vlm_benchmark.models import BaseVLMModel

class MyCustomModel(BaseVLMModel):
    async def predict(self, sample):
        # Implement prediction logic
        return "Generated caption"
    
    async def load_model(self):
        # Load model
        pass
    
    async def unload_model(self):
        # Cleanup
        pass

# Register the model
benchmark.model_registry.register_model_class("my_model", MyCustomModel)
```

### Adding New Datasets

```python
from vlm_benchmark.datasets import BaseDataset

class MyCustomDataset(BaseDataset):
    def _load_dataset(self):
        # Load dataset samples
        for item in my_data_source:
            self._samples.append({
                'id': item['id'],
                'image_path': item['image_path'],
                'prompt': item.get('prompt', 'Describe this image.'),
                'ground_truth': item.get('caption')
            })

# Register the dataset
benchmark.dataset_registry.register_dataset_class("my_dataset", MyCustomDataset)
```

### Adding New Evaluators

```python
from vlm_benchmark.evaluators import BaseEvaluator, EvaluationResult

class MyCustomEvaluator(BaseEvaluator):
    async def evaluate(self, predictions, dataset):
        scores = []
        for pred in predictions:
            score = self._calculate_score(pred)
            scores.append(score)
        
        return EvaluationResult(
            score=statistics.mean(scores),
            details={'individual_scores': scores},
            method='my_evaluator',
            ground_truth_required=True
        )

# Register the evaluator  
benchmark.evaluator_registry.register_evaluator_class("my_evaluator", MyCustomEvaluator)
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black vlm_benchmark/
flake8 vlm_benchmark/
```

### Type Checking
```bash
mypy vlm_benchmark/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this benchmark system in your research, please cite:

```bibtex
@software{vlm_benchmark,
    title={VLM Benchmark System: A Comprehensive Framework for Vision-Language Model Evaluation},
    author={Roy Yang, Shiyuan Feng},
    year={2025},
    url={}
}
```