#!/usr/bin/env python3
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
