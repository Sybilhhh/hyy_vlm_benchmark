"""
VLM Benchmark System
A comprehensive framework for evaluating Vision-Language Models
"""
import os
from pipeline.config import BenchmarkConfig
from models import ModelRegistry
from datasets import DatasetRegistry
from evaluators import EvaluatorRegistry
from utils import setup_logging
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
from tqdm import tqdm

class VLMBenchmark:
    """Main benchmark orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the benchmark system
        
        Args:
            config_path: Path to configuration file (YAML/JSON)
        """
        self.config = BenchmarkConfig(config_path)
        self.model_registry = ModelRegistry()
        self.dataset_registry = DatasetRegistry()
        self.evaluator_registry = EvaluatorRegistry()
        
        # Setup logging
        setup_logging(self.config.logging_config)
        self.logger = logging.getLogger(__name__)
        
    def list_available_models(self) -> List[str]:
        """List all available models"""
        return self.model_registry.list_models()
    
    def list_available_datasets(self) -> List[str]:
        """List all available datasets"""
        return self.dataset_registry.list_datasets()
    
    def list_available_evaluators(self) -> List[str]:
        """List all available evaluation methods"""
        return self.evaluator_registry.list_evaluators()
    
    async def run_benchmark(
        self,
        models: List[str],
        datasets: List[str], 
        evaluators: List[str],
        output_dir: str = "./results"
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark evaluation
        
        Args:
            models: List of model names to evaluate
            datasets: List of dataset names to use
            evaluators: List of evaluation methods to apply
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing all evaluation results
        """

        self.logger.info(f"Starting benchmark with {len(models)} models, {len(datasets)} datasets, {len(evaluators)} evaluators")

        results = {}
        
        for model_name in models:
            model = self.model_registry.load_model(model_name, self.config.models.get(model_name, {}))
            results[model_name] = {}
            
            for dataset_name in datasets:
                dataset = self.dataset_registry.load_dataset(dataset_name, self.config.datasets.get(dataset_name, {}))
                results[model_name][dataset_name] = {}
                
                if dataset.load_predictions_from_file:
                    predictions = dataset.predictions
                else:
                    # Generate predictions
                    self.logger.info(f"Generating predictions for {model_name} on {dataset_name}")
                    predictions = await self._generate_predictions(model, dataset, output_dir)
                
                if not evaluators:
                    continue
                for evaluator_name in evaluators:
                    # Get evaluator config and convert to dict
                    eval_config = self.config.evaluators.get(evaluator_name, {})
                    if hasattr(eval_config, '__dict__'):
                        eval_config_dict = eval_config.__dict__
                    else:
                        eval_config_dict = eval_config
                    
                    evaluator = self.evaluator_registry.load_evaluator(
                        evaluator_name, 
                        eval_config_dict
                    )
                    
                    self.logger.info(f"Evaluating {model_name} on {dataset_name} using {evaluator_name}")
                    score = await evaluator.evaluate(predictions, dataset)
                    results[model_name][dataset_name][evaluator_name] = score
                    
        # Save results
        self._save_results(results, output_dir)
        return results
    
    async def _generate_predictions(self, model, dataset, output_dir) -> List[Dict[str, Any]]:
        """Generate model predictions for a dataset with batch inference"""
        predictions = []
        
        # 获取批量大小配置
        batch_size = getattr(model, 'batch_size', 1)
        
        # 检查模型是否支持批量推理
        supports_batch = hasattr(model, 'predict_batch')
        
        if supports_batch and batch_size > 1:
            # 批量推理模式
            self.logger.info(f"Using batch inference with batch_size={batch_size}")
            
            # 将数据集分批
            samples_list = list(dataset)
            for i in tqdm(range(0, len(samples_list), batch_size), desc="Batch inference"):
                batch_samples = samples_list[i:i+batch_size]
                
                try:
                    # 批量推理
                    batch_predictions = await model.predict_batch(batch_samples)
                    
                    # 将预测结果添加到样本中
                    for sample, prediction in zip(batch_samples, batch_predictions):
                        sample.update({'prediction': prediction})
                        predictions.append(sample)
                        
                except Exception as e:
                    self.logger.error(f"Error in batch prediction: {e}")
                    # 批量失败时回退到单个推理
                    for sample in batch_samples:
                        try:
                            prediction = await model.predict(sample)
                            sample.update({'prediction': prediction})
                            predictions.append(sample)
                        except Exception as e2:
                            self.logger.error(f"Error generating prediction for sample {sample.get('id')}: {e2}")
                            sample.update({'prediction': f"[Error: {str(e2)}]"})
                            predictions.append(sample)
        else:
            # 单样本推理模式
            self.logger.info("Using single-sample inference")
            for sample in tqdm(dataset, total=len(dataset), desc="Single inference"):
                try:
                    prediction = await model.predict(sample)
                    sample.update({'prediction': prediction})
                    predictions.append(sample)
                except Exception as e:
                    self.logger.error(f"Error generating prediction: {e}")
                    sample.update({'prediction': f"[Error: {str(e)}]"})
                    predictions.append(sample)
                
        self._save_predictions(predictions, output_dir)
        return predictions
    
    def _save_predictions(self, predictions: List[Dict[str, any]], output_dir: str):
        """Save predictions to files"""
        output_path = Path(output_dir) / 'predictions'
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed predictions
        with open(output_path / "predictions.json", "w") as f:
            json.dump(predictions, f, indent=2, default=str)
        
        self.logger.info(f"Predictions saved to {output_dir}")


    def _save_results(self, results: Dict[str, Any], output_dir: str):
        """Save benchmark results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        with open(output_path / "detailed_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary
        summary = self._generate_summary(results)
        with open(output_path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {output_dir}")
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from results"""
        summary = {
            'models': list(results.keys()),
            'datasets': [],
            'evaluators': [],
            'scores': {}
        }
        
        for model_name, model_results in results.items():
            summary['scores'][model_name] = {}
            for dataset_name, dataset_results in model_results.items():
                if dataset_name not in summary['datasets']:
                    summary['datasets'].append(dataset_name)
                    
                summary['scores'][model_name][dataset_name] = {}
                for evaluator_name, score in dataset_results.items():
                    if evaluator_name not in summary['evaluators']:
                        summary['evaluators'].append(evaluator_name)
                    summary['scores'][model_name][dataset_name][evaluator_name] = score
                    if 'summary_table' in score.details:
                        print(score.details['summary_table'])

                    
        return summary


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize benchmark system

        if "roy" in os.getcwd():
            config_path = "/media/sda/roy/vlm_benchmark/configs/benchmark_config.yaml"
            models = ["tarsier2-7b"]
            # models = ["qwen2.5vl-7b"]
                # datasets=["dream-1k"],
            datasets = ["dream-1k"]
            # datasets=["event-hallusion"]
            # evaluators=["autodq"]
            # evaluators=["event-hallusion"]
            # evaluators=["etva"]
            evaluators = ["autodq"]
            output_dir="/media/sda/roy/vlm_benchmark/output/test"
        else:
            pass

        benchmark = VLMBenchmark(config_path)
        
        # List available components
        print("Available models:", benchmark.list_available_models())
        print("Available datasets:", benchmark.list_available_datasets())
        print("Available evaluators:", benchmark.list_available_evaluators())
        
        # Run benchmark
        results = await benchmark.run_benchmark(
            # models=["qwen2.5vl-7b"],
            models = models,
            # datasets=["dream-1k"],
            datasets=datasets,
            # evaluators=["autodq"],
            evaluators=evaluators,
            output_dir=output_dir
        )
        
        print("Benchmark completed!")
        
    asyncio.run(main())
