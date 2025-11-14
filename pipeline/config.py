"""
Configuration management for VLM benchmark system
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


class BaseConfig:
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
    
    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"{key} not found in {self.__class__.__name__}")

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"{key} not found in {self.__class__.__name__}")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to a dictionary"""
        return asdict(self)


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for a specific model"""
    type: str  # "local" or "api"
    model_path: Optional[str] = None  # For local models
    api_key: Optional[str] = None     # For API models
    api_endpoint: Optional[str] = None
    device: str = "auto"              # For local models
    batch_size: int = 1
    max_tokens: int = 1024
    temperature: float = 0.0
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig(BaseConfig):
    """Configuration for a specific dataset"""
    data_path: str
    split: str = "test"
    max_samples: Optional[int] = None
    image_dir: Optional[str] = None
    annotation_file: Optional[str] = None
    load_predictions_from_file: Optional[str] = None
    preprocessing: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluatorConfig(BaseConfig):
    """Configuration for a specific evaluator"""
    type: str
    llm_model: Optional[str] = None     # For evaluators that need LLM (like AutoDQ)
    llm_api_key: Optional[str] = None
    ground_truth_required: bool = True
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Allow any additional parameters to be stored"""
        # This will be called after __init__, allowing us to capture extra kwargs
        pass


@dataclass 
class LoggingConfig(BaseConfig):
    """Configuration for logging"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None

    
class BenchmarkConfig:
    """Main configuration class for the benchmark system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        self.models: Dict[str, ModelConfig] = {}
        self.datasets: Dict[str, DatasetConfig] = {}  
        self.evaluators: Dict[str, EvaluatorConfig] = {}
        self.logging_config: LoggingConfig = LoggingConfig()
        
        # Load default configuration
        self._load_defaults()
        
        # Load user configuration if provided
        if config_path:
            self.load_from_file(config_path)
    
    def _load_defaults(self):
        """Load default configurations"""
        
        # Default model configurations
        self.models = {
            "gpt-4o": ModelConfig(
                type="api",
                api_endpoint="https://api.openai.com/v1/chat/completions",
                max_tokens=1024
            ),
            "doubao-vl": ModelConfig(
                type="api", 
                api_endpoint="https://ark.cn-beijing.volces.com/api/v3/chat/completions"
            ),
            "gemini-pro-vision": ModelConfig(
                type="api",
                api_endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"
            ),
            "qwen2.5vl-7b": ModelConfig(
                type="local",
                model_path="Qwen/Qwen2.5-VL-7B-Instruct",
                device="auto"
            ),
            "qwen2.5vl-32b": ModelConfig(
                type="local", 
                model_path="Qwen/Qwen2.5-VL-32B-Instruct",
                device="auto"
            ),
            "qwen2.5vl-72b": ModelConfig(
                type="local",
                model_path="Qwen/Qwen2.5-VL-72B-Instruct", 
                device="auto"
            ),
            "qwen3vl-8b": ModelConfig(
                type="local",
                model_path="Qwen/Qwen3-VL-8B-Instruct",
                device="auto"
            ),
            "tarsier2-7b": ModelConfig(
                type="local",
                model_path="TIGER-Lab/Tarsier2-7B",
                device="auto"
            )
        }
        
        # Default dataset configurations
        self.datasets = {
            "capability": DatasetConfig(
                data_path="./data/capability/",
                annotation_file="annotations.json"
            ),
            "dream-1k": DatasetConfig(
                data_path="./data/dream-1k/",
                annotation_file="dream_1k.json"
            ),
            "et-bench-captioning": DatasetConfig(
                data_path="./data/et-bench-captioning/",
                annotation_file="captions.json"
            ),
            "video-hallucer": DatasetConfig(
                data_path="/data/datasets/VideoHallucer",
                annotation_file="/data/datasets/VideoHallucer"
            ),
            "event-hallusion": DatasetConfig(
                data_path="/data/datasets/EventHallusion_videos",
                annotation_file="/workspace/EventHallusion/questions"
            ),
            "perception-test": DatasetConfig(
                data_path="/data/datasets/PerceptionTest/videos",
                annotation_file="/data/datasets/PerceptionTest/sample.json"
            ),
            "tvbench": DatasetConfig(
                data_path="/data/datasets/TVBench/video",
                annotation_file="/data/datasets/TVBench/json"
            ),
            "tomato": DatasetConfig(
                data_path="/data/datasets/TOMATO/videos",
                annotation_file="/data/datasets/TOMATO/data"
            ),
            "test-50_camera": DatasetConfig(
                data_path="/home/dyvm6xra/dyvm6xrauser04/yuyang/vlm_benchmark/test_50",
                annotation_file= None
            ),
            "test-50_dense": DatasetConfig(
                data_path="/home/dyvm6xra/dyvm6xrauser04/yuyang/vlm_benchmark/test_50",
                annotation_file= None
            ),
        }
        
        # Default evaluator configurations  
        self.evaluators = {
            "autodq": EvaluatorConfig(
                type="autodq",
                llm_model="gpt-4o",
                ground_truth_required=True
            ),
            "davidsonian-sg": EvaluatorConfig(
                type="davidsonian_scene_graph",
                ground_truth_required=False
            ),
            "etbench": EvaluatorConfig(
                type="etbench"
            ),
            "video-hallucer": EvaluatorConfig(
                type="video-hallucer"
            ),
            "event-hallusion": EvaluatorConfig(
                type="event-hallusion"
            ),
            "perceptipn-test": EvaluatorConfig(
                type='perception-test'
            ),
            "tvbench": EvaluatorConfig(
                type='tvbencht'
            ),
            "tomato": EvaluatorConfig(
                type='tomato'
            ),
            "etva": EvaluatorConfig(
                type='etva'
            ),
            "etva": EvaluatorConfig(
                type='etva'
            ),
        }
    
    def load_from_file(self, config_path: str):
        """Load configuration from file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load based on file extension
        if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
        elif config_file.suffix.lower() == '.json':
            with open(config_file, 'r') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_file.suffix}")
        
        # Update configurations
        if 'models' in config_data:
            for model_name, model_config in config_data['models'].items():
                if model_name in self.models:
                    # Update existing config
                    for key, value in model_config.items():
                        setattr(self.models[model_name], key, value)
                else:
                    # Create new config
                    self.models[model_name] = ModelConfig(**model_config)
        
        if 'datasets' in config_data:
            for dataset_name, dataset_config in config_data['datasets'].items():
                if dataset_name in self.datasets:
                    for key, value in dataset_config.items():
                        setattr(self.datasets[dataset_name], key, value)
                else:
                    self.datasets[dataset_name] = DatasetConfig(**dataset_config)
        
        if 'evaluators' in config_data:
            for evaluator_name, evaluator_config in config_data['evaluators'].items():
                if evaluator_name in self.evaluators:
                    for key, value in evaluator_config.items():
                        setattr(self.evaluators[evaluator_name], key, value)
                else:
                    # Separate known fields from extra parameters
                    known_fields = {'type', 'llm_model', 'llm_api_key', 'ground_truth_required', 'additional_params'}
                    evaluator_params = {}
                    extra_params = {}
                    
                    for key, value in evaluator_config.items():
                        if key in known_fields:
                            evaluator_params[key] = value
                        else:
                            extra_params[key] = value
                    
                    # Create evaluator config with known fields
                    eval_config = EvaluatorConfig(**evaluator_params)
                    
                    # Set extra parameters as attributes
                    for key, value in extra_params.items():
                        setattr(eval_config, key, value)
                    
                    self.evaluators[evaluator_name] = eval_config
        
        if 'logging' in config_data:
            self.logging_config = LoggingConfig(**config_data['logging'])
    
    def save_to_file(self, config_path: str):
        """Save current configuration to file"""
        config_data = {
            'models': {name: config.__dict__ for name, config in self.models.items()},
            'datasets': {name: config.__dict__ for name, config in self.datasets.items()},
            'evaluators': {name: config.__dict__ for name, config in self.evaluators.items()},
            'logging': self.logging_config.__dict__
        }
        
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        elif config_file.suffix.lower() == '.json':
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_file.suffix}")
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        return self.models[model_name]
    
    def get_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """Get configuration for a specific dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
        return self.datasets[dataset_name]
    
    def get_evaluator_config(self, evaluator_name: str) -> EvaluatorConfig:
        """Get configuration for a specific evaluator"""
        if evaluator_name not in self.evaluators:
            raise ValueError(f"Evaluator '{evaluator_name}' not found in configuration")
        return self.evaluators[evaluator_name]


# Example configuration file template
EXAMPLE_CONFIG = """
# VLM Benchmark Configuration

models:
  gpt-4o:
    type: api
    api_key: "your-openai-api-key"
    temperature: 0.0
    max_tokens: 1024
  
  qwen2.5vl-7b:
    type: local
    model_path: "./models/Qwen2.5-VL-7B-Instruct"
    device: "cuda:0"
    batch_size: 4

datasets:
  capability:
    data_path: "./data/capability/"
    annotation_file: "annotations.json"
    max_samples: 1000
  
  dream-1k:
    data_path: "./data/dream-1k/" 
    annotation_file: "dream_1k.json"

evaluators:
  autodq:
    type: autodq
    llm_model: gpt-4o
    llm_api_key: "your-openai-api-key"
    ground_truth_required: true
  
  davidsonian-sg:
    type: davidsonian_scene_graph
    ground_truth_required: false

logging:
  level: INFO
  file: "./logs/benchmark.log"
"""

def create_example_config(output_path: str = "configs/benchmark_config.yaml"):
    """Create an example configuration file"""
    config_file = Path(output_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        f.write(EXAMPLE_CONFIG)
    
    print(f"Example configuration saved to {output_path}")


if __name__ == "__main__":
    # Create example configuration
    create_example_config()
    
    # Test loading configuration
    config = BenchmarkConfig()
    print("Available models:", list(config.models.keys()))
    print("Available datasets:", list(config.datasets.keys()))  
    print("Available evaluators:", list(config.evaluators.keys()))
