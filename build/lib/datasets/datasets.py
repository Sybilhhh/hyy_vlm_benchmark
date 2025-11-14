"""
Dataset registry and loaders for VLM benchmark system
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Iterator, Optional
import json
import logging
from pathlib import Path
import random


class BaseDataset(ABC):
    """Abstract base class for datasets"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = Path(config['data_path'])
        self.split = config.get('split', 'test')
        self.max_samples = config.get('max_samples')
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self._samples = []
        self._loaded = False
    
    @abstractmethod
    def _load_dataset(self):
        """Load dataset from files"""
        pass
    
    def load(self):
        """Load the dataset if not already loaded"""
        if not self._loaded:
            self._load_dataset()
            self._loaded = True
            
            # Apply max_samples limit if specified
            if self.max_samples and len(self._samples) > self.max_samples:
                random.shuffle(self._samples)
                self._samples = self._samples[:self.max_samples]
            
            self.logger.info(f"Loaded {len(self._samples)} samples")
    
    def __len__(self) -> int:
        """Return number of samples"""
        if not self._loaded:
            self.load()
        return len(self._samples)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples"""
        if not self._loaded:
            self.load()
        return iter(self._samples)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get sample by index"""
        if not self._loaded:
            self.load()
        return self._samples[index]


class CapabilityDataset(BaseDataset):
    """Capability benchmark dataset"""
    
    def _load_dataset(self):
        """Load Capability dataset"""
        annotation_file = self.data_path / self.config.get('annotation_file', 'annotations.json')
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        image_dir = self.data_path / self.config.get('image_dir', 'images')
        
        for item in annotations:
            image_path = image_dir / item['image']
            if image_path.exists():
                sample = {
                    'id': item.get('id', item['image']),
                    'image_path': str(image_path),
                    'prompt': item.get('prompt', 'Describe this image.'),
                    'ground_truth': item.get('caption', item.get('answer'))
                }
                self._samples.append(sample)
            else:
                self.logger.warning(f"Image not found: {image_path}")


class Dream1KDataset(BaseDataset):
    """Dream-1K dataset"""
    
    def _load_dataset(self):
        """Load Dream-1K dataset"""
        annotation_file = self.data_path / self.config.get('annotation_file', 'dream_1k.json')
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        image_dir = self.data_path / self.config.get('image_dir', 'images')
        
        # Handle different annotation formats
        if isinstance(data, list):
            annotations = data
        elif isinstance(data, dict):
            annotations = data.get('annotations', data.get('data', []))
        else:
            raise ValueError("Unexpected annotation format")
        
        for item in annotations:
            image_path = image_dir / item['image_name']
            if image_path.exists():
                sample = {
                    'id': item.get('id', item['image_name']),
                    'image_path': str(image_path),
                    'prompt': item.get('question', 'What do you see in this image?'),
                    'ground_truth': item.get('answer', item.get('caption'))
                }
                self._samples.append(sample)
            else:
                self.logger.warning(f"Image not found: {image_path}")


class ETBenchCaptioningDataset(BaseDataset):
    """E.T. Bench Captioning dataset"""
    
    def _load_dataset(self):
        """Load E.T. Bench Captioning dataset"""
        annotation_file = self.data_path / self.config.get('annotation_file', 'captions.json')
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        image_dir = self.data_path / self.config.get('image_dir', 'images')
        
        for item in annotations:
            image_filename = item.get('image_id', item.get('image'))
            if not image_filename.endswith(('.jpg', '.jpeg', '.png')):
                image_filename += '.jpg'  # Default extension
            
            image_path = image_dir / image_filename
            if image_path.exists():
                sample = {
                    'id': item.get('id', image_filename),
                    'image_path': str(image_path),
                    'prompt': 'Generate a detailed caption for this image.',
                    'ground_truth': item.get('caption', item.get('description'))
                }
                self._samples.append(sample)
            else:
                self.logger.warning(f"Image not found: {image_path}")


class GenericJSONDataset(BaseDataset):
    """Generic dataset loader for JSON annotations"""
    
    def _load_dataset(self):
        """Load generic JSON dataset"""
        annotation_file = self.data_path / self.config.get('annotation_file', 'annotations.json')
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            annotations = data
        elif isinstance(data, dict):
            # Try common keys for annotation lists
            for key in ['annotations', 'data', 'samples', 'images']:
                if key in data:
                    annotations = data[key]
                    break
            else:
                # Treat the dict itself as a single annotation
                annotations = [data]
        else:
            raise ValueError("Unexpected annotation format")
        
        image_dir = self.data_path / self.config.get('image_dir', 'images')
        
        # Define field mappings
        image_fields = ['image', 'image_name', 'image_path', 'filename', 'file_name']
        prompt_fields = ['prompt', 'question', 'text', 'instruction']
        gt_fields = ['ground_truth', 'caption', 'answer', 'label', 'description']
        id_fields = ['id', 'image_id', 'sample_id']
        
        for item in annotations:
            # Find image path
            image_name = None
            for field in image_fields:
                if field in item and item[field]:
                    image_name = item[field]
                    break
            
            if not image_name:
                self.logger.warning(f"No image field found in item: {list(item.keys())}")
                continue
            
            # Handle absolute vs relative paths
            if Path(image_name).is_absolute():
                image_path = Path(image_name)
            else:
                image_path = image_dir / image_name
            
            if image_path.exists():
                # Find other fields
                sample_id = None
                for field in id_fields:
                    if field in item:
                        sample_id = item[field]
                        break
                if not sample_id:
                    sample_id = image_name
                
                prompt = None
                for field in prompt_fields:
                    if field in item and item[field]:
                        prompt = item[field]
                        break
                if not prompt:
                    prompt = "Describe this image in detail."
                
                ground_truth = None
                for field in gt_fields:
                    if field in item and item[field]:
                        ground_truth = item[field]
                        break
                
                sample = {
                    'id': sample_id,
                    'image_path': str(image_path),
                    'prompt': prompt,
                    'ground_truth': ground_truth
                }
                
                # Add any additional fields
                for key, value in item.items():
                    if key not in ['id', 'image_path', 'prompt', 'ground_truth'] and \
                       key not in image_fields + prompt_fields + gt_fields + id_fields:
                        sample[f'extra_{key}'] = value
                
                self._samples.append(sample)
            else:
                self.logger.warning(f"Image not found: {image_path}")


class DatasetRegistry:
    """Registry for managing different datasets"""
    
    def __init__(self):
        self.datasets: Dict[str, BaseDataset] = {}
        self.dataset_classes = {
            'capability': CapabilityDataset,
            'dream-1k': Dream1KDataset,
            'et-bench-captioning': ETBenchCaptioningDataset,
            'generic': GenericJSONDataset
        }
        self.logger = logging.getLogger(__name__)
    
    def register_dataset(self, name: str, dataset: BaseDataset):
        """Register a dataset instance"""
        self.datasets[name] = dataset
        self.logger.info(f"Dataset '{name}' registered")
    
    def list_datasets(self) -> List[str]:
        """List all available datasets"""
        return list(set(list(self.datasets.keys()) + list(self.dataset_classes.keys())))
    
    def load_dataset(self, name: str, config: Dict[str, Any]) -> BaseDataset:
        """Load a dataset by name"""
        if name in self.datasets:
            return self.datasets[name]
        
        # Determine dataset class
        dataset_type = config.get('type', name.lower())
        if dataset_type in self.dataset_classes:
            dataset_class = self.dataset_classes[dataset_type]
        else:
            # Fall back to generic dataset
            dataset_class = GenericJSONDataset
            self.logger.warning(f"Unknown dataset type '{dataset_type}', using generic loader")
        
        # Create dataset instance
        dataset = dataset_class(config)
        dataset.load()
        
        # Register and return
        self.register_dataset(name, dataset)
        return dataset
    
    def register_dataset_class(self, name: str, dataset_class):
        """Register a new dataset class"""
        self.dataset_classes[name] = dataset_class
        self.logger.info(f"Dataset class '{name}' registered")
    
    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """Get information about a dataset"""
        if name in self.datasets:
            dataset = self.datasets[name]
            return {
                'name': name,
                'num_samples': len(dataset),
                'data_path': str(dataset.data_path),
                'split': dataset.split,
                'loaded': dataset._loaded
            }
        else:
            return {'name': name, 'loaded': False}


# Utility functions for dataset operations
def validate_dataset_structure(dataset_path: str, annotation_file: str = None) -> Dict[str, Any]:
    """Validate dataset structure and provide suggestions"""
    dataset_path = Path(dataset_path)
    results = {
        'valid': False,
        'issues': [],
        'suggestions': [],
        'structure': {}
    }
    
    if not dataset_path.exists():
        results['issues'].append(f"Dataset path does not exist: {dataset_path}")
        return results
    
    # Check for common annotation files
    annotation_files = []
    common_names = ['annotations.json', 'captions.json', 'data.json', 'dataset.json']
    if annotation_file:
        common_names.insert(0, annotation_file)
    
    for name in common_names:
        if (dataset_path / name).exists():
            annotation_files.append(name)
    
    if not annotation_files:
        results['issues'].append("No annotation files found")
        results['suggestions'].append(f"Create one of: {common_names}")
    else:
        results['structure']['annotation_files'] = annotation_files
    
    # Check for image directories
    image_dirs = []
    common_dirs = ['images', 'imgs', 'pictures', 'data']
    for dir_name in common_dirs:
        img_dir = dataset_path / dir_name
        if img_dir.exists() and img_dir.is_dir():
            # Count images
            image_count = len(list(img_dir.glob('*.jpg')) + 
                            list(img_dir.glob('*.jpeg')) + 
                            list(img_dir.glob('*.png')))
            image_dirs.append({'name': dir_name, 'count': image_count})
    
    if not image_dirs:
        results['issues'].append("No image directories found")
        results['suggestions'].append(f"Create image directory: {common_dirs[0]}")
    else:
        results['structure']['image_dirs'] = image_dirs
    
    results['valid'] = len(results['issues']) == 0
    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create dataset registry
    registry = DatasetRegistry()
    
    # Example dataset configuration
    config = {
        'data_path': './data/example_dataset/',
        'annotation_file': 'annotations.json',
        'image_dir': 'images',
        'max_samples': 100
    }
    
    try:
        # Load dataset
        dataset = registry.load_dataset('example', config)
        print(f"Dataset loaded with {len(dataset)} samples")
        
        # Get dataset info
        info = registry.get_dataset_info('example')
        print(f"Dataset info: {info}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        
        # Validate structure
        validation = validate_dataset_structure(config['data_path'])
        print(f"Validation results: {validation}")
