from .datasets import *

class DatasetRegistry:
    """Registry for managing different datasets"""
    
    def __init__(self):
        self.datasets: Dict[str, BaseDataset] = {}
        self.dataset_classes = {
            'capability': CapabilityDataset,
            'dream-1k': Dream1KDataset,
            'et-bench-captioning': ETBenchCaptioningDataset,
            'video-hallucer': VideoHallucerDataset,
            'event-hallusion': EventHallusionDataset,
            'perception-test': PerceptionTestDataset,
            'test-50_camera': Test50Dataset,
            'test-50_dense': Test50_dense_Dataset,
            'tvbench': TVBenchDataset,
            'tomato': TOMATODataset,
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