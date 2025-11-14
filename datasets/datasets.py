"""
Dataset registry and loaders for VLM benchmark system
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Iterator, Optional
import json
import logging
from pathlib import Path
import random

from constants import *

class BaseDataset(ABC):
    """Abstract base class for datasets"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = Path(config['data_path'])
        self.split = config.get('split', 'test')
        self.max_samples = config.get('max_samples')
        self.load_predictions_from_file = config.get('load_predictions_from_file')
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self._samples = []
        self._loaded = False
    
    @property
    def predictions(self):
        return self._samples

    # def _check_sample_integrity(self):
    #     for sample in self._samples:
    #         if not isinstance(sample.get(DATASET_REFERENCE, None), list):
    #             sample[DATASET_REFERENCE] = [sample[DATASET_REFERENCE]]
    #         if sample.get(DATASET_PROMPT, None) is not None and not isinstance(sample[DATASET_PROMPT], list):
    #             sample[DATASET_PROMPT] = [sample[DATASET_PROMPT]]

    @abstractmethod
    def _load_dataset(self):
        """Load dataset from files"""
        pass
    
    def _load_predictions(self):
        self.logger.info(f"Loading predictions from file: {self.load_predictions_from_file}")
        if not Path(self.load_predictions_from_file).exists():
            raise FileNotFoundError(f"Predictions file not found: {self.load_predictions_from_file}")
        
        with open(self.load_predictions_from_file, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            self._samples = data
        elif isinstance(data, dict):
            self._samples = data.get('annotations', data.get('data', []))
        else:
            raise ValueError("Unexpected annotation format")

    def load(self):
        """Load the dataset if not already loaded"""
        if not self._loaded:
            if self.load_predictions_from_file is not None:
                self._load_predictions()
            else:
                self._load_dataset()
                self._loaded = True
                
                # Apply max_samples limit if specified
                if self.max_samples and len(self._samples) > self.max_samples:
                    # random.shuffle(self._samples)
                    self._samples = self._samples[:self.max_samples]
                
                # self._check_sample_integrity()
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
        # annotation_file = self.data_path / self.config.get('annotation_file', 'dream_1k.json')
        
        annotation_file = Path(self.config.get('annotation_file'))

        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            annotations = data
        elif isinstance(data, dict):
            annotations = data.get('annotations', data.get('data', []))
        else:
            raise ValueError("Unexpected annotation format")


        # Handle different annotation formats

        for i, item in enumerate(annotations):
            # image_path = image_dir / item['image_name']
            video_path = self.data_path / item['video_file']
            if video_path.exists():
                sample = {
                    DATASET_ID: item.get('idx', -1),
                    DATASET_IMAGE_PATH: None,
                    DATASET_VIDEO_PATH: str(video_path),
                    # 'prompt': item.get('question', 'What do you see in this image?'),
                    DATASET_REFERENCE: item.get('description', None),
                    DATASET_EVENTS: item.get('events', None),
                    DATASET_N_SUBJECTS: item.get('n_subjects', None),
                    DATASET_N_SHOTS: item.get('n_shots', None),
                    DATASET_DURATION: item.get('duration', None),
                    DATASET_SOURCE: item.get('source', None)
                    # 'ground_truth': item.get('answer', item.get('caption'))
                }
                self._samples.append(sample)
            else:
                self.logger.warning(f"Video not found: {video_path}")


class ETBenchCaptioningDataset(BaseDataset):
    """E.T. Bench Captioning dataset"""
    
    def _load_dataset(self):
        """Load E.T. Bench Captioning dataset"""
        annotation_file = Path(self.config.get('annotation_file'))
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        for i, item in enumerate(annotations):
            video_path = self.data_path / "videos" / item['video']
            if video_path.exists():
                sample = {
                    DATASET_ID: item.get('idx', -1),
                    DATASET_VIDEO_PATH: str(video_path),
                    DATASET_PROMPT: item.get('q', None),
                    DATASET_REFERENCE: item.get('tgt', None),
                    DATASET_EVENTS: item.get('g', None),
                    DATASET_DURATION: item.get('duration', None),
                    DATASET_SOURCE: item.get('source', None),
                    DATASET_TASK: item.get('task', None),
                    # 'ground_truth': item.get('answer', item.get('caption'))
                }
                self._samples.append(sample)
            else:
                self.logger.warning(f"Video not found: {video_path}")

                
class VideoHallucerDataset(BaseDataset):

    def _load_dataset(self):
        tasks = ["object_relation", "temporal", "semantic_detail", "interaction", "external_factual", "external_nonfactual", "fact_detect"]
        for task in tasks:
            annotation_file = Path(self.config.get('annotation_file')) / task / f"{task}.json"

            if not annotation_file.exists():
                raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
            
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
        
            for i, item in enumerate(annotations):
                video_path = self.data_path / task / "videos" / item['basic']['video']
                if video_path.exists():
                    sample = {
                        DATASET_VIDEO_PATH: str(video_path),
                        DATASET_PROMPT: [item['basic'].get('question', None), item['hallucination'].get('question', None)],
                        DATASET_REFERENCE: [item['basic'].get('answer', None), item['hallucination'].get('answer', None)],
                        # DATASET_PROMPT_HALLUCINATION: item['hallucination'].get('question', None),
                        # DATASET_REFERENCE_HALLUCINATION: item['hallucination'].get('answer', None),
                        DATASET_SOURCE: item.get('type', None),
                        DATASET_TASK: task,
                        # 'ground_truth': item.get('answer', item.get('caption'))
                    }
                    self._samples.append(sample)
                else:
                    self.logger.warning(f"Video not found: {video_path}")

class EventHallusionDataset(BaseDataset):

    def _load_dataset(self):
        tasks = ["entire", "interleave", "misleading"]
        for task in tasks:
            annotation_file = Path(self.config.get('annotation_file')) / f"{task}_questions.json"

            if not annotation_file.exists():
                raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
            
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
        
            for i, item in enumerate(annotations):
                id = item['id']
                video_path = self.data_path / task / f"{id}.mp4"
                questions = item['questions']
                answer_prompt = "\nPlease answer yes or no:"
                desc_question = "Please describe this video in detail."
                if task == "interleave":
                    desc_reference = item['event_info']["unexpected"]
                else:
                    desc_reference = item['event_info']["caption"]
                if video_path.exists():
                    sample = {
                        DATASET_ID: id.split('_')[1],
                        DATASET_VIDEO_PATH: str(video_path),
                        DATASET_PROMPT: [q.get("question", None) + answer_prompt for q in questions] + [desc_question],
                        DATASET_REFERENCE: [q.get("answer", None) for q in questions] + [desc_reference],
                        DATASET_DURATION: item.get("length", None),
                        DATASET_SOURCE: item.get('category', None),
                        DATASET_TASK: task,
                        # 'ground_truth': item.get('answer', item.get('caption'))
                    }
                    assert len(sample[DATASET_PROMPT]) == len(sample[DATASET_REFERENCE]), "#prompts doesn't match with #references!"
                    self._samples.append(sample)
                else:
                    self.logger.warning(f"Video not found: {video_path}")

class PerceptionTestDataset(BaseDataset):

    def __init__(self, config):
        super().__init__(config)

        self.mc_prompt = "Watch the video and answer the follownig question:\n{}\n0) {}\n1) {}\n 2) {}\nProvide your answer in the format 'X' where X is the number of your choice."

    def _load_dataset(self):
        annotation_file = Path(self.config.get('annotation_file'))

        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
    
        for video in annotations:
            info = annotations[video]
            video_path = self.data_path / f"{video}.mp4"
            mc_questions = info['mc_question']
            prompts, answers = [], []
            for qa in mc_questions:
                question = qa['question']
                options = qa['options']
                prompts.append(self.mc_prompt.format(question, options[0], options[1], options[2]))
                answers.append(str(qa['answer_id']))

            if video_path.exists():
                sample = {
                    DATASET_VIDEO_PATH: str(video_path),
                    DATASET_PROMPT: prompts,
                    DATASET_REFERENCE: answers,
                }
                assert len(sample[DATASET_PROMPT]) == len(sample[DATASET_REFERENCE]), "#prompts doesn't match with #references!"
                self._samples.append(sample)
            else:
                self.logger.warning(f"Video not found: {video_path}")

class Test50Dataset(BaseDataset):
    """Test-50 benchmark dataset
    
    数据集结构：
    test_50/
    ├── movie_animation/     (1-10.mp4)
    ├── movie_live_action/   (201-210.mp4)
    ├── shorts/              (401-410.mp4)
    ├── stock/               (611-620.mp4)
    └── youtube/             (801-810.mp4)
    """ 

    def __init__(self, config):
        super().__init__(config)
        
        # 定义提示模板 - 电影制作报告
        self.caption_prompt = "For a film production report, describe the camera and lighting setup in this video"
        self.category_map = {
            'movie_animation': 'animated movie',
            'movie_live_action': 'live action movie',
            'shorts': 'short video',
            'stock': 'stock footage',
            'youtube': 'YouTube video'
        }

    def _load_dataset(self):
        """Load Test-50 dataset"""
        
        # 如果提供了annotation_file，则从文件加载
        annotation_file = self.config.get('annotation_file')
        if annotation_file:
            annotation_file = Path(annotation_file)
            if annotation_file.exists():
                self._load_from_annotation_file(annotation_file)
                return
            else:
                self.logger.warning(f"Annotation file not found: {annotation_file}, loading from directory structure")
        
        # 否则，从目录结构加载
        self._load_from_directory()

    def _load_from_annotation_file(self, annotation_file):
        """从标注文件加载数据"""
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        for item in annotations:
            video_id = item.get('video_id') or item.get('id')
            category = item.get('category', 'unknown')
            video_path = self.data_path / category / f"{video_id}.mp4"
            
            if video_path.exists():
                sample = {
                    DATASET_ID: video_id,
                    DATASET_VIDEO_PATH: str(video_path),
                    DATASET_PROMPT: item.get('prompt', self.caption_prompt),
                    DATASET_REFERENCE: item.get('caption') or item.get('description') or item.get('answer'),
                    DATASET_SOURCE: category,
                }
                self._samples.append(sample)
            else:
                self.logger.warning(f"Video not found: {video_path}")

    def _load_from_directory(self):
        """从目录结构加载数据（无标注文件）"""
        categories = ['movie_animation', 'movie_live_action', 'shorts', 'stock', 'youtube']
        
        for category in categories:
            category_path = self.data_path / category
            if not category_path.exists():
                self.logger.warning(f"Category directory not found: {category_path}")
                continue
            
            # 获取该类别下的所有mp4文件
            video_files = sorted(category_path.glob('*.mp4'))
            
            for video_path in video_files:
                video_id = video_path.stem  # 获取文件名（不含扩展名）
                
                sample = {
                    DATASET_ID: f"{category}_{video_id}",
                    DATASET_VIDEO_PATH: str(video_path),
                    DATASET_PROMPT: self.caption_prompt,
                    DATASET_REFERENCE: None,  # 如果没有标注，设为None
                    DATASET_SOURCE: category,
                    DATASET_TASK: 'video_captioning',
                }
                self._samples.append(sample)
        
        self.logger.info(f"Loaded {len(self._samples)} videos from {len(categories)} categories")

class Test50_dense_Dataset(BaseDataset):
    """Test-50 benchmark dataset
    
    数据集结构：
    test_50/
    ├── movie_animation/     (1-10.mp4)
    ├── movie_live_action/   (201-210.mp4)
    ├── shorts/              (401-410.mp4)
    ├── stock/               (611-620.mp4)
    └── youtube/             (801-810.mp4)
    """ 

    def __init__(self, config):
        super().__init__(config)
        
        # 定义提示模板 - 电影制作报告
        self.caption_prompt = "Please describe this video covering the central elements, their characteristics and positions, the setting and atmosphere, how things move or change, and what visual style and narrative intent the video has."
        self.category_map = {
            'movie_animation': 'animated movie',
            'movie_live_action': 'live action movie',
            'shorts': 'short video',
            'stock': 'stock footage',
            'youtube': 'YouTube video'
        }

    def _load_dataset(self):
        """Load Test-50 dataset"""
        
        # 如果提供了annotation_file，则从文件加载
        annotation_file = self.config.get('annotation_file')
        if annotation_file:
            annotation_file = Path(annotation_file)
            if annotation_file.exists():
                self._load_from_annotation_file(annotation_file)
                return
            else:
                self.logger.warning(f"Annotation file not found: {annotation_file}, loading from directory structure")
        
        # 否则，从目录结构加载
        self._load_from_directory()

    def _load_from_annotation_file(self, annotation_file):
        """从标注文件加载数据"""
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        for item in annotations:
            video_id = item.get('video_id') or item.get('id')
            category = item.get('category', 'unknown')
            video_path = self.data_path / category / f"{video_id}.mp4"
            
            if video_path.exists():
                sample = {
                    DATASET_ID: video_id,
                    DATASET_VIDEO_PATH: str(video_path),
                    DATASET_PROMPT: item.get('prompt', self.caption_prompt),
                    DATASET_REFERENCE: item.get('caption') or item.get('description') or item.get('answer'),
                    DATASET_SOURCE: category,
                }
                self._samples.append(sample)
            else:
                self.logger.warning(f"Video not found: {video_path}")

    def _load_from_directory(self):
        """从目录结构加载数据（无标注文件）"""
        categories = ['movie_animation', 'movie_live_action', 'shorts', 'stock', 'youtube']
        
        for category in categories:
            category_path = self.data_path / category
            if not category_path.exists():
                self.logger.warning(f"Category directory not found: {category_path}")
                continue
            
            # 获取该类别下的所有mp4文件
            video_files = sorted(category_path.glob('*.mp4'))
            
            for video_path in video_files:
                video_id = video_path.stem  # 获取文件名（不含扩展名）
                
                sample = {
                    DATASET_ID: f"{category}_{video_id}",
                    DATASET_VIDEO_PATH: str(video_path),
                    DATASET_PROMPT: self.caption_prompt,
                    DATASET_REFERENCE: None,  # 如果没有标注，设为None
                    DATASET_SOURCE: category,
                    DATASET_TASK: 'video_captioning',
                }
                self._samples.append(sample)
        
        self.logger.info(f"Loaded {len(self._samples)} videos from {len(categories)} categories")


class TVBenchDataset(BaseDataset):

    def __init__(self, config):
        super().__init__(config)

        self.scqa_prompt = "Watch the video and answer the follownig single-choice question:\n{}\nChoose the correct option from the following canditates:\n{}\nProvide your answer in the format 'xxx' where xxx is the correct option from the candidates."

    def _load_dataset(self):
        # tasks = ["action_count", "action_localization", "action_sequence", "egocentric_sequence", "moving_direction", "object_count", "object_shuffle", "scene_transition", "unexpected_action", "action_antonym"]
        tasks = ["action_localization"]
        for task in tasks:
            annotation_file = Path(self.config.get('annotation_file')) / f"{task}.json"

            if not annotation_file.exists():
                raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
            
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
        
            for i, item in enumerate(annotations):
                video = item.get("video", None)
                video_path = self.data_path / task / video
                question = item.get("question", None)
                candidates = item.get("candidates", None)
                if video_path.exists():
                    sample = {
                        DATASET_ID: str(i),
                        DATASET_VIDEO_PATH: str(video_path),
                        DATASET_PROMPT: self.scqa_prompt.format(question, candidates),
                        DATASET_REFERENCE: item.get("answer", None),
                        DATASET_DURATION: item.get("video_length", None),
                        DATASET_TASK: task,
                        # 'ground_truth': item.get('answer', item.get('caption'))
                    }
                    self._samples.append(sample)
                else:
                    self.logger.warning(f"Video not found: {video_path}")

class TOMATODataset(BaseDataset):

    def __init__(self, config):
        super().__init__(config)

        self.scqa_prompt = "Watch the video and answer the follownig single-choice question:\n{}\nChoose the correct option from the following canditates:\n{}\nProvide your answer in the format 'xxx' where xxx is the correct option from the candidates."

    def _load_dataset(self):
        tasks = ["count", "direction", "rotation", "shape&trend", "velocity&frequency", "visual_cues"]
        for task in tasks:
            annotation_file = Path(self.config.get('annotation_file')) / f"{task}.json"

            if not annotation_file.exists():
                raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
            
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
        
            for id, item in annotations.items():
                video = item.get("key", None)
                source = item.get("demonstration_type", None)
                video_path = self.data_path / source / f"{video}.mp4"
                question = item.get("question", None)
                options = item.get("options", None)
                if video_path.exists():
                    sample = {
                        DATASET_ID: id,
                        DATASET_VIDEO_PATH: str(video_path),
                        DATASET_PROMPT: self.scqa_prompt.format(question, options),
                        DATASET_REFERENCE: item.get("answer", None),
                        DATASET_OPTIONS: item.get("options", None),
                        DATASET_TASK: task,
                        DATASET_SOURCE: source,
                        # 'ground_truth': item.get('answer', item.get('caption'))
                    }
                    self._samples.append(sample)
                else:
                    self.logger.warning(f"Video not found: {video_path}")



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
