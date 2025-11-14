"""
Model registry and implementations for VLM benchmark system
"""

import os, sys
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio
import aiohttp
import torch
import logging
from pathlib import Path
import json
import base64
from PIL import Image
import io
import math
from constants import *
import yaml

# Try importing model-specific libraries (they might not all be available)
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

try:
    from transformers import AutoProcessor, AutoTokenizer, AutoConfig, GenerationConfig
    # Try to import the new API first (transformers >= 5.0)
    try:
        from transformers import AutoModelForImageTextToText
        AutoModelForVision2Seq = AutoModelForImageTextToText  # Alias for compatibility
    except ImportError:
        # Fall back to old API for older transformers versions
        from transformers import AutoModelForVision2Seq
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import openai
    from openai import AzureOpenAI, AsyncAzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from qwen_vl_utils import process_vision_info
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False


try:
    from submodules.tarsier.models.modeling_tarsier import LlavaConfig, TarsierForConditionalGeneration
    from submodules.tarsier.dataset.tarsier_datamodule import init_processor
    from submodules.tarsier.dataset.utils import *
    TARSIER_AVAILABLE = True
except ImportError:
    TARSIER_AVAILABLE = False

# try:
#     from vllm import LLM, SamplingParams
#     from vllm.multimodal import MultiModalDataDict
#     VLLM_AVAILABLE = True
# except ImportError:
#     VLLM_AVAILABLE = False

VLLM_AVAILABLE = False

class BaseVLMModel(ABC):
    """Base VLM model with common functionality"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        
        if not HF_AVAILABLE:
            raise ImportError("Transformers library not available. Install with: pip install transformers")
        
        self.processor = None
        self.model_path = config.get('model_path')
        if not self.model_path:
            raise ValueError("Model path is required for local models")
        
        # Video config
        self.num_video_frames: int = int(config.get('num_video_frames', 16))
        self.max_video_frames: int = int(config.get('max_video_frames', 32))
        self.resize: Optional[Tuple[int, int]] = config.get('resize')
        self.video_backend: str = config.get('video_backend', 'auto')
        
        # Generation config
        self.generate_kwargs = {
            "max_new_tokens": int(config.get('max_new_tokens', 1024)),
            "top_p": float(config.get('top_p', 1.0)),
            "temperature": float(config.get('temperature', 0.000001)),
        }
    
    def _check_prompt(self, sample: Dict[str, Any]) -> str:
        prompt = sample.get(DATASET_PROMPT)
        if prompt is None:
            return VIDEO_DEFAULT_PROMPT
        else:
            return prompt
    
    async def predict(self, sample: Dict[str, Any]) -> Union[str, List[str]]:
        """Generate prediction for image OR video"""
        if not self.is_loaded():
            await self.load_model()
        
        image_path = sample.get("image_path")
        video_path = sample.get("video_path")
        prompts = self._check_prompt(sample)
        
        if isinstance(prompts, str):
            return await self._predict_one_sample(prompts, image_path, video_path)
        elif isinstance(prompts, list):
            return await self._predict_batch(prompts, image_path, video_path)
        else:
            raise ValueError("Invalid prompt format")
        
    @abstractmethod
    async def load_model(self):
        """Load the model"""
        pass
    
    @abstractmethod
    async def unload_model(self):
        """Unload the model"""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        pass
    
    @abstractmethod
    async def _predict_one_sample(self, prompt: str, image_path: Optional[str] = None,
                                video_path: Optional[str] = None) -> str:
        """Predict single sample"""
        pass
    
    @abstractmethod
    async def _predict_batch(self, prompts: List[str], image_path: Optional[str] = None,
                           video_path: Optional[str] = None) -> List[str]:
        """Predict batch of samples"""
        pass

class BaseVLMAPIModel(ABC):
    """Abstract base class for VLM models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def is_initialized(self):
        return self.client is not None
        
    @abstractmethod
    async def predict(self, sample: Dict[str, Any]) -> str:
        """
        Generate prediction for a single sample
        
        Args:
            sample: Dictionary containing 'image_path' and optional 'prompt'
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    async def load_model(self):
        """Load the model (if needed)"""
        pass
    
    @abstractmethod
    async def unload_model(self):
        """Unload the model to free memory"""
        pass
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load image from path"""
        return Image.open(image_path).convert('RGB')
    
    def _encode_image_base64(self, image: Image.Image) -> str:
        """Encode PIL image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

class OpenAIVLMModel(BaseVLMAPIModel):
    """OpenAI GPT-4o Vision model"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        # self.client = None
        self.api_key = config.get('api_key')
        self.api_version = config.get('api_version')
        self.endpoint = config.get('endpoint')
        self.model_name = config.get('model_name')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
    
    @property
    def get_api(self):
        return self.client

    async def load_model(self):
        """Initialize OpenAI client"""
        # self.client = test.AsyncOpenAI(api_key=self.api_key)
        self.client = AsyncAzureOpenAI(
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                api_key=self.api_key
            )
        self.logger.info("OpenAI client initialized")
    
    async def unload_model(self):
        """No explicit unloading needed for API models"""
        self.client = None
    
    def _extract_video_frames(self, video_path: str, num_frames: int = 8) -> List[Image.Image]:
        """Extract frames from video"""
        if not CV2_AVAILABLE:
            raise ImportError("cv2 not available. Install with: pip install opencv-python")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames evenly
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
        
        cap.release()
        return frames

    async def _predict_one_sample(self, prompt, image = None, video = None):
        """Generate prediction with optional image or video input"""
        if not self.client:
            await self.load_model()

        # Build content list
        content = [{"type": "text", "text": prompt}]
        
        # Handle video input - extract frames and add them
        if video is not None:
            try:
                num_frames = self.config.get('num_video_frames', 8)
                frames = self._extract_video_frames(video, num_frames=num_frames)
                
                for frame in frames:
                    frame_base64 = self._encode_image_base64(frame)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_base64}"
                        }
                    })
                self.logger.info(f"Added {len(frames)} video frames to request")
            except Exception as e:
                self.logger.warning(f"Failed to extract video frames: {e}")
        
        # Handle image input
        elif image is not None:
            try:
                if isinstance(image, str):
                    # Image path provided
                    image = self._load_image(image)
                
                image_base64 = self._encode_image_base64(image)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                })
                self.logger.info("Added image to request")
            except Exception as e:
                self.logger.warning(f"Failed to process image: {e}")

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.config.get('max_tokens', 1024),
            temperature=self.config.get('temperature', 0.0)
        )

        return response.choices[0].message.content

    async def predict(self, sample: Dict[str, Any]) -> str:
        """Generate prediction using OpenAI API"""
        if not self.client:
            await self.load_model()

        try:
            # Extract video or image path
            video_path = sample.get(DATASET_VIDEO_PATH)
            image_path = sample.get(DATASET_IMAGE_PATH)
            
            # Get prompts
            prompts = sample.get(DATASET_PROMPT, None)
            
            # Handle single or multiple prompts
            if isinstance(prompts, str):
                # Single prompt
                return await self._predict_one_sample(
                    prompts, 
                    image=image_path, 
                    video=video_path
                )
            elif isinstance(prompts, list):
                # Multiple prompts - process each with the same video/image
                responses = []
                for prompt in prompts:
                    try:
                        response = await self._predict_one_sample(
                            prompt, 
                            image=image_path, 
                            video=video_path
                        )
                        responses.append(response)
                    except Exception as e:
                        error_message = str(e)
                        if 'content_filter' in error_message or 'ResponsibleAIPolicyViolation' in error_message:
                            video_name = Path(video_path).name if video_path else (Path(image_path).name if image_path else "unknown")
                            self.logger.warning(f"Content filter triggered for {video_name}, prompt: {prompt[:50]}...")
                            responses.append(f"[Content filtered by Azure OpenAI policy]")
                        else:
                            raise
                return responses
            else:
                # No prompt provided, use default
                default_prompt = VIDEO_DEFAULT_PROMPT if video_path else IMAGE_DEFAULT_PROMPT
                return await self._predict_one_sample(
                    default_prompt,
                    image=image_path,
                    video=video_path
                )
            
        except Exception as e:
            # Check if it's a content filter error
            error_message = str(e)
            if 'content_filter' in error_message or 'ResponsibleAIPolicyViolation' in error_message:
                video_name = Path(video_path).name if video_path else (Path(image_path).name if image_path else "unknown")
                self.logger.warning(f"Content filter triggered for {video_name}, returning placeholder message")
                return f"[Content filtered by Azure OpenAI policy - video: {video_name}]"
            else:
                self.logger.error(f"Error in OpenAI prediction: {e}")
                raise


class GenericAPIModel(BaseVLMModel):
    """Generic API-based VLM model (for Doubao, Gemini, etc.)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.session = None
        self.api_endpoint = config.get('api_endpoint')
        self.api_key = config.get('api_key')
        
        if not self.api_endpoint:
            raise ValueError("API endpoint is required")
    
    async def load_model(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
        self.logger.info(f"API session initialized for {self.api_endpoint}")
    
    async def unload_model(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def predict(self, sample: Dict[str, Any]) -> str:
        """Generate prediction using generic API"""
        if not self.session:
            await self.load_model()
        
        try:
            # Load and encode image
            image = self._load_image(sample['image_path'])
            image_base64 = self._encode_image_base64(image)
            
            # Prepare request payload (this is generic - may need customization per API)
            prompt = sample.get('prompt', "Describe this image in detail.")
            payload = {
                "messages": [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "image": f"data:image/jpeg;base64,{image_base64}"}
                        ]
                    }
                ],
                "max_tokens": self.config.get('max_tokens', 1024),
                "temperature": self.config.get('temperature', 0.0)
            }
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Make API request
            async with self.session.post(
                self.api_endpoint,
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    # This parsing may need to be customized per API
                    return result.get('choices', [{}])[0].get('message', {}).get('content', '')
                else:
                    error_text = await response.text()
                    raise Exception(f"API request failed: {response.status} - {error_text}")
                    
        except Exception as e:
            self.logger.error(f"Error in API prediction: {e}")
            raise

class InferenceBackend:
    """Base class for inference backends"""
    
    def __init__(self, parent_model, config: Dict[str, Any]):
        self.parent = parent_model
        self.config = config
        self.logger = parent_model.logger
    
    async def load_model(self):
        raise NotImplementedError
    
    async def unload_model(self):
        raise NotImplementedError
    
    def is_loaded(self) -> bool:
        raise NotImplementedError
    
    async def predict_one_sample(self, messages: List[Dict[str, Any]]) -> str:
        raise NotImplementedError
    
    async def predict_batch(self, batch_messages: List[List[Dict[str, Any]]]) -> List[str]:
        raise NotImplementedError


class HuggingfaceInferenceBackend(InferenceBackend):
    """Native Hugging Face transformers inference backend"""
    
    def __init__(self, parent_model, config: Dict[str, Any]):
        super().__init__(parent_model, config)
        self.model = None
        self.device = self._get_device(config.get('device', 'auto'))
    
    def _get_device(self, device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)
    
    async def load_model(self):
        """Load Hugging Face model for native inference"""
        self.logger.info(f"Loading model from {self.parent.model_path} (native backend)")
        
        model_config = AutoConfig.from_pretrained(self.parent.model_path, trust_remote_code=False)
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.parent.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map='auto',
            trust_remote_code=True
        )
        
        self.parent.processor = AutoProcessor.from_pretrained(
            self.parent.model_path,
            trust_remote_code=True
        )
        
        self.logger.info("Native model loaded successfully")
    
    async def unload_model(self):
        """Unload native model and free memory"""
        self.model = None
        self.parent.processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Native model unloaded")
    
    def is_loaded(self) -> bool:
        """Check if native model is loaded"""
        return self.model is not None and self.parent.processor is not None
    
    async def predict_one_sample(self, messages: List[Dict[str, Any]]) -> str:
        """Native inference for single sample"""
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        
        text = self.parent.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.parent.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")
        
        generated_ids = self.model.generate(**inputs, **self.parent.generate_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.parent.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return response[0]
    
    async def predict_batch(self, batch_messages: List[List[Dict[str, Any]]]) -> List[str]:
        """Native inference processes sequentially"""
        responses = []
        for messages in batch_messages:
            response = await self.predict_one_sample(messages)
            responses.append(response)
        return responses


class VLLMInferenceBackend(InferenceBackend):
    """VLLM inference backend"""
    
    def __init__(self, parent_model, config: Dict[str, Any]):
        super().__init__(parent_model, config)
        self.vllm_model = None
        
        # VLLM specific config
        self.vllm_config = {
            # 'tensor_parallel_size': config.get('tensor_parallel_size', 1),
            'gpu_memory_utilization': config.get('gpu_memory_utilization', 0.9),
            'max_model_len': config.get('max_model_len', 8192),
            'trust_remote_code': config.get('trust_remote_code', True),
            'dtype': config.get('dtype', 'auto'),
        }
        # self.vllm_config = {
        #     'limit_mm_per_prompt': {"image": 10, "video": 10},
        # }
        
        # Convert generate_kwargs to VLLM SamplingParams format
        self.sampling_params = SamplingParams(
            max_tokens=self.parent.generate_kwargs['max_new_tokens'],
            top_p=self.parent.generate_kwargs['top_p'],
            temperature=self.parent.generate_kwargs['temperature'],
            stop_token_ids=config.get('stop_token_ids', None),
        )
    
    async def load_model(self):
        """Load model using VLLM"""
        self.logger.info(f"Loading model from {self.parent.model_path} (VLLM backend)")
        
        # Load processor for tokenization
        self.parent.processor = AutoProcessor.from_pretrained(
            self.parent.model_path,
            trust_remote_code=True
        )
        
        # Initialize VLLM model
        self.vllm_model = LLM(
            model=self.parent.model_path,
            **self.vllm_config
        )
        
        self.logger.info("VLLM model loaded successfully")
    
    async def unload_model(self):
        """Unload VLLM model and free memory"""
        self.vllm_model = None
        self.parent.processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("VLLM model unloaded")
    
    def is_loaded(self) -> bool:
        """Check if VLLM model is loaded"""
        return self.vllm_model is not None and self.parent.processor is not None
    
    async def predict_one_sample(self, messages: List[Dict[str, Any]]) -> str:
        """VLLM inference for single sample"""
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        
        text = self.parent.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Prepare multimodal data for VLLM
        multi_modal_data = {}
        if image_inputs:
            multi_modal_data["image"] = image_inputs[0] if len(image_inputs) == 1 else image_inputs
        if video_inputs:
            multi_modal_data["video"] = video_inputs[0] if len(video_inputs) == 1 else video_inputs
        
        # Generate with VLLM
        llm_inputs = {
            "prompt": text,
            "multi_modal_data": multi_modal_data,

            # FPS will be returned in video_kwargs
            "mm_processor_kwargs": video_kwargs,
        }

        outputs = self.vllm_model.generate([llm_inputs], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text
    
    async def predict_batch(self, batch_messages: List[List[Dict[str, Any]]]) -> List[str]:
        """Efficient batch processing with VLLM"""
        batch_texts = []
        batch_multi_modal_data = []
        
        # Prepare all samples
        for messages in batch_messages:
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            
            text = self.parent.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)
            
            # Prepare multimodal data
            multi_modal_data = {}
            if image_inputs:
                multi_modal_data["image"] = image_inputs[0] if len(image_inputs) == 1 else image_inputs
            if video_inputs:
                multi_modal_data["video"] = video_inputs[0] if len(video_inputs) == 1 else video_inputs
            
            batch_multi_modal_data.append(multi_modal_data if multi_modal_data else None)
        
        # Generate batch
        if any(data is not None for data in batch_multi_modal_data):
            outputs = self.vllm_model.generate(
                batch_texts,
                sampling_params=self.sampling_params,
                multi_modal_data=[MultiModalDataDict(data) if data else None for data in batch_multi_modal_data]
            )
        else:
            outputs = self.vllm_model.generate(batch_texts, sampling_params=self.sampling_params)
        
        return [output.outputs[0].text for output in outputs]


class QwenVLMModel(BaseVLMModel):
    """Qwen VLM model with automatic backend selection"""
    
    def __init__(self, config: Dict[str, Any]):
        if not QWEN_AVAILABLE:
            raise ImportError("Qwen utils library not available. Install with: pip install qwen-vl-utils[decord]==0.0.8")
        
        super().__init__(config)
        
        # Automatically choose the best available backend
        if VLLM_AVAILABLE:
            self.backend = VLLMInferenceBackend(self, config)
            backend_name = "VLLM"
        else:
            self.backend = HuggingfaceInferenceBackend(self, config)
            backend_name = "native"
        
        self.logger.info(f"Qwen model initialized with {backend_name} backend")
    
    def _extract_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata including fps"""
        if not CV2_AVAILABLE:
            self.logger.warning("cv2 not available, cannot extract video metadata")
            return {}
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            return {
                "fps": fps,
                "total_frames": total_frames,
                "duration": duration
            }
        except Exception as e:
            self.logger.warning(f"Failed to extract video metadata: {e}")
            return {}
    
    def _build_sample(self, image_path: Optional[str], video_path: Optional[str], 
                     prompt: Optional[str]) -> List[Dict[str, Any]]:
        """Build message format for Qwen model"""
        content = []
        
        if image_path:
            content.append({"type": "image", "image": image_path})
        elif video_path:
            video_content = {
                "type": "video",
                "video": video_path,
                "max_pixels": self.config.get("max_pixels", 460800),
            }
            
            # Check if this is Qwen3-VL or Qwen2.5-VL
            model_path = str(self.config.get("model_path", ""))
            is_qwen3 = "Qwen3" in model_path or "qwen3" in model_path.lower()
            
            if is_qwen3:
                # Qwen3-VL: Provide video_metadata for accurate fps extraction
                metadata = self._extract_video_metadata(video_path)
                if metadata and "fps" in metadata:
                    video_content["video_metadata"] = {"fps": metadata["fps"]}
                    self.logger.debug(f"Added video metadata with fps={metadata['fps']}")
            else:
                # Qwen2.5-VL: Use explicit fps parameter
                video_content["fps"] = self.config.get("fps", 1)
            
            content.append(video_content)
        
        content.append({"type": "text", "text": prompt})
        
        return [{"role": "user", "content": content}]
    
    async def load_model(self):
        """Load model using the selected backend"""
        await self.backend.load_model()
    
    async def unload_model(self):
        """Unload model using the selected backend"""
        await self.backend.unload_model()
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.backend.is_loaded()
    
    async def _predict_one_sample(self, prompt: str, image_path: Optional[str] = None,
                                video_path: Optional[str] = None) -> str:
        """Predict single sample using the selected backend"""
        messages = self._build_sample(image_path, video_path, prompt)
        return await self.backend.predict_one_sample(messages)
    
    async def _predict_batch(self, prompts: List[str], image_path: Optional[str] = None,
                           video_path: Optional[str] = None) -> List[str]:
        """Predict batch using the selected backend"""
        batch_messages = []
        for prompt in prompts:
            messages = self._build_sample(image_path, video_path, prompt)
            batch_messages.append(messages)
        
        return await self.backend.predict_batch(batch_messages)
    
    @property
    def backend_type(self) -> str:
        """Get the current backend type"""
        return "VLLM" if isinstance(self.backend, VLLMInferenceBackend) else "Native"

class TarsierVLMModel(BaseVLMModel):
    def __init__(self, config: Dict[str, Any]):
        if not TARSIER_AVAILABLE:
            raise ImportError("Tarsier library not available.")
        
        super().__init__(config)
    
        # Automatically choose the best available backend
        # if VLLM_AVAILABLE:
        #     self.backend = VLLMInferenceBackend(self, config)
        #     backend_name = "VLLM"
        # else:
        self.backend = HuggingfaceInferenceBackend(self, config)
        backend_name = "native"
        
        self.logger.info(f"Tarsier model initialized with {backend_name} backend")

    def _build_sample(self, image_path: Optional[str], video_path: Optional[str], 
                     prompt: Optional[str]) -> List[Dict[str, Any]]:
        """Build message format for Qwen model"""

        if video_path:
            final_prompt = prompt
            return [format_one_sample(video_path, final_prompt)]
        if image_path:
            final_prompt = prompt
            return [format_one_sample(image_path, final_prompt)]
        raise ValueError("Neither 'image_path' nor 'video_path' provided in sample.")

    async def load_model(self):
        """Load model using the selected backend"""
        self.logger.info(f"Loading model from {self.model_path}")

        model_config = LlavaConfig.from_pretrained(self.model_path,
                                            trust_remote_code=False)
        self.model = TarsierForConditionalGeneration.from_pretrained(
            self.model_path,
            config = model_config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map='auto',
            trust_remote_code=False
        )
        if self.model.config.pad_token_id is None:
            self.generate_kwargs.update({
                "pad_token_id": self.model.generation_config.eos_token_id
            })

        self.processor = init_processor(self.model_path, self.config)
        self.logger.info("Model loaded successfully")
    
    async def unload_model(self):
        """Unload model using the selected backend"""
        # await self.backend.unload_model()
        pass
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        # return self.backend.is_loaded()
        return self.model is not None
    
    async def _predict_one_sample(self, prompt: str, image_path: Optional[str] = None,
                                video_path: Optional[str] = None) -> str:
        if self.model is None or self.processor is None:
            await self.load_model()

        single = self._build_sample(image_path, video_path, prompt)
        # print(single)
        
        preprocessed = self.processor(single)

        # Move only tensors to model device
        model_inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in preprocessed.items()
            if isinstance(v, torch.Tensor)
        }
        # Generate
        outputs = self.model.generate(
            **model_inputs,
            # max_new_tokens=self.max_new_tokens,
            **self.generate_kwargs,
        )

        # Decode only the newly generated tokens (strip the prompt prefix)
        # Official code uses: processor.processor.tokenizer (processor has an inner HF processor)
        tokenizer = self.processor.processor.tokenizer
        # Input length for this single example
        input_len = model_inputs["input_ids"][0].shape[0]
        output_ids = outputs[0][input_len:]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)

        return response
    
    async def _predict_batch(self, prompts: List[str], image_path: Optional[str] = None,
                           video_path: Optional[str] = None) -> List[str]:
        """Predict batch using the selected backend"""
        responses = []
        for prompt in prompts:
            response = await self._predict_one_sample(prompt, image_path, video_path)
            responses.append(response)
        
        return responses
    
    @property
    def backend_type(self) -> str:
        """Get the current backend type"""
        return "VLLM" if isinstance(self.backend, VLLMInferenceBackend) else "Native"

class ModelRegistry:
    """Registry for managing different VLM models"""
    
    def __init__(self):
        self.models: Dict[str, BaseVLMModel] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_model(self, name: str, model: BaseVLMModel, config: Dict[str, Any]):
        """Register a model instance"""
        self.models[name] = model
        self.model_configs[name] = config
        self.logger.info(f"Model '{name}' registered")
    
    def list_models(self) -> List[str]:
        """List all available models"""
        # Return both registered models and known model types
        known_models = [
            "gpt-4o", "doubao-vl", "gemini-pro-vision",
            "qwen2.5vl-7b", "qwen2.5vl-32b", "qwen2.5vl-72b", "qwen2.5vl-8b", 
            "tarsier2"
        ]
        return list(set(list(self.models.keys()) + known_models))
    
    def load_model(self, name: str, config: Dict[str, Any]) -> BaseVLMModel:
        """Load a model by name"""
        if name in self.models:
            return self.models[name]
        
        # Create new model instance based on configuration
        model_type = config.get('type', 'local')
        
        if name.startswith('gpt-4') or 'openai' in name.lower():
            model = OpenAIVLMModel(config)
        elif model_type == 'api':
            model = GenericAPIModel(config)
        elif 'qwen' in name.lower():
            model = QwenVLMModel(config)
        elif 'tarsier' in name.lower():
            model = TarsierVLMModel(config)
        elif model_type == 'local':
            model = HuggingFaceVLMModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Register and return
        self.register_model(name, model, config)
        return model
    
    def unload_model(self, name: str):
        """Unload a specific model"""
        if name in self.models:
            asyncio.create_task(self.models[name].unload_model())
            del self.models[name]
            self.logger.info(f"Model '{name}' unloaded")
    
    def unload_all_models(self):
        """Unload all models"""
        for name in list(self.models.keys()):
            self.unload_model(name)