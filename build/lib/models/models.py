"""
Model registry and implementations for VLM benchmark system
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import asyncio
import aiohttp
import torch
import logging
from pathlib import Path
import json
import base64
from PIL import Image
import io

# Try importing model-specific libraries (they might not all be available)
try:
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import scripts.openai_test as openai_test
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class BaseVLMModel(ABC):
    """Abstract base class for VLM models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
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


class OpenAIVLMModel(BaseVLMModel):
    """OpenAI GPT-4o Vision model"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        self.client = None
        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
    
    async def load_model(self):
        """Initialize OpenAI client"""
        self.client = openai_test.AsyncOpenAI(api_key=self.api_key)
        self.logger.info("OpenAI client initialized")
    
    async def unload_model(self):
        """No explicit unloading needed for API models"""
        self.client = None
    
    async def predict(self, sample: Dict[str, Any]) -> str:
        """Generate prediction using OpenAI API"""
        if not self.client:
            await self.load_model()
        
        try:
            # Load and encode image
            image = self._load_image(sample['image_path'])
            image_base64 = self._encode_image_base64(image)
            
            # Prepare messages
            prompt = sample.get('prompt', "Describe this image in detail.")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
            
            # Generate response
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=self.config.get('max_tokens', 1024),
                temperature=self.config.get('temperature', 0.0)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
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


class HuggingFaceVLMModel(BaseVLMModel):
    """Local Hugging Face VLM model (Qwen, Tarsier, etc.)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not HF_AVAILABLE:
            raise ImportError("Transformers library not available. Install with: pip install transformers")
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = self._get_device(config.get('device', 'auto'))
        self.model_path = config.get('model_path')
        
        if not self.model_path:
            raise ValueError("Model path is required for local models")
    
    def _get_device(self, device_str: str) -> torch.device:
        """Get appropriate device"""
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)
    
    async def load_model(self):
        """Load Hugging Face model"""
        try:
            self.logger.info(f"Loading model from {self.model_path}")
            
            # Load model components
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True
            )
            
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
            except:
                # Fallback to tokenizer if processor not available
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    async def unload_model(self):
        """Unload model to free memory"""
        self.model = None
        self.processor = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Model unloaded")
    
    async def predict(self, sample: Dict[str, Any]) -> str:
        """Generate prediction using local model"""
        if not self.model:
            await self.load_model()
        
        try:
            # Load image
            image = self._load_image(sample['image_path'])
            prompt = sample.get('prompt', "Describe this image in detail.")
            
            # Process inputs
            if self.processor:
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
            else:
                # Custom processing if processor not available
                inputs = self._custom_process_inputs(prompt, image)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.get('max_tokens', 1024),
                    temperature=self.config.get('temperature', 0.0),
                    do_sample=self.config.get('temperature', 0.0) > 0,
                    pad_token_id=self.model.config.eos_token_id
                )
            
            # Decode response
            if self.processor:
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
            else:
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from response if present
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in local model prediction: {e}")
            raise
    
    def _custom_process_inputs(self, prompt: str, image: Image.Image):
        """Custom input processing when processor is not available"""
        # This would need to be implemented based on specific model requirements
        # For now, return a basic tokenized version
        if self.tokenizer:
            return self.tokenizer(prompt, return_tensors="pt").to(self.device)
        else:
            raise NotImplementedError("Custom input processing not implemented")


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
            "qwen2.5vl-7b", "qwen2.5vl-32b", "qwen2.5vl-72b", 
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