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
    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer, AutoConfig, GenerationConfig
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

try:
    from vllm import LLM, SamplingParams
    from vllm.multimodal import MultiModalDataDict
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# try:
#     from transformers import LlavaConfig
#     from tarsier.modeling_tarsier import TarsierForConditionalGeneration  # trust_remote_code=True underneath
#     from tarsier.data.processor import init_processor, format_one_sample
# except Exception as e:
#     LlavaConfig = None
#     TarsierForConditionalGeneration = None
#     init_processor = None
#     format_one_sample = None
#     _TARSIER_IMPORT_ERROR = e


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
    
    async def _predict_one_sample(self, prompt, image = None, video = None):
        # method is not finished, need to implement image and video encoding
        if not self.client:
            await self.load_model()

        messages=[
                {
                    "role": "user",
                    "content": prompt,
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
            # Load and encode image
            # image = self._load_image(sample['image_path'])
            # image_base64 = self._encode_image_base64(image)
            
            # Prepare messages
            # image_path = sample.get("image_path")
            # video_path = sample.get("video_path")

            # prompts = self._check_prompt(sample)
            prompts = sample.get(DATASET_PROMPT, None)
            #always str for now
            if isinstance(prompts, str):
                return await self._predict_one_sample(prompts)
            elif isinstance(prompts, list):
                responses = []
                for prompt in prompts:
                    response = await self._predict_one_sample(prompt)
                    responses.append(response)

                return responses
            # messages = [
            #     {
            #         "role": "user",
            #         "content": [
            #             {"type": "text", "text": prompt},
            #             {
            #                 "type": "image_url",
            #                 "image_url": {
            #                     "url": f"data:image/jpeg;base64,{image_base64}"
            #                 }
            #             }
            #         ]
            #     }
            # ]

                #                 self.client.get_api.chat.completions.create(
                #         model=model,
                #         messages=messages,
                #     ),
                #     timeout=self.timeout_s
                # )
                # content = resp.choices[0].message.content or ""
            

            
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
    """Local Hugging Face VLM model (Qwen, Tarsier, etc.) with image & video support.

    Video handling:
      - Provide sample['video_path'] to caption a video.
      - Frames are evenly sampled (configurable) and passed to the processor.
      - First tries processor(..., videos=frames); if not supported, falls back to images=list_of_PIL_frames.
    """

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

        # Video config
        self.num_video_frames: int = int(config.get('num_video_frames', 16))  # evenly-sampled
        self.max_video_frames: int = int(config.get('max_video_frames', 32))
        self.resize: Optional[Tuple[int, int]] = config.get('resize')  # e.g., (width, height) or None
        self.video_backend: str = config.get('video_backend', 'auto')  # 'auto'|'opencv'

        # Generation config
        self.generate_kwargs = {
            "max_new_tokens": int(config.get('max_new_tokens', 1024)),
            "top_p": float(config.get('top_p', 1.0)),
            "temperature": float(config.get('temperature', 0.000001)),
        }
        # self.generate_kwargs = dict(self.config.get("generate_kwargs", self.generate_kwargs))

    def _get_device(self, device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    async def load_model(self):
        """Load Hugging Face model"""

        self.logger.info(f"Loading model from {self.model_path}")

        # Let transformers place weights; we'll move inputs at call time.
        # Passing a torch.device to device_map is incorrect; use 'auto' instead.
        model_config = AutoConfig.from_pretrained(self.model_path,
                                            trust_remote_code=False)

        # print(model_config)
        # if 'auto_map' in model_config:
        # model_config['vision_config']['auto_map']['AutoConfig'] = "submodules/tarsier/" + model_config['vision_config']['auto_map']['AutoConfig']
        # model_config['text_config']['auto_map']['AutoConfig'] = "submodules/tarsier/" + model_config['text_config']['auto_map']['AutoConfig']
        # model_config['vision_config']['auto_map']['AutoModel'] = "submodules/tarsier/" + model_config['vision_config']['auto_map']['AutoModel']
        # model_config['text_config']['auto_map']['AutoModel'] = "submodules/tarsier/" + model_config['text_config']['auto_map']['AutoModel']

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map='auto',
            trust_remote_code=True
        )
        # self.processor = init_processor(self.model_path, self.config)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.logger.info("Model loaded successfully")
    # exit()

        # except Exception as e:
        #     self.logger.error(f"Error loading model: {e}")
        #     raise

    async def unload_model(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Model unloaded")

    # ---------- Public API ----------

    @abstractmethod
    async def _predict_one_sample(self, prompt: str, image_path: Optional[str] = None, video_path: Optional[str] = None, cache = None) -> str:
        pass

    async def predict(self, sample: Dict[str, Any]) -> str:
        """Generate prediction for image OR video.
        sample:
          - image_path: str, optional
          - video_path: str, optional
          - prompt: str, optional
        """
        if self.model is None or self.processor is None:
            await self.load_model()

        image_path = sample.get("image_path")
        video_path = sample.get("video_path")
        start_time = sample.get("start_time")
        end_time = sample.get("end_time")

        prompts = self._check_prompt(sample)

        if isinstance(prompts, str):
            return await self._predict_one_sample(prompts, image_path, video_path)
        elif isinstance(prompts, list):
            responses = []
            for prompt in prompts:
                response = await self._predict_one_sample(prompt, image_path, video_path)
                responses.append(response)

            return responses
        else:
            print('no')
        # if not self.model:
        #     await self.load_model()

        # try:
        #     prompt = sample.get('prompt')
        #     image_path = sample.get('image_path')
        #     video_path = sample.get('video_path')

        #     if video_path:
        #         frames = self._load_video_frames(
        #             video_path,
        #             num_frames=self.num_video_frames,
        #             resize=self.resize
        #         )
        #         if not frames:
        #             raise ValueError(f"No frames decoded from: {video_path}")
        #         if not prompt:
        #             prompt = "Describe this video in detail."
        #         inputs = self._prepare_inputs_for_video(prompt, frames)
        #     elif image_path:
        #         image = self._load_image(image_path)
        #         if not prompt:
        #             prompt = "Describe this image in detail."
        #         inputs = self._prepare_inputs_for_image(prompt, image)
        #     else:
        #         raise ValueError("sample must include either 'image_path' or 'video_path'")

        #     # Generate
        #     # print("generating predictions")
        #     with torch.no_grad():
        #         outputs = self.model.generate(
        #             **inputs,
        #             max_new_tokens=self.gen_max_new_tokens,
        #             temperature=self.temperature,
        #             do_sample=(self.temperature > 0.0),
        #             pad_token_id=self._pad_token_id()
        #         )

        #     response = self._decode(outputs)
        #     # Remove the input prompt from response if present
        #     if prompt in response:
        #         response = response.replace(prompt, "").strip()
        #     return response

        # except Exception as e:
        #     self.logger.error(f"Error in local model prediction: {e}")
        #     raise

    # ---------- Input prep ----------

    def _prepare_inputs_for_image(self, prompt: str, image: Image.Image) -> Dict[str, Any]:
        if self.processor is not None:
            try:
                proc = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                )
            except TypeError:
                # Some processors use "prompt" instead of "text"
                proc = self.processor(
                    prompt=prompt,
                    images=image,
                    return_tensors="pt"
                )
            return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in proc.items()}
        else:
            # Without a processor, we cannot pass image tensors generically
            return self._custom_process_inputs(prompt)

    def _prepare_inputs_for_video(self, prompt: str, frames: List[Image.Image]) -> Dict[str, Any]:
        """Try processor videos= first; fallback to images=list_of_frames."""
        if self.processor is None:
            raise NotImplementedError("Video requires a processor for multimodal inputs.")


        # # Cap to max_video_frames if needed
        # if self.max_video_frames and len(frames) > self.max_video_frames:
        #     step = math.ceil(len(frames) / self.max_video_frames)
        #     frames = frames[::step][:self.max_video_frames]

        # # Try the "videos=" API (some processors expect this)
        # try:
        #     proc = self.processor(
        #         # text=prompt,
        #         videos=frames,
        #         return_tensors="pt"
        #     )
        # except TypeError:
        #     print('type error')
        #     # Fallback: many HF VLMs accept image sequences via images=[...]
        #     try:
        #         proc = self.processor(
        #             text=prompt,
        #             images=frames,
        #             return_tensors="pt"
        #         )
        #     except TypeError:
        #         # Some use "prompt=" instead of "text="
        #         proc = self.processor(
        #             prompt=prompt,
        #             images=frames,
        #             return_tensors="pt"
        #         )

        # return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in proc.items()}

    # ---------- Media loading ----------

    def _check_prompt(self, sample: Dict[str, Any]) -> str:
        prompt = sample.get(DATASET_PROMPT)
        if prompt is None:
            return VIDEO_DEFAULT_PROMPT
        else:
            return prompt

    def _load_image(self, image_path: str) -> Image.Image:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = Image.open(image_path).convert("RGB")
        if self.resize:
            w, h = self.resize
            img = img.resize((w, h), Image.BICUBIC)
        return img

    def _load_video_frames(
        self,
        video_path: str,
        num_frames: int = 16,
        resize: Optional[Tuple[int, int]] = None
    ) -> List[Image.Image]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        if self.video_backend not in ("auto", "opencv"):
            raise ValueError("Unsupported video backend; use 'auto' or 'opencv'.")

        if (self.video_backend == "auto" and CV2_AVAILABLE) or self.video_backend == "opencv":
            return self._load_video_frames_opencv(video_path, num_frames, resize)
        raise RuntimeError("OpenCV backend unavailable. Install opencv-python or set video_backend to 'opencv' after installing.")

    def _load_video_frames_opencv(
        self,
        video_path: str,
        num_frames: int,
        resize: Optional[Tuple[int, int]]
    ) -> List[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            # Attempt to read sequentially if frame count is unknown
            frames_bgr = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames_bgr.append(frame)
            total = len(frames_bgr)
            indices = self._evenly_spaced_indices(total, num_frames)
            selected = [frames_bgr[i] for i in indices]
        else:
            indices = self._evenly_spaced_indices(total, num_frames)
            selected = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok:
                    break
                selected.append(frame)

        cap.release()

        frames: List[Image.Image] = []
        for bgr in selected:
            if resize:
                w, h = resize
                bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            frames.append(Image.fromarray(rgb))

        return frames

    @staticmethod
    def _evenly_spaced_indices(total: int, num: int) -> List[int]:
        if total <= 0:
            return []
        if num <= 1:
            return [0]
        if num >= total:
            return list(range(total))
        # Even spacing over [0, total-1]
        return [int(round(i * (total - 1) / (num - 1))) for i in range(num)]

    # ---------- Decoding & utility ----------

    def _decode(self, outputs: torch.Tensor) -> str:
        # Prefer processor.tokenizer if available
        if self.processor is not None:
            tok = getattr(self.processor, "tokenizer", None)
            if tok is not None:
                return tok.decode(outputs[0], skip_special_tokens=True)
            # Some processors forward decode
            if hasattr(self.processor, "decode"):
                return self.processor.decode(outputs[0], skip_special_tokens=True)

        if self.tokenizer is not None:
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Fallback: model may expose a tokenizer
        model_tok = getattr(self.model, "tokenizer", None)
        if model_tok is not None:
            return model_tok.decode(outputs[0], skip_special_tokens=True)

        raise RuntimeError("No tokenizer/decoder available to decode model outputs.")

    def _pad_token_id(self) -> Optional[int]:
        # Try model then tokenizer, then None
        for obj in (getattr(self.model, "config", None), self.tokenizer, getattr(self.processor, "tokenizer", None)):
            if obj is None:
                continue
            pad = getattr(obj, "pad_token_id", None)
            eos = getattr(obj, "eos_token_id", None)
            if pad is not None:
                return pad
            if eos is not None:
                return eos
        return None

    def _custom_process_inputs(self, prompt: str):
        """Custom input processing when processor is not available.
        NOTE: This path cannot support images/videos generically.
        """
        if self.tokenizer:
            return self.tokenizer(prompt, return_tensors="pt").to(self.device)
        else:
            raise NotImplementedError("Custom input processing not implemented")

class TarsierVLMModel(HuggingFaceVLMModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not TARSIER_AVAILABLE:
            raise ImportError("Tarsier library not available.")
        print("Tarsier model initialized")
    
    async def load_model(self):
    # try:

        self.logger.info(f"Loading model from {self.model_path}")

        # Let transformers place weights; we'll move inputs at call time.
        # Passing a torch.device to device_map is incorrect; use 'auto' instead.
        model_config = LlavaConfig.from_pretrained(self.model_path,
                                            trust_remote_code=False)
        # print(self.generate_kwargs)
        # print(model_config)
        # if 'auto_map' in model_config:
        # model_config['vision_config']['auto_map']['AutoConfig'] = "submodules/tarsier/" + model_config['vision_config']['auto_map']['AutoConfig']
        # model_config['text_config']['auto_map']['AutoConfig'] = "submodules/tarsier/" + model_config['text_config']['auto_map']['AutoConfig']
        # model_config['vision_config']['auto_map']['AutoModel'] = "submodules/tarsier/" + model_config['vision_config']['auto_map']['AutoModel']
        # model_config['text_config']['auto_map']['AutoModel'] = "submodules/tarsier/" + model_config['text_config']['auto_map']['AutoModel']

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
        # print(f"Loading Processor from {self.model_path}")
        # Try processor; fall back to tokenizer
        # try:
        # self.processor = AutoProcessor.from_pretrained(
        #     self.model_path,
        #     trust_remote_code=True
        # )
        # except Exception:
        #     self.processor = None
        #     self.tokenizer = AutoTokenizer.from_pretrained(
        #         self.model_path,
        #         trust_remote_code=True
        #     )

        self.logger.info("Model loaded successfully")
        # exit()

        # except Exception as e:
        #     self.logger.error(f"Error loading model: {e}")
        #     raise
    def _build_sample(self, prompt: Optional[str], image_path: Optional[str], video_path: Optional[str]) -> Dict[str, Any]:
        """
        Build the single-sample dict using the official helper format_one_sample.
        Tarsier's format_one_sample accepts either an image or a video path.
        """
        if video_path:
            final_prompt = prompt
            return format_one_sample(video_path, final_prompt)
        if image_path:
            final_prompt = prompt
            return format_one_sample(image_path, final_prompt)
        raise ValueError("Neither 'image_path' nor 'video_path' provided in sample.")

    @torch.inference_mode()
    async def _predict_one_sample(self, prompt: str, image_path: Optional[str] = None, video_path: Optional[str] = None, cache = None) -> str:

        if self.model is None or self.processor is None:
            await self.load_model()

        single = self._build_sample(prompt, image_path, video_path)
        # print(single)
        
        preprocessed = self.processor(single)
        # print(preprocessed)
        # print(preprocessed['pixel_values'].shape)
        # exit()

        # if cache is None:
        #     cache = self.processor(single)  # returns dict[str, Union[Tensor, ...]]
        # else:
        # # Video is already processed, only need to change the prompt now
        #     for i in range(len(cache['raw_data_dict'])):
        #         cache['raw_data_dict'][i] = single

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

class QwenVLMModel(HuggingFaceVLMModel):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not QWEN_AVAILABLE:
            raise ImportError("Qwen utils library not available. Install with: pip install qwen-vl-utils[decord]==0.0.8")
        print("qwen model initialized")

    def _build_sample(self, image_path: Optional[str], video_path: Optional[str], prompt: Optional[str]) -> Dict[str, Any]:
        if image_path:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
        if video_path:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": self.config.get("max_pixels", 460800),
                            "fps": self.config.get("fps", 1),
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

        return messages

    async def _predict_one_sample(self, prompt: str, image_path: Optional[str] = None, video_path: Optional[str] = None, cache = None) -> str:

        if self.model is None or self.processor is None:
            await self.load_model()
            
        messages = self._build_sample(image_path, video_path, prompt)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            # fps=fps,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")
        # print(inputs['pixel_values_videos'].shape)

        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return response[0]
    
# class TarsierVLMModel(HuggingFaceVLMModel):
#     """
#     Minimal async inference wrapper for Tarsier2, analogous to your QwenVLMModel.

#     Expected config keys (with defaults):
#       - model_name_or_path: str (required)
#       - data_config: str | dict (optional; passed to init_processor)
#       - device_map: 'auto' | dict | str = 'auto'
#       - torch_dtype: 'bfloat16' | 'float16' | 'float32' = 'bfloat16'
#       - max_new_tokens: int = 128
#       - generate_kwargs: dict = {}
#       - default_image_prompt: str = "Describe the image in detail."
#       - default_video_prompt: str = "Describe the video in detail."
#     """

#     def __init__(self, config: Dict[str, Any]):
#         super().__init__(config)
#         if any(x is None for x in [LlavaConfig, TarsierForConditionalGeneration, init_processor, format_one_sample]):
#             raise ImportError(
#                 f"Tarsier libraries not available (original error: {_TARSIER_IMPORT_ERROR}).\n"
#                 "Make sure you have the official Tarsier2 repo installed and importable."
#             )
#         self.model = None
#         self.processor = None

#         # Cache some frequently used config bits
#         self.model_name_or_path: str = self.config.get("model_name_or_path")
#         if not self.model_name_or_path:
#             raise ValueError("config['model_name_or_path'] is required for TarsierVLMModel.")

#         self.data_config = self.config.get("data_config")  # path or dict (optional, passed to init_processor)

#         self.device_map = self.config.get("device_map", "auto")
#         _dtype_str = str(self.config.get("torch_dtype", "bfloat16")).lower()
#         _dtype_map = {
#             "bfloat16": torch.bfloat16,
#             "bf16": torch.bfloat16,
#             "float16": torch.float16,
#             "fp16": torch.float16,
#             "float32": torch.float32,
#             "fp32": torch.float32,
#         }
#         self.torch_dtype = _dtype_map.get(_dtype_str, torch.bfloat16)

#         self.max_new_tokens = int(self.config.get("max_new_tokens", 128))
#         self.generate_kwargs = dict(self.config.get("generate_kwargs", {}))

#         self.default_image_prompt = self.config.get("default_image_prompt", "Describe the image in detail.")
#         self.default_video_prompt = self.config.get("default_video_prompt", "Describe the video in detail.")

#     async def load_model(self):
#         if self.model is not None and self.processor is not None:
#             return

#         # Processor
#         self.processor = init_processor(self.model_name_or_path, self.data_config)

#         # Model
#         model_config = LlavaConfig.from_pretrained(
#             self.model_name_or_path,
#             trust_remote_code=True,
#         )
#         self.model = TarsierForConditionalGeneration.from_pretrained(
#             self.model_name_or_path,
#             config=model_config,
#             device_map=self.device_map,
#             torch_dtype=self.torch_dtype,
#             trust_remote_code=True,
#         )
#         self.model.eval()

#     def _build_sample(self, image_path: Optional[str], video_path: Optional[str], prompt: Optional[str]) -> Dict[str, Any]:
#         """
#         Build the single-sample dict using the official helper format_one_sample.
#         Tarsier's format_one_sample accepts either an image or a video path.
#         """
#         if video_path:
#             final_prompt = prompt or self.default_video_prompt
#             return format_one_sample(video_path, final_prompt)
#         if image_path:
#             final_prompt = prompt or self.default_image_prompt
#             return format_one_sample(image_path, final_prompt)
#         raise ValueError("Neither 'image_path' nor 'video_path' provided in sample.")

#     @torch.inference_mode()
#     async def predict(self, sample: Dict[str, Any]) -> str:
#         """
#         sample keys:
#           - image_path: str (optional)
#           - video_path: str (optional)
#           - prompt: str (optional; overrides defaults above)
#         """
#         if self.model is None or self.processor is None:
#             await self.load_model()

#         image_path = sample.get("image_path")
#         video_path = sample.get("video_path")
#         prompt = sample.get("prompt")

#         # Build and tokenize
#         single = self._build_sample(image_path, video_path, prompt)
#         batch = self.processor(single)  # returns dict[str, Union[Tensor, ...]]

#         # Move only tensors to model device
#         model_inputs = {
#             k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
#             for k, v in batch.items()
#             if isinstance(v, torch.Tensor)
#         }

#         # Generate
#         outputs = self.model.generate(
#             **model_inputs,
#             max_new_tokens=self.max_new_tokens,
#             **self.generate_kwargs,
#         )

#         # Decode only the newly generated tokens (strip the prompt prefix)
#         # Official code uses: processor.processor.tokenizer (processor has an inner HF processor)
#         tokenizer = self.processor.processor.tokenizer
#         # Input length for this single example
#         input_len = model_inputs["input_ids"][0].shape[0]
#         output_ids = outputs[0][input_len:]
#         text = tokenizer.decode(output_ids, skip_special_tokens=True)

#         return text

# class HuggingFaceVLMModel(BaseVLMModel):
#     """Local Hugging Face VLM model (Qwen, Tarsier, etc.)"""
    
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__(config)
#         if not HF_AVAILABLE:
#             raise ImportError("Transformers library not available. Install with: pip install transformers")
        
#         self.model = None
#         self.processor = None
#         self.tokenizer = None
#         self.device = self._get_device(config.get('device', 'auto'))
#         self.model_path = config.get('model_path')
        
#         if not self.model_path:
#             raise ValueError("Model path is required for local models")
    
#     def _get_device(self, device_str: str) -> torch.device:
#         """Get appropriate device"""
#         if device_str == "auto":
#             return torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         return torch.device(device_str)
    
#     async def load_model(self):
#         """Load Hugging Face model"""
#         try:
#             self.logger.info(f"Loading model from {self.model_path}")
            
#             # Load model components
#             self.model = AutoModel.from_pretrained(
#                 self.model_path,
#                 torch_dtype=torch.bfloat16,
#                 device_map=self.device,
#                 trust_remote_code=True
#             )
            
#             try:
#                 self.processor = AutoProcessor.from_pretrained(
#                     self.model_path,
#                     trust_remote_code=True
#                 )
#             except:
#                 # Fallback to tokenizer if processor not available
#                 self.tokenizer = AutoTokenizer.from_pretrained(
#                     self.model_path,
#                     trust_remote_code=True
#                 )
            
#             self.logger.info("Model loaded successfully")
            
#         except Exception as e:
#             self.logger.error(f"Error loading model: {e}")
#             raise
    
#     async def unload_model(self):
#         """Unload model to free memory"""
#         self.model = None
#         self.processor = None
#         self.tokenizer = None
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         self.logger.info("Model unloaded")
    
#     async def predict(self, sample: Dict[str, Any]) -> str:
#         """Generate prediction using local model"""
#         if not self.model:
#             await self.load_model()
        
#         try:
#             # Load image
#             image = self._load_image(sample['image_path'])
#             prompt = sample.get('prompt', "Describe this image in detail.")
            
#             # Process inputs
#             if self.processor:
#                 inputs = self.processor(
#                     text=prompt,
#                     images=image,
#                     return_tensors="pt"
#                 ).to(self.device)
#             else:
#                 # Custom processing if processor not available
#                 inputs = self._custom_process_inputs(prompt, image)
            
#             # Generate
#             with torch.no_grad():
#                 outputs = self.model.generate(
#                     **inputs,
#                     max_new_tokens=self.config.get('max_tokens', 1024),
#                     temperature=self.config.get('temperature', 0.0),
#                     do_sample=self.config.get('temperature', 0.0) > 0,
#                     pad_token_id=self.model.config.eos_token_id
#                 )
            
#             # Decode response
#             if self.processor:
#                 response = self.processor.decode(outputs[0], skip_special_tokens=True)
#             else:
#                 response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
#             # Remove the input prompt from response if present
#             if prompt in response:
#                 response = response.replace(prompt, "").strip()
            
#             return response
            
#         except Exception as e:
#             self.logger.error(f"Error in local model prediction: {e}")
#             raise
    
#     def _custom_process_inputs(self, prompt: str, image: Image.Image):
#         """Custom input processing when processor is not available"""
#         # This would need to be implemented based on specific model requirements
#         # For now, return a basic tokenized version
#         if self.tokenizer:
#             return self.tokenizer(prompt, return_tensors="pt").to(self.device)
#         else:
#             raise NotImplementedError("Custom input processing not implemented")


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