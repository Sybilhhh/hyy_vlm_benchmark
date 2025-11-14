#!/usr/bin/env python3
"""
Script to download VLM models from HuggingFace
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def download_model(model_name: str, output_dir: str):
    """Download model from HuggingFace"""
    try:
        from transformers import AutoModel, AutoProcessor
        
        print(f"Downloading {model_name}...")
        model = AutoModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")

if __name__ == "__main__":
    models_to_download = [
        ("Qwen/Qwen2.5-VL-7B-Instruct", "./models/Qwen2.5-VL-7B-Instruct"),
        ("TIGER-Lab/Tarsier2-7B", "./models/Tarsier2-7B"),
    ]
    
    for model_name, output_dir in models_to_download:
        download_model(model_name, output_dir)
