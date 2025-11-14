from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download

model_name = "Qwen/Qwen3-VL-8B-Instruct"
cache_dir = "/home/dyvm6xra/dyvm6xrauser04/models"

# 1) Download all model files to a local directory (like git lfs but via API)
local_dir = snapshot_download(
    repo_id=model_name,
    local_dir=cache_dir,
    local_dir_use_symlinks=False,  # set True if you prefer symlinks into HF cache
    revision=None,                 # or a specific commit hash/tag
    ignore_patterns=["*.md", "LICENSE", ".*"],  # optional: skip non-essential files
)

print(f"Model snapshot downloaded to: {local_dir}")