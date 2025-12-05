#!/usr/bin/env python3
"""Download required model weights for brain-robot."""

import os
from pathlib import Path

def download_qwen_model():
    """Download Qwen2.5-VL-7B-Instruct model."""
    from huggingface_hub import snapshot_download
    
    model_dir = Path(__file__).parent.parent / "models" / "qwen2.5-vl-7b"
    
    if model_dir.exists() and any(model_dir.glob("*.safetensors")):
        print(f"Model already exists at {model_dir}")
        return
    
    print("Downloading Qwen2.5-VL-7B-Instruct...")
    snapshot_download(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        local_dir=str(model_dir),
    )
    print(f"Model downloaded to {model_dir}")

if __name__ == "__main__":
    download_qwen_model()
