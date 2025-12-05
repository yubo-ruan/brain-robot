# Brain-Robot

Brain-inspired LLM-guided robot control for LIBERO benchmark.

## Installation

```bash
# Clone the repository
git clone https://github.com/yubo-ruan/brain-robot.git
cd brain-robot

# Install dependencies (includes LIBERO)
pip install -r requirements.txt

# Download Qwen2.5-VL-7B model (~16GB, required for VLM features)
python scripts/download_models.py
```

> **Note**: Model weights are not included in the repo. Run the download script after cloning.

## License

MIT License
