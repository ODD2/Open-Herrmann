# Herrmann-1.0
This project re-implements the Herrmann-1.0 presented in [ICASSP 2024](https://audiomatic-research.github.io/herrmann-1/).

## Installation
```bash
# AudioCraft Preliminaries
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install audiocraft
conda install "ffmpeg<5" -c conda-forge
# MusicGen
pip install git+https://github.com/huggingface/transformers.git
# Herrmann
pip install scenedetect[opencv]

```
