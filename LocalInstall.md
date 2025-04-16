## Windows

```cmd
python -m venv .venv
.venv\Scripts\activate

# Failed
#pip install -r requirements.txt

# Failed
pip install faster-whisper
```

[cudnn ops64\_9.dll is not found](https://github.com/SYSTRAN/faster-whisper/issues/1080)
```cmd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 2025/4/16 Ubuntu

```shell
uv venv
chmod +x .venv/bin/activate
source .venv/bin/activate
uv pip install faster-whisper
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
python transcribe.py ./audio/Sample1/Sample1.mp3 # failed

# find / -name "libcudnn*.so*" 2>/dev/null
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
#python
#    import torch
#    print("CUDA Available:", torch.cuda.is_available())
#    print("cuDNN Enabled:", torch.backends.cudnn.enabled)
python transcribe.py ./audio/Sample1/Sample1.mp3
```