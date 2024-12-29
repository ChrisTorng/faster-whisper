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