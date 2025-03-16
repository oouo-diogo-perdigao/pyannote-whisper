# Instruções para usar
No wsl Instalação, ja com o python 3.10 instalado 
- `sudo apt install python3.10-venv`
- `python -m venv .venv`
- `source .venv/bin/activate`
- `sudo apt update && sudo apt install ffmpeg -y`
- `ffmpeg -version`
- `pip install -r requirements.txt`
- `pip install -qq torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 torchtext==0.12.0`
- `pip install -qq speechbrain==0.5.12`
- `pip install -qq pyannote.audio==2.1.1`
- `pip install -qq git+https://github.com/openai/whisper.git`
- `pip install "numpy<2"`
- `pip uninstall torch torchvision torchaudio`
- `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- `nvidia-smi`
- `pip install -U -r requirements.txt`
- `pip install -U speechbrain pyannote.audio whisper`
- `pip install "pytorch-lightning>=2.0.0"`
- `pip install python-dotenv`

	Acesse https://huggingface.co/settings/tokens
	Crie um novo token com permissão de "write"
- `huggingface-cli login`
	chave_do_hug
	n
- `export HUGGINGFACE_TOKEN="chave_do_hug"`
	transcribe.py adicionar chave
Teste:
- `python -m pyannote_whisper.cli.transcribe data/afjiv.wav --model tiny --diarization True`

```
python -m pyannote_whisper.cli.transcribe "/mnt/f/OneDrive/Music/RPG Roque/S09/2024-08-01 - Espadas da Luz Reveladora.mp3" \
  --model medium \
  --diarization True \
  --task transcribe \
  --language pt \
  --device cuda \
  --output_format "VTT"
```

Codigo pratico que lista e cria os audios.

- `pip install tabulate`
- `pip install pynput`
- `sudo apt install python3-xlib`
- `python rpgvtt.py "/mnt/f/OneDrive/Music/RPG Roque"` ou 
- `python rpgvtt.py "/mnt/f/OneDrive/Music/RPG Roque" --teste`