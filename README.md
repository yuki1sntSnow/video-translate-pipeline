# Video Translation Pipeline

End-to-end workflow: download → translate/fix SRT with LLM → generate TTS → compose final video with optional burned subtitles.

## Features
- YouTube download via `yt-dlp` (cookies/proxy supported)
- LLM subtitle clean + translate (OpenAI-compatible API)
- Fish Speech TTS (voice clone from `ref/`)
- FFmpeg compose with audio modes: replace / mix / bgm and overlap handling

## Requirements
- Python 3.10+
- FFmpeg on PATH
- PyTorch + torchaudio (install CUDA wheel if you have GPU)
- Fish Speech checkpoint `openaudio-s1-mini` in `checkpoints/`

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Download the TTS model:
- Get `openaudio-s1-mini` from Fish Speech releases and place under `checkpoints/openaudio-s1-mini/`.

Config:
```bash
copy env.example .env   # fill OPENAI_API_KEY and other values
```

Reference voice:
```
ref/
  cxk.wav   # 5–15s reference
  cxk.txt   # transcript of the reference
```

## Usage
Full pipeline from URL:
```bash
python pipeline.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

Start from a specific stage:
```bash
python pipeline.py --start-from translate --video video.mp4 --srt video.en.srt
python pipeline.py --start-from tts --video video.mp4 --srt translate/video.Chinese.srt
python pipeline.py --start-from compose --video video.mp4 --srt translate/video.Chinese.srt
```

Key CLI flags (all have env equivalents in `.env`):
- `--url` / `--video` / `--srt`
- `--target-language` (default Chinese)
- `--speaker` (reference name in `ref/`)
- `--overlap-strategy` truncate | speed | hybrid | none
- `--audio-mode` replace | mix | bgm
- `--burn-subs` to burn subtitles into video

## Environment variables (set in `.env`)
- OpenAI: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MODEL_NAME`, `PROXY` (optional)
- Download: `VIDEO_URL`, `DOWNLOAD_DIR`, `SUB_LANGS`, `AUTO_SUBS`, `SUBS_ONLY`, `YTDLP_FORMAT`, `EXTRACTOR_ARGS`, `COOKIES_FROM_BROWSER`, `COOKIES_FILE`
- Translation: `TRANSLATE_DIR`, `TARGET_LANGUAGE`
- TTS: `TTS_INPUT_DIR`, `TTS_OUTPUT_DIR`, `TTS_SPEAKER`, `TTS_DAC_CHECKPOINT`, `TTS_T2S_CHECKPOINT`, `TTS_DEVICE`, `TTS_TEMPERATURE`, `TTS_TOP_P`, `TTS_REPETITION_PENALTY`, `TTS_SEED`, `TTS_HALF`, `TTS_COMPILE`
- Compose: `FINAL_OUTPUT_DIR`, `AUDIO_MODE`, `ORIGINAL_VOLUME`, `TTS_VOLUME`, `BURN_SUBS`, `OVERLAP_STRATEGY`, `MAX_TTS_SPEED`, `SEPARATION_METHOD`
- Runtime: `TEMP_DIR`, `LOG_DIR`, `ISOLATE_RUN`, `SRT_MIN_CHARS`

## Notes
- Each run uses timestamped subfolders for isolation.
- Subtitle cleaning strips invisible chars and normalizes TTS tail punctuation to `。` without changing the SRT file.
- Overlap strategy controls how long TTS segments are sped up or trimmed to fit subtitle slots. Use `none` to keep raw durations (may overlap).
