"""Centralized configuration - all environment variables in one place."""

import os
from pathlib import Path


def env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from env var."""
    val = os.getenv(key, "")
    if not val:
        return default
    return val.lower() not in ("0", "false", "no", "off", "")


def env_int(key: str, default: int) -> int:
    """Get int from env var."""
    return int(os.getenv(key, str(default)))


def env_float(key: str, default: float) -> float:
    """Get float from env var."""
    return float(os.getenv(key, str(default)))


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATA_BASE_DIR = os.getenv("DATA_BASE_DIR", "data")
LOG_DIR = os.getenv("LOG_DIR", "logs")
REF_DIR = os.getenv("REF_DIR", "ref")
CHECKPOINTS_DIR = os.getenv("CHECKPOINTS_DIR", "checkpoints")

# ============================================================
# OpenAI / LLM
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
PROXY = os.getenv("PROXY", "")

# ============================================================
# Download (yt-dlp)
# ============================================================
VIDEO_URL = os.getenv("VIDEO_URL", "")
LOCAL_VIDEO = os.getenv("LOCAL_VIDEO", "")
LOCAL_SRT = os.getenv("LOCAL_SRT", "")
SUB_LANGS = os.getenv("SUB_LANGS", "en")
AUTO_SUBS = env_bool("AUTO_SUBS", False)
SUBS_ONLY = env_bool("SUBS_ONLY", False)
NO_DOWNLOAD_SUBS = env_bool("NO_DOWNLOAD_SUBS", False)
YTDLP_FORMAT = os.getenv("YTDLP_FORMAT", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best")
EXTRACTOR_ARGS = os.getenv("EXTRACTOR_ARGS", "")
COOKIES_FROM_BROWSER = os.getenv("COOKIES_FROM_BROWSER", "")
COOKIES_FILE = os.getenv("COOKIES_FILE", "config/cookies.txt")

# ============================================================
# ASR (Whisper)
# ============================================================
ASR_MODEL = os.getenv("ASR_MODEL", "large-v3")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cuda")
ASR_BATCH_SIZE = env_int("ASR_BATCH_SIZE", 16)
ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "float16")
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "")
ASR_MODEL_DIR = os.getenv("ASR_MODEL_DIR", "checkpoints/whisper")
HF_TOKEN = os.getenv("HF_TOKEN", "")
SKIP_ASR = env_bool("SKIP_ASR", False)

# ============================================================
# Multi-Speaker (Diarization)
# ============================================================
MULTI_SPEAKER = env_bool("MULTI_SPEAKER", False)
ASR_MIN_SPEAKERS = env_int("ASR_MIN_SPEAKERS", 0)
ASR_MAX_SPEAKERS = env_int("ASR_MAX_SPEAKERS", 0)
SPEAKER_MAPPING = os.getenv("SPEAKER_MAPPING", "")

# ============================================================
# Translation
# ============================================================
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "Chinese")
TRANSLATE_CHUNK_SIZE = env_int("TRANSLATE_CHUNK_SIZE", 200)
TRANSLATE_MAX_CONCURRENT = env_int("TRANSLATE_MAX_CONCURRENT", 5)
TRANSLATE_MAX_WAIT = env_int("TRANSLATE_MAX_WAIT", 300)
TRANSLATE_POLL_INTERVAL = env_int("TRANSLATE_POLL_INTERVAL", 2)

# ============================================================
# TTS (Fish Speech)
# ============================================================
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "")
TTS_DAC_CHECKPOINT = os.getenv("TTS_DAC_CHECKPOINT", "checkpoints/openaudio-s1-mini/codec.pth")
TTS_T2S_CHECKPOINT = os.getenv("TTS_T2S_CHECKPOINT", "checkpoints/openaudio-s1-mini")
TTS_DEVICE = os.getenv("TTS_DEVICE", "cuda")
TTS_HALF = env_bool("TTS_HALF", False)
TTS_COMPILE = env_bool("TTS_COMPILE", False)
TTS_TEMPERATURE = env_float("TTS_TEMPERATURE", 0.3)
TTS_TOP_P = env_float("TTS_TOP_P", 0.7)
TTS_REPETITION_PENALTY = env_float("TTS_REPETITION_PENALTY", 1.2)
TTS_SEED = env_int("TTS_SEED", 42)
TTS_VOLUME = env_float("TTS_VOLUME", 1.0)

# ============================================================
# Compose
# ============================================================
AUDIO_MODE = os.getenv("AUDIO_MODE", "replace")  # replace | mix | bgm
ORIGINAL_VOLUME = env_float("ORIGINAL_VOLUME", 0.15)
BURN_SUBS = env_bool("BURN_SUBS", True)
OVERLAP_STRATEGY = os.getenv("OVERLAP_STRATEGY", "hybrid")  # truncate | speed | hybrid | none
MAX_TTS_SPEED = env_float("MAX_TTS_SPEED", 1.3)
SEPARATION_METHOD = os.getenv("SEPARATION_METHOD", "")  # demucs | spleeter | ""

# ============================================================
# Subtitle Style (ASS)
# ============================================================
SUB_FONT_NAME = os.getenv("SUB_FONT_NAME", "Microsoft YaHei")
SUB_FONT_SIZE = env_int("SUB_FONT_SIZE", 45)
SUB_BOLD = env_bool("SUB_BOLD", False)
SUB_ITALIC = env_bool("SUB_ITALIC", False)
SUB_PRIMARY_COLOR = os.getenv("SUB_PRIMARY_COLOR", "&H00FFFFFF")
SUB_SECONDARY_COLOR = os.getenv("SUB_SECONDARY_COLOR", "&H000000FF")
SUB_OUTLINE_COLOR = os.getenv("SUB_OUTLINE_COLOR", "&H00000000")
SUB_BACK_COLOR = os.getenv("SUB_BACK_COLOR", "&H80000000")
SUB_OUTLINE = env_float("SUB_OUTLINE", 2.0)
SUB_SHADOW = env_float("SUB_SHADOW", 1.0)
SUB_ALIGNMENT = env_int("SUB_ALIGNMENT", 2)
SUB_MARGIN_L = env_int("SUB_MARGIN_L", 30)
SUB_MARGIN_R = env_int("SUB_MARGIN_R", 30)
SUB_MARGIN_V = env_int("SUB_MARGIN_V", 25)
SUB_BORDER_STYLE = env_int("SUB_BORDER_STYLE", 1)
SUB_MAX_LINE_WIDTH = env_float("SUB_MAX_LINE_WIDTH", 22.0)
SUB_MAX_LINES = env_int("SUB_MAX_LINES", 2)

# ============================================================
# Misc
# ============================================================
SRT_MIN_CHARS = env_int("SRT_MIN_CHARS", 10)
DEBUG = env_bool("DEBUG", False)


def get_tts_config() -> dict:
    """Get TTS configuration as dict for tts.py / text2semantic_inference.py."""
    return {
        "dac_checkpoint": TTS_DAC_CHECKPOINT,
        "t2s_checkpoint": TTS_T2S_CHECKPOINT,
        "device": TTS_DEVICE,
        "half": TTS_HALF,
        "compile": TTS_COMPILE,
        "temperature": TTS_TEMPERATURE,
        "top_p": TTS_TOP_P,
        "repetition_penalty": TTS_REPETITION_PENALTY,
        "seed": TTS_SEED,
    }


def get_llm_config() -> dict:
    """Get LLM configuration as dict."""
    return {
        "api_key": OPENAI_API_KEY,
        "base_url": OPENAI_BASE_URL,
        "model": MODEL_NAME,
        "proxy": PROXY,
    }


def get_subtitle_style_config() -> dict:
    """Get subtitle style configuration as dict."""
    return {
        "font_name": SUB_FONT_NAME,
        "font_size": SUB_FONT_SIZE,
        "bold": SUB_BOLD,
        "italic": SUB_ITALIC,
        "primary_color": SUB_PRIMARY_COLOR,
        "secondary_color": SUB_SECONDARY_COLOR,
        "outline_color": SUB_OUTLINE_COLOR,
        "back_color": SUB_BACK_COLOR,
        "outline": SUB_OUTLINE,
        "shadow": SUB_SHADOW,
        "alignment": SUB_ALIGNMENT,
        "margin_l": SUB_MARGIN_L,
        "margin_r": SUB_MARGIN_R,
        "margin_v": SUB_MARGIN_V,
        "border_style": SUB_BORDER_STYLE,
    }
