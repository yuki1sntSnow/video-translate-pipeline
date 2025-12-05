"""ASR Module - WhisperX speech recognition with optional speaker diarization."""

import gc
import os
import subprocess
from pathlib import Path

import torch
import whisperx


# ===== Configuration =====
DEFAULT_DEVICE = "cuda"
DEFAULT_BATCH_SIZE = 16
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_MODEL = "large-v3"
DEFAULT_WHISPER_DIR = "checkpoints/whisper"


def extract_audio(video_path: Path, audio_path: Path) -> Path:
    """
    Extract audio from video file using FFmpeg.

    Args:
        video_path: Input video file (MP4, etc.)
        audio_path: Output audio file (MP3)

    Returns:
        Path to extracted audio file
    """
    video_path = Path(video_path).resolve()
    audio_path = Path(audio_path).resolve()

    if audio_path.exists():
        print(f"[ASR] 音频文件已存在: {audio_path}")
        return audio_path

    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    print(f"[ASR] 提取音频: {video_path.name} → {audio_path.name}")

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",  # 覆盖已存在文件
                "-i",
                str(video_path),
                "-vn",  # 不包含视频
                "-acodec",
                "libmp3lame",
                "-q:a",
                "2",  # 音频质量 (0-9, 越小质量越好)
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"[ASR] ✓ 音频提取完成: {audio_path}")
        return audio_path
    except subprocess.CalledProcessError as e:
        print(f"[ASR] FFmpeg 错误:")
        print(f"  返回码: {e.returncode}")
        print(f"  错误输出: {e.stderr}")
        raise RuntimeError(f"FFmpeg 音频提取失败: {e.stderr}") from e
    except FileNotFoundError:
        raise RuntimeError("FFmpeg 未安装或不在 PATH 中，请先安装 FFmpeg")


def _patch_torch_load():
    """Patch torch.load to allow full model weights loading."""
    original_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = trusted_load
    return original_load


def _cleanup_gpu():
    """Release GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: list[dict], include_speaker: bool = False) -> str:
    """
    Convert WhisperX segments to SRT format.

    Args:
        segments: List of segment dicts with 'start', 'end', 'text' keys
        include_speaker: If True, include speaker label in text

    Returns:
        SRT formatted string
    """
    srt_lines = []
    idx = 0

    for segment in segments:
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        text = segment.get("text", "").strip()

        if not text:
            continue

        idx += 1
        start_ts = format_timestamp(start)
        end_ts = format_timestamp(end)

        # Add speaker label if available and requested
        if include_speaker:
            speaker = segment.get("speaker", "SPEAKER_00")
            text = f"[{speaker}] {text}"

        srt_lines.append(f"{idx}")
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.append(text)
        srt_lines.append("")  # Empty line between entries

    return "\n".join(srt_lines)


def transcribe_audio(
    audio_path: Path,
    device: str = DEFAULT_DEVICE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    model_name: str = DEFAULT_MODEL,
    model_dir: str = DEFAULT_WHISPER_DIR,
    language: str | None = None,
    multi_speaker: bool = False,
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict:
    """
    Transcribe audio file using WhisperX.

    Args:
        audio_path: Path to audio file
        device: Device to use ('cuda' or 'cpu')
        batch_size: Batch size for transcription
        compute_type: Compute type ('float16', 'int8', etc.)
        model_name: Whisper model name ('large-v3', 'large-v2', etc.)
        model_dir: Directory for model cache
        language: Force language code (e.g., 'en', 'zh'), None for auto-detect
        multi_speaker: Enable speaker diarization
        hf_token: HuggingFace token for speaker diarization model
        min_speakers: Minimum number of speakers (optional)
        max_speakers: Maximum number of speakers (optional)

    Returns:
        Dict with 'segments', 'language', and optionally 'speakers' keys
    """
    audio_path = Path(audio_path).resolve()

    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    print(f"[ASR] 加载模型: {model_name} ({compute_type})")
    print(f"[ASR] 设备: {device}, 批大小: {batch_size}")
    if multi_speaker:
        print(f"[ASR] 多说话人模式: 已启用")

    # Patch torch.load for full model loading
    _patch_torch_load()

    # Load model
    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        download_root=model_dir,
    )

    # Load and transcribe audio
    print(f"[ASR] 转录中: {audio_path.name}")
    audio = whisperx.load_audio(str(audio_path))

    transcribe_options = {"batch_size": batch_size}
    if language:
        transcribe_options["language"] = language

    result = model.transcribe(audio, **transcribe_options)
    detected_language = result.get("language", "en")
    print(f"[ASR] 检测到语言: {detected_language}")
    print(f"[ASR] 初步转录完成，共 {len(result['segments'])} 段")

    # Cleanup transcription model
    del model
    _cleanup_gpu()

    # Alignment for accurate timestamps
    print(f"[ASR] 对齐时间戳...")
    model_a, metadata = whisperx.load_align_model(
        language_code=detected_language,
        device=device,
    )

    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    print(f"[ASR] ✓ 对齐完成，共 {len(result['segments'])} 段")

    # Cleanup alignment model
    del model_a
    _cleanup_gpu()

    # Speaker diarization (optional)
    speakers = set()
    if multi_speaker:
        token = hf_token or os.getenv("HF_TOKEN")
        if not token:
            print(f"[ASR] ⚠ 警告: 未提供 HF_TOKEN，跳过说话人分离")
        else:
            print(f"[ASR] 进行说话人分离...")
            from whisperx.diarize import DiarizationPipeline

            diarize_model = DiarizationPipeline(use_auth_token=token, device=device)
            diarize_segments = diarize_model(
                audio,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Collect unique speakers
            for seg in result["segments"]:
                if "speaker" in seg:
                    speakers.add(seg["speaker"])

            print(
                f"[ASR] ✓ 说话人分离完成，检测到 {len(speakers)} 个说话人: {sorted(speakers)}"
            )

            del diarize_model
            _cleanup_gpu()

    return {
        "segments": result["segments"],
        "language": detected_language,
        "speakers": sorted(speakers) if speakers else [],
    }


# Supported audio formats (skip extraction if input is already audio)
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".opus"}


def transcribe_video(
    video_path: str | Path,
    output_dir: str | Path | None = None,
    device: str = DEFAULT_DEVICE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    model_name: str = DEFAULT_MODEL,
    model_dir: str = DEFAULT_WHISPER_DIR,
    language: str | None = None,
    keep_audio: bool = True,
    multi_speaker: bool = False,
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> Path:
    """
    Full pipeline: Extract audio from video (if needed) and transcribe to SRT.

    Args:
        video_path: Input video or audio file path
        output_dir: Output directory (default: same as input)
        device: Device to use ('cuda' or 'cpu')
        batch_size: Batch size for transcription
        compute_type: Compute type
        model_name: Whisper model name
        model_dir: Model cache directory
        language: Force language code, None for auto-detect
        keep_audio: Keep extracted audio file (only applies when extracting from video)
        multi_speaker: Enable speaker diarization
        hf_token: HuggingFace token for diarization model
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers

    Returns:
        Path to output SRT file
    """
    input_path = Path(video_path).resolve()

    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Check if input is already an audio file
    is_audio_input = input_path.suffix.lower() in AUDIO_EXTENSIONS

    # Define output paths
    srt_path = output_dir / f"{input_path.stem}.srt"

    print(f"[ASR] ===== 开始处理 =====")
    if is_audio_input:
        print(f"[ASR] 音频: {input_path}")
        audio_path = input_path  # Use input directly
    else:
        print(f"[ASR] 视频: {input_path}")
        audio_path = output_dir / f"{input_path.stem}.mp3"
    print(f"[ASR] 输出目录: {output_dir}")
    if multi_speaker:
        print(f"[ASR] 多说话人模式: 启用")

    # Step 1: Extract audio (skip if input is already audio)
    if is_audio_input:
        print(f"[ASR] 输入已是音频格式，跳过提取步骤")
    else:
        extract_audio(input_path, audio_path)

    # Step 2: Transcribe
    result = transcribe_audio(
        audio_path,
        device=device,
        batch_size=batch_size,
        compute_type=compute_type,
        model_name=model_name,
        model_dir=model_dir,
        language=language,
        multi_speaker=multi_speaker,
        hf_token=hf_token,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    # Step 3: Save SRT (include speaker if multi_speaker enabled)
    srt_content = segments_to_srt(result["segments"], include_speaker=multi_speaker)
    srt_path.write_text(srt_content, encoding="utf-8")
    print(f"[ASR] ✓ SRT 已保存: {srt_path}")

    if result.get("speakers"):
        print(f"[ASR] 检测到说话人: {result['speakers']}")

    # Optional: remove audio file (only if we extracted it)
    if not keep_audio and not is_audio_input and audio_path.exists():
        audio_path.unlink()
        print(f"[ASR] 已删除临时音频文件")

    print(f"[ASR] ===== 处理完成 =====")

    return srt_path


# ===== CLI =====
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR - Video to SRT")
    parser.add_argument("video", help="Video file path")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-d", "--device", default=os.getenv("ASR_DEVICE", "cuda"))
    parser.add_argument(
        "-b", "--batch-size", type=int, default=int(os.getenv("ASR_BATCH_SIZE", "16"))
    )
    parser.add_argument("-m", "--model", default=os.getenv("ASR_MODEL", "large-v3"))
    parser.add_argument(
        "-l", "--language", help="Force language code (e.g., 'en', 'zh')"
    )
    parser.add_argument(
        "--no-keep-audio",
        action="store_true",
        help="Delete extracted audio after processing",
    )
    parser.add_argument(
        "--multi-speaker", action="store_true", help="Enable speaker diarization"
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="HuggingFace token for diarization",
    )
    parser.add_argument("--min-speakers", type=int, help="Minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, help="Maximum number of speakers")

    args = parser.parse_args()

    srt_path = transcribe_video(
        video_path=args.video,
        output_dir=args.output,
        device=args.device,
        batch_size=args.batch_size,
        model_name=args.model,
        language=args.language,
        keep_audio=not args.no_keep_audio,
        multi_speaker=args.multi_speaker,
        hf_token=args.hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    print(f"\nOutput: {srt_path}")
