"""
Audio/Video Mixing and Composition

Combines original video, translated subtitles, and TTS audio using FFmpeg.

Audio Modes:
    - replace: Replace original audio with TTS audio
    - mix: Mix TTS with lowered original audio (ducking)
    - dual: Keep both tracks as separate audio streams

Features:
    - Vocal/BGM separation interface (placeholder for demucs/spleeter)
    - Subtitle burning option
    - Audio timing alignment with subtitles
"""

import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path

from srt_utils import parse_srt, SubtitleEntry, parse_timestamp


@dataclass
class AudioSegment:
    """Audio segment with timing info."""

    path: Path
    start_time: float  # seconds
    duration: float  # seconds


def check_ffmpeg() -> bool:
    """Check if FFmpeg is available."""
    return shutil.which("ffmpeg") is not None


def get_audio_duration(audio_path: Path) -> float:
    """Get audio file duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def get_video_duration(video_path: Path) -> float:
    """Get video file duration in seconds."""
    return get_audio_duration(video_path)


def separate_vocals(
    audio_path: Path, output_dir: Path, method: str = "demucs"
) -> tuple[Path, Path]:
    """
    Separate vocals from background music.

    Args:
        audio_path: Input audio file
        output_dir: Output directory for separated tracks
        method: Separation method ("demucs" or "spleeter")

    Returns:
        (vocals_path, bgm_path) tuple

    Note: This is a placeholder. Actual implementation requires:
        - demucs: pip install demucs
        - spleeter: pip install spleeter
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = output_dir / f"{audio_path.stem}_vocals.wav"
    bgm_path = output_dir / f"{audio_path.stem}_bgm.wav"

    if method == "demucs":
        # Demucs separation
        cmd = [
            "demucs",
            "--two-stems",
            "vocals",
            "-o",
            str(output_dir),
            str(audio_path),
        ]
        try:
            subprocess.run(cmd, check=True)
            # Demucs outputs to output_dir/htdemucs/track_name/
            demucs_out = output_dir / "htdemucs" / audio_path.stem
            if (demucs_out / "vocals.wav").exists():
                shutil.copy(demucs_out / "vocals.wav", vocals_path)
                shutil.copy(demucs_out / "no_vocals.wav", bgm_path)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(f"Demucs separation failed: {e}")

    elif method == "spleeter":
        # Spleeter separation
        cmd = [
            "spleeter",
            "separate",
            "-o",
            str(output_dir),
            "-p",
            "spleeter:2stems",
            str(audio_path),
        ]
        try:
            subprocess.run(cmd, check=True)
            # Spleeter outputs to output_dir/track_name/
            spleeter_out = output_dir / audio_path.stem
            if (spleeter_out / "vocals.wav").exists():
                shutil.copy(spleeter_out / "vocals.wav", vocals_path)
                shutil.copy(spleeter_out / "accompaniment.wav", bgm_path)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(f"Spleeter separation failed: {e}")

    else:
        raise ValueError(f"Unknown separation method: {method}")

    return vocals_path, bgm_path


def extract_audio_from_video(video_path: Path, output_path: Path) -> Path:
    """Extract audio track from video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",  # No video
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def process_segment_overlap(
    segments: list[AudioSegment],
    srt_entries: list["SubtitleEntry"],
    strategy: str = "truncate",
    max_speed: float = 1.5,
) -> list[tuple[AudioSegment, str]]:
    """
    Process audio segments to handle overlapping issues.

    Args:
        segments: List of audio segments with timing
        srt_entries: Original SRT entries for end time calculation
        strategy: "truncate" (cut excess), "speed" (speed up), "hybrid", or "none"
        max_speed: Maximum speed multiplier for "speed" strategy

    Returns:
        List of (segment, filter_string) tuples
    """
    from srt_utils import parse_timestamp

    if strategy == "none":
        # Do not alter segments; place exactly by start time
        return [(seg, "") for seg in segments]

    processed = []
    overlap_count = 0
    speed_count = 0
    truncate_count = 0

    for i, seg in enumerate(segments):
        # Find the corresponding SRT entry
        matching_entry = None
        for entry in srt_entries:
            if abs(parse_timestamp(entry.start_time) - seg.start_time) < 0.1:
                matching_entry = entry
                break

        if matching_entry is None:
            # No matching entry, use as-is
            processed.append((seg, ""))
            continue

        # Calculate available time slot
        entry_end = parse_timestamp(matching_entry.end_time)
        slot_duration = entry_end - seg.start_time

        # Check if TTS exceeds the slot
        if seg.duration > slot_duration + 0.1:  # 0.1s tolerance
            overlap_count += 1
            overflow = seg.duration - slot_duration

            if strategy == "truncate":
                # Simply truncate to fit
                filter_str = f"atrim=0:{slot_duration}"
                truncate_count += 1
                processed.append((seg, filter_str))

            elif strategy == "speed":
                # Speed up to fit in slot
                speed_factor = min(seg.duration / slot_duration, max_speed)
                if speed_factor <= max_speed:
                    filter_str = f"atempo={speed_factor:.3f}"
                    speed_count += 1
                else:
                    # Too much speedup needed, truncate instead
                    filter_str = f"atempo={max_speed},atrim=0:{slot_duration}"
                    truncate_count += 1
                processed.append((seg, filter_str))

            elif strategy == "hybrid":
                # Use speed for small overflows, truncate for large ones
                speed_factor = seg.duration / slot_duration
                if speed_factor <= max_speed:
                    # Speed up is acceptable
                    filter_str = f"atempo={speed_factor:.3f}"
                    speed_count += 1
                else:
                    # Too fast, just truncate
                    filter_str = f"atrim=0:{slot_duration}"
                    truncate_count += 1
                processed.append((seg, filter_str))
            else:
                processed.append((seg, ""))
        else:
            # Fits in slot, no processing needed
            processed.append((seg, ""))

    if overlap_count > 0:
        print(f"[Compose] ⚠ Detected {overlap_count} overlapping segments")
        if speed_count > 0:
            print(f"[Compose]   → {speed_count} segments sped up")
        if truncate_count > 0:
            print(f"[Compose]   → {truncate_count} segments truncated")

    return processed


def concat_audio_with_timing(
    segments: list[AudioSegment],
    output_path: Path,
    total_duration: float,
    sample_rate: int = 44100,
    srt_entries: list["SubtitleEntry"] | None = None,
    overlap_strategy: str = "hybrid",
    max_speed: float = 1.5,
) -> Path:
    """
    Concatenate audio segments with silence padding based on timing.

    Creates a single audio file where each segment is placed at its start_time.

    Args:
        segments: List of audio segments
        output_path: Output file path
        total_duration: Total duration of output
        sample_rate: Audio sample rate
        srt_entries: Original SRT entries for overlap detection
        overlap_strategy: "truncate", "speed", "hybrid", or "none"
        max_speed: Maximum speed multiplier (default 1.5x)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process overlapping segments if SRT entries provided
    if srt_entries and overlap_strategy != "none":
        processed_segments = process_segment_overlap(
            segments, srt_entries, overlap_strategy, max_speed
        )
    else:
        processed_segments = [(seg, "") for seg in segments]

    # Build complex filter for positioning audio segments
    inputs = []
    filter_parts = []

    for i, (seg, extra_filter) in enumerate(processed_segments):
        inputs.extend(["-i", str(seg.path)])
        # Add delay to position the segment at correct time
        delay_ms = int(seg.start_time * 1000)

        if extra_filter:
            # Apply overlap processing filter, then delay
            filter_parts.append(
                f"[{i}]{extra_filter},adelay={delay_ms}|{delay_ms}[a{i}]"
            )
        else:
            filter_parts.append(f"[{i}]adelay={delay_ms}|{delay_ms}[a{i}]")

    # Mix all delayed segments with volume boost
    # Using loudnorm for consistent volume (not dynaudnorm which causes gradual increase)
    mix_inputs = "".join(f"[a{i}]" for i in range(len(processed_segments)))
    filter_parts.append(
        f"{mix_inputs}amix=inputs={len(processed_segments)}:duration=longest:normalize=0,"
        f"volume=3.0[out]"
    )

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-t",
        str(total_duration),
        "-ar",
        str(sample_rate),
        str(output_path),
    ]

    subprocess.run(cmd, check=True)
    return output_path


def build_tts_audio_track(
    srt_entries: list[SubtitleEntry],
    tts_output_dir: Path,
    output_path: Path,
    video_duration: float,
    overlap_strategy: str = "hybrid",
    max_speed: float = 1.5,
) -> Path:
    """
    Build a single audio track from TTS segments aligned to subtitle timing.

    Args:
        srt_entries: Subtitle entries with timing info
        tts_output_dir: Directory containing TTS audio files
        output_path: Output audio file path
        video_duration: Total video duration
        overlap_strategy: Strategy for handling overlaps ("truncate", "speed", "hybrid", "none")
        max_speed: Maximum speed multiplier for speed strategy
    """
    segments = []
    missing_files = []

    for entry in srt_entries:
        # TTS files are named like "0001.wav"
        tts_file = tts_output_dir / f"{entry.index:04d}.wav"
        if tts_file.exists():
            start_time = parse_timestamp(entry.start_time)
            duration = get_audio_duration(tts_file)
            segments.append(
                AudioSegment(path=tts_file, start_time=start_time, duration=duration)
            )
        else:
            missing_files.append(f"{entry.index:04d}.wav")

    if missing_files:
        print(
            f"[Compose] ⚠ Missing {len(missing_files)} TTS files: {missing_files[:5]}..."
        )

    if not segments:
        raise RuntimeError("No TTS audio files found")

    print(f"[Compose] Found {len(segments)} TTS audio segments")
    return concat_audio_with_timing(
        segments,
        output_path,
        video_duration,
        srt_entries=srt_entries,
        overlap_strategy=overlap_strategy,
        max_speed=max_speed,
    )


def mix_audio_tracks(
    original_audio: Path,
    tts_audio: Path,
    output_path: Path,
    mode: str = "mix",
    original_volume: float = 0.3,
    tts_volume: float = 1.0,
    bgm_audio: Path | None = None,
) -> Path:
    """
    Mix audio tracks based on mode.

    Modes:
        - replace: Only TTS audio
        - mix: TTS + lowered original audio
        - bgm: TTS + BGM only (requires vocal separation first)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "replace":
        # Just copy TTS audio
        shutil.copy(tts_audio, output_path)
        return output_path

    elif mode == "mix":
        # Mix TTS with lowered original
        # Use loudnorm on TTS to boost perceived volume before mixing
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(tts_audio),
            "-i",
            str(original_audio),
            "-filter_complex",
            f"[0]loudnorm=I=-14:TP=-1:LRA=11,volume={tts_volume}[tts];"
            f"[1]volume={original_volume}[orig];"
            f"[tts][orig]amix=inputs=2:duration=first[out]",
            "-map",
            "[out]",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
        return output_path

    elif mode == "bgm":
        if not bgm_audio or not bgm_audio.exists():
            raise RuntimeError(
                "BGM audio required for 'bgm' mode. Run vocal separation first."
            )

        # Mix TTS with BGM only
        # Use loudnorm to boost TTS before mixing
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(tts_audio),
            "-i",
            str(bgm_audio),
            "-filter_complex",
            f"[0]loudnorm=I=-14:TP=-1:LRA=11,volume={tts_volume}[tts];"
            f"[1]volume=1.0[bgm];"
            f"[tts][bgm]amix=inputs=2:duration=first[out]",
            "-map",
            "[out]",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
        return output_path

    else:
        raise ValueError(f"Unknown mix mode: {mode}")


def compose_video(
    video_path: Path,
    audio_path: Path,
    subtitle_path: Path | None,
    output_path: Path,
    burn_subtitles: bool = False,
) -> Path:
    """
    Compose final video with new audio and optional subtitles.

    Args:
        video_path: Original video file
        audio_path: Mixed/replaced audio track
        subtitle_path: SRT subtitle file (optional)
        output_path: Output video file
        burn_subtitles: Whether to burn subtitles into video
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if burn_subtitles and subtitle_path and subtitle_path.exists():
        # Burn subtitles into video
        # Need to escape special characters in path for ffmpeg filter
        sub_path_escaped = str(subtitle_path).replace("\\", "/").replace(":", "\\:")

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-vf",
            f"subtitles='{sub_path_escaped}'",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
    else:
        # Just replace audio, optionally add subtitle stream
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
        ]

        if subtitle_path and subtitle_path.exists():
            cmd.extend(["-i", str(subtitle_path)])
            cmd.extend(
                [
                    "-map",
                    "0:v",
                    "-map",
                    "1:a",
                    "-map",
                    "2:s",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-c:s",
                    "mov_text",  # For MP4 subtitle format
                ]
            )
        else:
            cmd.extend(
                [
                    "-map",
                    "0:v",
                    "-map",
                    "1:a",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                ]
            )

        cmd.extend(["-shortest", str(output_path)])

    subprocess.run(cmd, check=True)
    return output_path


def full_compose(
    video_path: Path,
    srt_path: Path,
    tts_output_dir: Path,
    output_path: Path,
    temp_dir: Path,
    audio_mode: str = "mix",
    original_volume: float = 0.3,
    tts_volume: float = 1.0,
    burn_subtitles: bool = False,
    separate_vocals_first: bool = False,
    separation_method: str = "demucs",
    overlap_strategy: str = "hybrid",
    max_speed: float = 1.5,
) -> Path:
    """
    Full composition pipeline.

    Args:
        video_path: Original video
        srt_path: Translated SRT file
        tts_output_dir: Directory with TTS audio files
        output_path: Output video path
        temp_dir: Temporary directory for intermediate files
        audio_mode: "replace", "mix", or "bgm"
        original_volume: Volume level for original audio in mix mode (0.0-1.0)
        tts_volume: Volume level for TTS audio (0.0-1.0)
        burn_subtitles: Whether to burn subtitles into video
        separate_vocals_first: Whether to separate vocals from original
        separation_method: "demucs" or "spleeter"
        overlap_strategy: Strategy for TTS overlap ("truncate", "speed", "hybrid", "none")
        max_speed: Maximum speed multiplier for TTS speedup (default 1.5x)
    """
    if not check_ffmpeg():
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")

    temp_dir.mkdir(parents=True, exist_ok=True)

    # Parse SRT for timing info
    entries = parse_srt(srt_path)
    video_duration = get_video_duration(video_path)

    print(f"[Compose] Video duration: {video_duration:.2f}s")
    print(f"[Compose] Subtitle entries: {len(entries)}")
    print(f"[Compose] Overlap strategy: {overlap_strategy} (max speed: {max_speed}x)")

    # Step 1: Build TTS audio track with timing
    print("[Compose] Building TTS audio track...")
    tts_track = temp_dir / "tts_track.wav"
    build_tts_audio_track(
        entries,
        tts_output_dir,
        tts_track,
        video_duration,
        overlap_strategy=overlap_strategy,
        max_speed=max_speed,
    )

    # Step 2: Handle original audio based on mode
    if audio_mode in ("mix", "bgm"):
        print("[Compose] Extracting original audio...")
        original_audio = temp_dir / "original_audio.wav"
        extract_audio_from_video(video_path, original_audio)

        bgm_audio = None
        if audio_mode == "bgm" or separate_vocals_first:
            print(f"[Compose] Separating vocals using {separation_method}...")
            try:
                _, bgm_audio = separate_vocals(
                    original_audio, temp_dir / "separated", method=separation_method
                )
            except RuntimeError as e:
                print(f"[WARNING] Vocal separation failed: {e}")
                print("[WARNING] Falling back to mix mode")
                audio_mode = "mix"

        # Step 3: Mix audio tracks
        print(f"[Compose] Mixing audio (mode={audio_mode})...")
        mixed_audio = temp_dir / "mixed_audio.wav"
        mix_audio_tracks(
            original_audio=original_audio,
            tts_audio=tts_track,
            output_path=mixed_audio,
            mode=audio_mode,
            original_volume=original_volume,
            tts_volume=tts_volume,
            bgm_audio=bgm_audio,
        )
        final_audio = mixed_audio
    else:
        final_audio = tts_track

    # Step 4: Compose final video
    print("[Compose] Composing final video...")
    compose_video(
        video_path=video_path,
        audio_path=final_audio,
        subtitle_path=srt_path if not burn_subtitles else srt_path,
        output_path=output_path,
        burn_subtitles=burn_subtitles,
    )

    print(f"[Compose] Done! Output: {output_path}")
    return output_path
