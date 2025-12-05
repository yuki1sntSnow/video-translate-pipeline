"""
Video Translation Pipeline: download → asr → translate → tts → compose

Usage:
  # Full pipeline
  python pipeline.py --url "https://youtube.com/watch?v=xxx"

  # Start from specific stage (use existing data folder)
  python pipeline.py --start-from asr --data-dir data/20251205_120000
  python pipeline.py --start-from translate --data-dir data/20251205_120000
  python pipeline.py --start-from tts --data-dir data/20251205_120000
  python pipeline.py --start-from compose --data-dir data/20251205_120000
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from download import load_env, download_media
from translate import translate_subs

# ===== Stages =====
STAGES = ["download", "asr", "translate", "tts", "compose"]


class TeeLogger:
    """Write to both stdout and a log file."""

    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_path, "w", encoding="utf-8")
        self.log_file.write(
            f"Pipeline Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        self.log_file.write("=" * 60 + "\n\n")
        self.log_file.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def find_files(directory: Path, pattern: str) -> list[Path]:
    """Find files matching pattern in directory."""
    return sorted(directory.glob(pattern))


def run_tts_batch(input_dir: Path, output_dir: Path, temp_dir: Path, speaker: str):
    """Run TTS batch processing (single speaker)."""
    from tts import run_batch_tts
    import argparse as ap
    import config as cfg

    tts_args = ap.Namespace(
        input_dir=str(input_dir),
        ref_dir=cfg.REF_DIR,
        output_dir=str(output_dir),
        temp_dir=str(temp_dir),
        speaker=speaker,
        dac_checkpoint=cfg.TTS_DAC_CHECKPOINT,
        t2s_checkpoint=cfg.TTS_T2S_CHECKPOINT,
        device=cfg.TTS_DEVICE,
        half=cfg.TTS_HALF,
        compile=cfg.TTS_COMPILE,
        temperature=cfg.TTS_TEMPERATURE,
        top_p=cfg.TTS_TOP_P,
        repetition_penalty=cfg.TTS_REPETITION_PENALTY,
        seed=cfg.TTS_SEED,
    )
    run_batch_tts(tts_args)


def run_tts_multi_speaker(
    input_dir: Path,
    output_dir: Path,
    temp_dir: Path,
    speaker_mapping: dict[str, str] | None = None,
    fallback_speaker: str | None = None,
):
    """Run TTS batch processing for multiple speakers."""
    from tts import run_multi_speaker_tts
    import config as cfg

    run_multi_speaker_tts(
        input_dir=input_dir,
        ref_dir=Path(cfg.REF_DIR),
        output_dir=output_dir,
        temp_dir=temp_dir,
        dac_checkpoint=cfg.TTS_DAC_CHECKPOINT,
        t2s_checkpoint=cfg.TTS_T2S_CHECKPOINT,
        device=cfg.TTS_DEVICE,
        half=cfg.TTS_HALF,
        compile=cfg.TTS_COMPILE,
        temperature=cfg.TTS_TEMPERATURE,
        top_p=cfg.TTS_TOP_P,
        repetition_penalty=cfg.TTS_REPETITION_PENALTY,
        seed=cfg.TTS_SEED,
        speaker_mapping=speaker_mapping,
        fallback_speaker=fallback_speaker,
    )


def parse_speaker_mapping(mapping_str: str | None) -> dict[str, str] | None:
    """Parse speaker mapping string.

    Format: 'SPEAKER_00:host,SPEAKER_01:guest'
    Returns: {'SPEAKER_00': 'host', 'SPEAKER_01': 'guest'}
    """
    if not mapping_str:
        return None

    result = {}
    for pair in mapping_str.split(","):
        pair = pair.strip()
        if ":" in pair:
            asr_name, ref_name = pair.split(":", 1)
            result[asr_name.strip()] = ref_name.strip()

    return result if result else None


def build_parser() -> argparse.ArgumentParser:
    import config as cfg

    parser = argparse.ArgumentParser(
        description="Video Translation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Stages: {' → '.join(STAGES)}",
    )

    # Pipeline control
    parser.add_argument("--url", default=cfg.VIDEO_URL, help="Video URL to download")
    parser.add_argument("--start-from", choices=STAGES, default="download", help="Start from stage")
    parser.add_argument("--stop-after", choices=STAGES, help="Stop after stage")

    # Input files (for starting from middle)
    parser.add_argument("--video", default=cfg.LOCAL_VIDEO, help="Video file path (optional, auto-detected)")
    parser.add_argument("--srt", default=cfg.LOCAL_SRT, help="SRT file path (optional, auto-detected)")
    parser.add_argument("--data-dir", help="Data directory for this run (auto-created if not specified)")

    # Download options
    parser.add_argument("--subs", default=cfg.SUB_LANGS, help="Subtitle languages")
    parser.add_argument("--auto-subs", action="store_true", default=cfg.AUTO_SUBS)
    parser.add_argument("--subs-only", action="store_true", default=cfg.SUBS_ONLY)
    parser.add_argument("--no-download-subs", action="store_true", default=cfg.NO_DOWNLOAD_SUBS, help="Skip downloading subtitles")
    parser.add_argument("--cookies-file", default=cfg.COOKIES_FILE)
    parser.add_argument("--proxy", default=cfg.PROXY)

    # Translation options
    parser.add_argument("--target-language", "-t", default=cfg.TARGET_LANGUAGE)
    parser.add_argument("--chunk-size", type=int, default=cfg.TRANSLATE_CHUNK_SIZE, help="Lines per translation chunk")
    parser.add_argument("--max-concurrent", type=int, default=cfg.TRANSLATE_MAX_CONCURRENT, help="Max concurrent tasks")

    # ASR options
    parser.add_argument("--asr-model", default=cfg.ASR_MODEL, help="Whisper model")
    parser.add_argument("--asr-device", default=cfg.ASR_DEVICE, help="Device (cuda/cpu)")
    parser.add_argument("--asr-batch-size", type=int, default=cfg.ASR_BATCH_SIZE)
    parser.add_argument("--asr-language", default=cfg.ASR_LANGUAGE or None, help="Force ASR language")
    parser.add_argument("--skip-asr", action="store_true", default=cfg.SKIP_ASR, help="Skip ASR")

    # Multi-speaker options
    parser.add_argument("--multi-speaker", action="store_true", default=cfg.MULTI_SPEAKER, help="Enable multi-speaker mode")
    parser.add_argument("--hf-token", default=cfg.HF_TOKEN, help="HuggingFace token")
    parser.add_argument("--min-speakers", type=int, default=cfg.ASR_MIN_SPEAKERS or None, help="Minimum speakers (0=auto)")
    parser.add_argument("--max-speakers", type=int, default=cfg.ASR_MAX_SPEAKERS or None, help="Maximum speakers (0=auto)")
    parser.add_argument("--speaker-mapping", default=cfg.SPEAKER_MAPPING or None, help="Map speakers to refs, e.g. 'SPEAKER_00:host,SPEAKER_01:guest'")

    # TTS options
    parser.add_argument("--speaker", default=cfg.TTS_SPEAKER)

    # Composition options
    parser.add_argument("--audio-mode", choices=["replace", "mix", "bgm"], default=cfg.AUDIO_MODE)
    parser.add_argument("--original-volume", type=float, default=cfg.ORIGINAL_VOLUME)
    parser.add_argument("--burn-subs", action="store_true", default=cfg.BURN_SUBS)
    parser.add_argument("--overlap-strategy", choices=["truncate", "speed", "hybrid", "none"], default=cfg.OVERLAP_STRATEGY)
    parser.add_argument("--max-tts-speed", type=float, default=cfg.MAX_TTS_SPEED)
    parser.add_argument("--max-line-width", type=float, default=22.0, help="Max subtitle line width")
    parser.add_argument("--max-subtitle-lines", type=int, default=2, help="Max lines per subtitle")

    # Subtitle style options
    parser.add_argument("--subtitle-font", default=cfg.SUB_FONT_NAME)
    parser.add_argument("--subtitle-font-size", type=int, default=cfg.SUB_FONT_SIZE)
    parser.add_argument("--subtitle-primary-color", default=cfg.SUB_PRIMARY_COLOR)
    parser.add_argument("--subtitle-outline-color", default=cfg.SUB_OUTLINE_COLOR)
    parser.add_argument("--subtitle-back-color", default=cfg.SUB_BACK_COLOR)
    parser.add_argument("--subtitle-outline-width", type=int, default=int(cfg.SUB_OUTLINE))
    parser.add_argument("--subtitle-shadow-depth", type=int, default=int(cfg.SUB_SHADOW))
    parser.add_argument("--subtitle-margin-v", type=int, default=cfg.SUB_MARGIN_V)

    # Other
    parser.add_argument("--debug", action="store_true", default=cfg.DEBUG, help="Save intermediate files")

    return parser


def should_run_stage(stage: str, start_from: str, stop_after: str | None) -> bool:
    """Check if a stage should run based on start/stop settings."""
    start_idx = STAGES.index(start_from)
    stage_idx = STAGES.index(stage)

    if stage_idx < start_idx:
        return False

    if stop_after:
        stop_idx = STAGES.index(stop_after)
        if stage_idx > stop_idx:
            return False

    return True


def main(argv: list[str] | None = None) -> None:
    load_env()

    # 重新加载 config 模块以获取最新的环境变量
    import importlib
    import config
    importlib.reload(config)

    args = build_parser().parse_args(argv)

    # 确定数据目录
    if args.data_dir:
        # 使用指定的数据目录（继续之前的任务）
        data_dir = Path(args.data_dir)
        run_timestamp = data_dir.name
        if not data_dir.exists():
            print(f"[ERROR] Data directory not found: {data_dir}")
            sys.exit(1)
    else:
        # 创建新的数据目录
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = Path(os.getenv("DATA_BASE_DIR", "data")) / run_timestamp
        data_dir.mkdir(parents=True, exist_ok=True)

    # 临时文件目录
    temp_dir = data_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件
    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_path = log_dir / f"pipeline_{run_timestamp}.log"
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print(f"[Pipeline] Log: {log_path}")
    print(f"[Pipeline] Data dir: {data_dir}")
    print(f"[Pipeline] Start from: {args.start_from}")
    if args.stop_after:
        print(f"[Pipeline] Stop after: {args.stop_after}")

    # Track state across stages
    video_path: Path | None = Path(args.video) if args.video else None
    srt_path: Path | None = Path(args.srt) if args.srt else None
    translated_srt: Path | None = None

    # 如果从中间阶段开始，自动检测已有文件
    if args.start_from != "download" and not video_path:
        video_files = find_files(data_dir, "*.mp4")
        # 排除 *_final.mp4
        video_files = [f for f in video_files if not f.stem.endswith("_final")]
        if video_files:
            video_path = video_files[0]
            print(f"[Pipeline] Auto-detected video: {video_path.name}")

    if args.start_from not in ["download", "asr"] and not srt_path:
        # 尝试查找原始字幕（不含 _translated）
        srt_files = find_files(data_dir, "*.srt")
        original_srts = [f for f in srt_files if "_translated" not in f.stem]
        if original_srts:
            srt_path = original_srts[0]
            print(f"[Pipeline] Auto-detected SRT: {srt_path.name}")

    if args.start_from in ["tts", "compose"]:
        # 尝试查找翻译后的字幕
        translated_files = find_files(data_dir, "*_translated.srt")
        if translated_files:
            translated_srt = translated_files[0]
            print(f"[Pipeline] Auto-detected translated SRT: {translated_srt.name}")

    # TTS 输入/输出目录
    tts_input_dir = temp_dir / "tts_input"
    tts_output_dir = temp_dir / "tts_output"

    # ==================== Stage 1: Download ====================
    if should_run_stage("download", args.start_from, args.stop_after):
        print(f"\n{'='*60}")
        print(f"[Stage 1/5] DOWNLOAD")
        print(f"{'='*60}\n")

        if not args.url:
            print("[ERROR] --url required for download stage")
            sys.exit(1)

        print(f"URL: {args.url}")
        print(f"Output: {data_dir}")
        if args.no_download_subs:
            print(f"Subtitles: disabled (will use ASR)\n")
        else:
            print(f"Subtitles: {args.subs}\n")

        download_media(
            url=args.url,
            output_dir=data_dir,
            sub_langs=args.subs,
            proxy=args.proxy,
            cookies_from_browser=os.getenv("COOKIES_FROM_BROWSER"),
            cookies_file=args.cookies_file,
            extractor_args=os.getenv("EXTRACTOR_ARGS"),
            format_selector=os.getenv("YTDLP_FORMAT", "bestvideo+bestaudio/best"),
            allow_auto_subs=args.auto_subs,
            skip_download=args.subs_only,
            skip_subs=args.no_download_subs,
        )

        # 查找下载的文件
        video_files = find_files(data_dir, "*.mp4")
        srt_files = find_files(data_dir, "*.srt")

        if video_files:
            video_path = video_files[0]
            print(f"\n[Download] Video: {video_path.name}")

        if srt_files:
            srt_path = srt_files[0]
            print(f"[Download] SRT: {srt_path.name}")

        if args.stop_after == "download":
            print("\n[Pipeline] Stopped after download stage.")

    # ==================== Stage 2: ASR ====================
    if should_run_stage("asr", args.start_from, args.stop_after):
        print(f"\n{'='*60}")
        print(f"[Stage 2/5] ASR (Speech Recognition)")
        print(f"{'='*60}\n")

        # Check if we need ASR (no SRT available)
        need_asr = not srt_path or not srt_path.exists()

        if args.skip_asr:
            print("[ASR] Skipped (--skip-asr flag)")
        elif not need_asr:
            print(f"[ASR] Skipped - SRT already available: {srt_path}")
        else:
            if not video_path or not video_path.exists():
                # Try to find video in data_dir
                video_files = find_files(data_dir, "*.mp4")
                video_files = [f for f in video_files if not f.stem.endswith("_final")]
                if video_files:
                    video_path = video_files[0]
                else:
                    print(f"[ERROR] Video file required for ASR: {video_path}")
                    sys.exit(1)

            print(f"Video: {video_path}")
            print(f"Model: {args.asr_model}")
            print(f"Device: {args.asr_device}")
            print(f"Multi-speaker: {args.multi_speaker}")
            if args.multi_speaker:
                print(f"  Min speakers: {args.min_speakers or 'auto'}")
                print(f"  Max speakers: {args.max_speakers or 'auto'}")
            print(f"Output: {data_dir}\n")

            from asr import transcribe_video

            try:
                srt_path = transcribe_video(
                    video_path=video_path,
                    output_dir=data_dir,
                    device=args.asr_device,
                    batch_size=args.asr_batch_size,
                    model_name=args.asr_model,
                    language=args.asr_language,
                    keep_audio=True,
                    multi_speaker=args.multi_speaker,
                    hf_token=args.hf_token,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                )
                print(f"\n[ASR] ✓ Done: {srt_path}")
            except Exception as e:
                print(f"[ERROR] ASR failed: {e}")
                sys.exit(1)

        if args.stop_after == "asr":
            print("\n[Pipeline] Stopped after ASR stage.")

    # ==================== Stage 3: Translate ====================
    if should_run_stage("translate", args.start_from, args.stop_after):
        print(f"\n{'='*60}")
        print(f"[Stage 3/5] TRANSLATE (LLM chunked translation)")
        print(f"{'='*60}\n")

        if not srt_path or not srt_path.exists():
            print(f"[ERROR] SRT file required: {srt_path}")
            sys.exit(1)

        # 翻译后的字幕文件命名: {name}_translated.srt
        output_name = f"{srt_path.stem}_translated.srt"
        translated_srt = data_dir / output_name

        print(f"Input:  {srt_path}")
        print(f"Target: {args.target_language}")
        print(f"Chunk size: {args.chunk_size} lines")
        print(f"Max concurrent: {args.max_concurrent}")
        print(f"Output: {translated_srt}\n")

        # Temp dir for intermediate files (only in debug mode)
        translate_temp_dir = (temp_dir / "translate") if args.debug else None

        try:
            translate_subs(
                args.target_language,
                srt_path,
                translated_srt,
                chunk_size=args.chunk_size,
                temp_dir=translate_temp_dir,
                max_concurrent=args.max_concurrent,
            )
            print(f"\n[Translate] ✓ Done: {translated_srt}")
        except Exception as e:
            print(f"[ERROR] Translation failed: {e}")
            sys.exit(1)

        if args.stop_after == "translate":
            print("\n[Pipeline] Stopped after translate stage.")

    # ==================== Stage 4: TTS ====================
    if should_run_stage("tts", args.start_from, args.stop_after):
        print(f"\n{'='*60}")
        print(f"[Stage 4/5] TTS")
        print(f"{'='*60}\n")

        # Use translated SRT if available, otherwise use input SRT
        tts_srt = translated_srt or srt_path
        if not tts_srt or not tts_srt.exists():
            print(f"[ERROR] SRT file required for TTS: {tts_srt}")
            sys.exit(1)

        print(f"SRT: {tts_srt}")
        print(f"Multi-speaker: {args.multi_speaker}")
        if args.multi_speaker:
            print(
                f"Speaker mapping: {args.speaker_mapping or 'auto (ASR speaker names)'}"
            )
        else:
            print(f"Speaker: {args.speaker}")
        print(f"Output: {tts_output_dir}\n")

        # Import srt_utils functions
        from srt_utils import (
            parse_srt,
            entries_to_txt_files,
            entries_to_txt_files_by_speaker,
            get_speakers_from_entries,
        )

        # Parse SRT
        entries = parse_srt(tts_srt)
        print(f"[TTS] Loaded {len(entries)} entries")

        # Clean old output
        tts_output_dir.mkdir(parents=True, exist_ok=True)
        for old_file in tts_output_dir.glob("*.wav"):
            old_file.unlink()

        tts_temp_dir = temp_dir / "tts"

        if args.multi_speaker:
            # Multi-speaker mode: group by speaker
            speakers = get_speakers_from_entries(entries)
            print(f"[TTS] Found {len(speakers)} speaker(s): {speakers}")

            # Export to speaker subdirs
            speaker_files = entries_to_txt_files_by_speaker(entries, tts_input_dir)
            total_files = sum(len(files) for files in speaker_files.values())
            print(f"[TTS] Exported {total_files} text files")

            # Parse speaker mapping
            speaker_mapping = parse_speaker_mapping(args.speaker_mapping)
            if speaker_mapping:
                print(f"[TTS] Speaker mapping: {speaker_mapping}")
            print(f"[TTS] Fallback speaker: {args.speaker}")

            try:
                run_tts_multi_speaker(
                    tts_input_dir,
                    tts_output_dir,
                    tts_temp_dir,
                    speaker_mapping=speaker_mapping,
                    fallback_speaker=args.speaker,
                )
                print(f"\n[TTS] ✓ Done: {tts_output_dir}")
            except Exception as e:
                print(f"[ERROR] TTS failed: {e}")
                sys.exit(1)
        else:
            # Single speaker mode
            txt_files = entries_to_txt_files(entries, tts_input_dir)
            print(f"[TTS] Exported {len(txt_files)} text files")

            try:
                run_tts_batch(tts_input_dir, tts_output_dir, tts_temp_dir, args.speaker)
                print(f"\n[TTS] ✓ Done: {tts_output_dir}")
            except Exception as e:
                print(f"[ERROR] TTS failed: {e}")
                sys.exit(1)

        if args.stop_after == "tts":
            print("\n[Pipeline] Stopped after TTS stage.")

    # ==================== Stage 5: Compose ====================
    if should_run_stage("compose", args.start_from, args.stop_after):
        print(f"\n{'='*60}")
        print(f"[Stage 5/5] COMPOSE")
        print(f"{'='*60}\n")

        if not video_path or not video_path.exists():
            # Try to find video in data_dir
            video_files = find_files(data_dir, "*.mp4")
            video_files = [f for f in video_files if not f.stem.endswith("_final")]
            if video_files:
                video_path = video_files[0]
            else:
                print(f"[ERROR] Video file required: {video_path}")
                sys.exit(1)

        compose_srt = translated_srt or srt_path
        if not compose_srt or not compose_srt.exists():
            print(f"[ERROR] SRT file required: {compose_srt}")
            sys.exit(1)

        # 检查 TTS 输出目录是否存在
        if not tts_output_dir.exists() or not list(tts_output_dir.glob("*.wav")):
            print(f"[ERROR] TTS audio files not found in: {tts_output_dir}")
            print(f"[ERROR] Please run TTS stage first or check the temp/tts_output directory.")
            sys.exit(1)

        from compose import full_compose
        from srt_utils import SubtitleStyle

        # 最终输出文件: {name}_final.mp4
        output_name = f"{video_path.stem}_final.mp4"
        output_path = data_dir / output_name

        # Build subtitle style from args
        subtitle_style = SubtitleStyle(
            font_name=args.subtitle_font,
            font_size=args.subtitle_font_size,
            primary_color=args.subtitle_primary_color,
            outline_color=args.subtitle_outline_color,
            back_color=args.subtitle_back_color,
            outline=args.subtitle_outline_width,
            shadow=args.subtitle_shadow_depth,
            margin_v=args.subtitle_margin_v,
        )

        print(f"Video:    {video_path}")
        print(f"SRT:      {compose_srt}")
        print(f"TTS Dir:  {tts_output_dir}")
        print(f"Mode:     {args.audio_mode}")
        print(f"Overlap:  {args.overlap_strategy}")
        print(f"Subtitle: {args.subtitle_font} {args.subtitle_font_size}pt")
        print(f"Output:   {output_path}\n")

        try:
            full_compose(
                video_path=video_path,
                srt_path=compose_srt,
                tts_output_dir=tts_output_dir,
                output_path=output_path,
                temp_dir=temp_dir / "compose",
                audio_mode=args.audio_mode,
                original_volume=args.original_volume,
                tts_volume=float(os.getenv("TTS_VOLUME", "1.2")),
                burn_subtitles=args.burn_subs,
                overlap_strategy=args.overlap_strategy,
                max_speed=args.max_tts_speed,
                max_line_width=args.max_line_width,
                max_lines=args.max_subtitle_lines,
                subtitle_style=subtitle_style,
            )
            print(f"\n[Compose] ✓ Done: {output_path}")
        except Exception as e:
            print(f"[ERROR] Composition failed: {e}")
            sys.exit(1)

    # ==================== Summary ====================
    print(f"\n{'='*60}")
    print(f"Pipeline Complete!")
    print(f"{'='*60}")
    print(f"  Data dir:   {data_dir}")
    print(f"  Video:      {video_path}")
    print(f"  SRT:        {translated_srt or srt_path}")
    print(f"  Output:     {data_dir / f'{video_path.stem}_final.mp4' if video_path else 'N/A'}")
    print(f"  Log:        {log_path}")

    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main(sys.argv[1:])
