"""
Video Translation Pipeline

Stages:
  1. download  - Download video and subtitles from URL
  2. translate - Translate subtitles (with LLM fix + translate)
  3. tts       - Generate TTS audio from translated subtitles
  4. compose   - Compose final video with TTS audio

Usage:
  # Full pipeline
  python pipeline.py --url "https://youtube.com/watch?v=xxx"

  # Start from specific stage
  python pipeline.py --start-from translate --video video.mp4 --srt video.en.srt
  python pipeline.py --start-from tts --video video.mp4 --srt translate/video.Chinese.srt
  python pipeline.py --start-from compose --video video.mp4 --srt translate/video.Chinese.srt
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from download import load_env, download_media
from translate import translate_subs
from srt_utils import srt_to_tts_input


# ===== Stages =====
STAGES = ["download", "translate", "tts", "compose"]


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
    """Run TTS batch processing."""
    from tts import run_batch_tts
    import argparse as ap

    tts_args = ap.Namespace(
        input_dir=str(input_dir),
        ref_dir="ref",
        output_dir=str(output_dir),
        temp_dir=str(temp_dir),
        speaker=speaker,
        dac_checkpoint=os.getenv(
            "TTS_DAC_CHECKPOINT", "checkpoints/openaudio-s1-mini/codec.pth"
        ),
        t2s_checkpoint=os.getenv("TTS_T2S_CHECKPOINT", "checkpoints/openaudio-s1-mini"),
        device=os.getenv("TTS_DEVICE", "cuda"),
        half=os.getenv("TTS_HALF", "0") not in ("0", "", "false", "False"),
        compile=os.getenv("TTS_COMPILE", "0") not in ("0", "", "false", "False"),
        temperature=float(os.getenv("TTS_TEMPERATURE", "0.8")),
        top_p=float(os.getenv("TTS_TOP_P", "0.8")),
        repetition_penalty=float(os.getenv("TTS_REPETITION_PENALTY", "1.1")),
        seed=int(os.getenv("TTS_SEED", "42")),
    )
    run_batch_tts(tts_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Video Translation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Stages: {' -> '.join(STAGES)}

Examples:
  # Full pipeline from URL
  python pipeline.py --url "https://youtube.com/watch?v=xxx"

  # Start from translate (need video + original SRT)
  python pipeline.py --start-from translate --video video.mp4 --srt video.en.srt

  # Start from TTS (need video + translated SRT)
  python pipeline.py --start-from tts --video video.mp4 --srt translate/video.Chinese.srt

  # Start from compose (need video + translated SRT, TTS already done)
  python pipeline.py --start-from compose --video video.mp4 --srt translate/video.Chinese.srt
        """,
    )

    # Pipeline control
    parser.add_argument(
        "--url", default=os.getenv("VIDEO_URL"), help="Video URL to download"
    )
    parser.add_argument(
        "--start-from", choices=STAGES, default="download", help="Start from stage"
    )
    parser.add_argument("--stop-after", choices=STAGES, help="Stop after stage")

    # Input files (for starting from middle)
    parser.add_argument(
        "--video", default=os.getenv("LOCAL_VIDEO"), help="Video file path"
    )
    parser.add_argument("--srt", default=os.getenv("LOCAL_SRT"), help="SRT file path")

    # Directories
    parser.add_argument(
        "--download-dir", default=os.getenv("DOWNLOAD_DIR", "downloads")
    )
    parser.add_argument(
        "--translate-dir", default=os.getenv("TRANSLATE_DIR", "translate")
    )
    parser.add_argument("--tts-input-dir", default=os.getenv("TTS_INPUT_DIR", "input"))
    parser.add_argument(
        "--tts-output-dir", default=os.getenv("TTS_OUTPUT_DIR", "output")
    )
    parser.add_argument(
        "--final-output-dir", default=os.getenv("FINAL_OUTPUT_DIR", "final")
    )

    # Download options
    parser.add_argument(
        "--subs", default=os.getenv("SUB_LANGS", "en"), help="Subtitle languages"
    )
    parser.add_argument(
        "--auto-subs",
        action="store_true",
        default=os.getenv("AUTO_SUBS", "1") not in ("0", "", "false"),
    )
    parser.add_argument(
        "--subs-only",
        action="store_true",
        default=os.getenv("SUBS_ONLY", "0") not in ("0", "", "false"),
    )
    parser.add_argument("--cookies-file", default=os.getenv("COOKIES_FILE"))
    parser.add_argument("--proxy", default=os.getenv("PROXY"))

    # Translation options
    parser.add_argument(
        "--target-language", "-t", default=os.getenv("TARGET_LANGUAGE", "Chinese")
    )

    # TTS options
    parser.add_argument("--speaker", default=os.getenv("TTS_SPEAKER", "cxk"))

    # Composition options
    parser.add_argument(
        "--audio-mode",
        choices=["replace", "mix", "bgm"],
        default=os.getenv("AUDIO_MODE", "replace"),
    )
    parser.add_argument(
        "--original-volume",
        type=float,
        default=float(os.getenv("ORIGINAL_VOLUME", "0.2")),
    )
    parser.add_argument(
        "--burn-subs",
        action="store_true",
        default=os.getenv("BURN_SUBS", "1") not in ("0", "", "false"),
    )
    parser.add_argument(
        "--overlap-strategy",
        choices=["truncate", "speed", "hybrid", "none"],
        default=os.getenv("OVERLAP_STRATEGY", "hybrid"),
    )
    parser.add_argument(
        "--max-tts-speed", type=float, default=float(os.getenv("MAX_TTS_SPEED", "1.3"))
    )

    # Other
    parser.add_argument(
        "--isolate-run",
        action="store_true",
        default=os.getenv("ISOLATE_RUN", "0") not in ("0", "", "false"),
    )

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
    args = build_parser().parse_args(argv)

    # Timestamp唯一标识本次任务
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 日志文件
    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_path = log_dir / f"pipeline_{run_timestamp}.log"
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print(f"[Pipeline] Log: {log_path}")
    print(f"[Pipeline] Start from: {args.start_from}")
    if args.stop_after:
        print(f"[Pipeline] Stop after: {args.stop_after}")

    # 所有目录均按时间戳分隔
    download_dir = Path(args.download_dir) / run_timestamp
    translate_dir = Path(args.translate_dir) / run_timestamp
    tts_input_dir = Path(args.tts_input_dir) / run_timestamp
    tts_output_dir = Path(args.tts_output_dir) / run_timestamp
    final_output_dir = Path(args.final_output_dir) / run_timestamp
    temp_dir = Path(os.getenv("TEMP_DIR", "temp")) / run_timestamp
    print(f"[Pipeline] Run timestamp: {run_timestamp}")

    # Track state across stages
    video_path: Path | None = Path(args.video) if args.video else None
    srt_path: Path | None = Path(args.srt) if args.srt else None
    translated_srt: Path | None = None

    # ==================== Stage 1: Download ====================
    if should_run_stage("download", args.start_from, args.stop_after):
        print(f"\n{'='*60}")
        print(f"[Stage 1/4] DOWNLOAD")
        print(f"{'='*60}\n")

        if not args.url:
            print("[ERROR] --url required for download stage")
            sys.exit(1)

        print(f"URL: {args.url}")
        print(f"Output: {download_dir}\n")

        download_media(
            url=args.url,
            output_dir=download_dir,
            sub_langs=args.subs,
            proxy=args.proxy,
            cookies_from_browser=os.getenv("COOKIES_FROM_BROWSER"),
            cookies_file=args.cookies_file,
            extractor_args=os.getenv("EXTRACTOR_ARGS"),
            format_selector=os.getenv("YTDLP_FORMAT", "bestvideo+bestaudio/best"),
            allow_auto_subs=args.auto_subs,
            skip_download=args.subs_only,
        )

        # 查找下载的文件
        video_files = find_files(download_dir, "*.mp4")
        srt_files = find_files(download_dir, "*.srt")

        if video_files:
            video_path = video_files[0]
            print(f"\n[Download] Video: {video_path.name}")

        if srt_files:
            srt_path = srt_files[0]
            print(f"[Download] SRT: {srt_path.name}")

        if args.stop_after == "download":
            print("\n[Pipeline] Stopped after download stage.")

    # ==================== Stage 2: Translate ====================
    if should_run_stage("translate", args.start_from, args.stop_after):
        print(f"\n{'='*60}")
        print(f"[Stage 2/4] TRANSLATE (LLM fix + translate)")
        print(f"{'='*60}\n")

        if not srt_path or not srt_path.exists():
            print(f"[ERROR] SRT file required: {srt_path}")
            sys.exit(1)

        translate_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"{srt_path.stem}.{args.target_language}.srt"
        translated_srt = translate_dir / output_name

        print(f"Input:  {srt_path}")
        print(f"Target: {args.target_language}")
        print(f"Output: {translated_srt}\n")

        try:
            translate_subs(args.target_language, srt_path, translated_srt)
            print(f"\n[Translate] ✓ Done: {translated_srt}")
        except Exception as e:
            print(f"[ERROR] Translation failed: {e}")
            sys.exit(1)

        if args.stop_after == "translate":
            print("\n[Pipeline] Stopped after translate stage.")

    # ==================== Stage 3: TTS ====================
    if should_run_stage("tts", args.start_from, args.stop_after):
        print(f"\n{'='*60}")
        print(f"[Stage 3/4] TTS")
        print(f"{'='*60}\n")

        # Use translated SRT if available, otherwise use input SRT
        tts_srt = translated_srt or srt_path
        if not tts_srt or not tts_srt.exists():
            print(f"[ERROR] SRT file required for TTS: {tts_srt}")
            sys.exit(1)

        print(f"SRT: {tts_srt}")
        print(f"Speaker: {args.speaker}")
        print(f"Output: {tts_output_dir}\n")

        # Parse SRT and export to input folder
        entries = srt_to_tts_input(tts_srt, tts_input_dir)
        print(f"[TTS] Extracted {len(entries)} entries")

        # Clean old output
        tts_output_dir.mkdir(parents=True, exist_ok=True)
        for old_file in tts_output_dir.glob("*.wav"):
            old_file.unlink()

        # Run TTS
        tts_temp_dir = temp_dir / "tts"
        try:
            run_tts_batch(tts_input_dir, tts_output_dir, tts_temp_dir, args.speaker)
            print(f"\n[TTS] ✓ Done: {tts_output_dir}")
        except Exception as e:
            print(f"[ERROR] TTS failed: {e}")
            sys.exit(1)

        if args.stop_after == "tts":
            print("\n[Pipeline] Stopped after TTS stage.")

    # ==================== Stage 4: Compose ====================
    if should_run_stage("compose", args.start_from, args.stop_after):
        print(f"\n{'='*60}")
        print(f"[Stage 4/4] COMPOSE")
        print(f"{'='*60}\n")

        if not video_path or not video_path.exists():
            # Try to find video in download dir
            video_files = find_files(download_dir, "*.mp4")
            if video_files:
                video_path = video_files[0]
            else:
                print(f"[ERROR] Video file required: {video_path}")
                sys.exit(1)

        compose_srt = translated_srt or srt_path
        if not compose_srt or not compose_srt.exists():
            print(f"[ERROR] SRT file required: {compose_srt}")
            sys.exit(1)

        from compose import full_compose

        final_output_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"{video_path.stem}.{args.target_language}.mp4"
        output_path = final_output_dir / output_name

        print(f"Video:    {video_path}")
        print(f"SRT:      {compose_srt}")
        print(f"TTS Dir:  {tts_output_dir}")
        print(f"Mode:     {args.audio_mode}")
        print(f"Overlap:  {args.overlap_strategy}")
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
            )
            print(f"\n[Compose] ✓ Done: {output_path}")
        except Exception as e:
            print(f"[ERROR] Composition failed: {e}")
            sys.exit(1)

    # ==================== Summary ====================
    print(f"\n{'='*60}")
    print(f"Pipeline Complete!")
    print(f"{'='*60}")
    print(f"  Video:      {video_path}")
    print(f"  SRT:        {translated_srt or srt_path}")
    print(f"  TTS:        {tts_output_dir}")
    print(f"  Output:     {final_output_dir}")
    print(f"  Log:        {log_path}")

    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main(sys.argv[1:])
